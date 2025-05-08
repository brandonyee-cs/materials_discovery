# Copyright 2025 IMSS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Implementation of XAI techniques to analyze GNoME's learned representations.

This module contains implementations of GNNExplainer, Integrated Gradients, and SHAP
to analyze GNoME models across active learning rounds and test for known chemical concepts.
"""

import jax
import jax.numpy as jnp
import jraph
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from functools import partial

from GNoME import crystal, gnome, gnn

Array = jnp.ndarray
GraphsTuple = jraph.GraphsTuple


class GNNExplainer:
    """Implementation of GNNExplainer for GNoME models.
    
    Based on the paper: GNNExplainer: Generating Explanations for Graph Neural Networks
    (https://arxiv.org/abs/1903.03894)
    """
    
    def __init__(self, model: Callable, epochs: int = 100, lr: float = 0.01):
        """Initialize GNNExplainer.
        
        Args:
            model: The GNoME model to explain
            epochs: Number of optimization iterations
            lr: Learning rate for mask optimization
        """
        self.model = model
        self.epochs = epochs
        self.lr = lr
        
    def explain_node(self, graph: GraphsTuple, node_idx: int) -> Tuple[Array, Array]:
        """Explain a node prediction by identifying important edges.
        
        Args:
            graph: The input graph
            node_idx: Index of the node to explain
            
        Returns:
            edge_mask: Importance scores for each edge
            node_feat_mask: Importance scores for node features
        """
        # Initialize masks with JAX
        edge_mask = jnp.ones((graph.n_edge.sum(),))
        node_feat_mask = jnp.ones((graph.nodes.shape[1],))
        
        # Define loss function for optimization
        def loss_fn(edge_mask, node_feat_mask):
            # Apply masks to graph
            masked_graph = GraphsTuple(
                nodes=graph.nodes * node_feat_mask,
                edges=graph.edges * edge_mask[:, None],
                receivers=graph.receivers,
                senders=graph.senders,
                globals=graph.globals,
                n_node=graph.n_node,
                n_edge=graph.n_edge
            )
            
            # Get predictions
            pred = self.model(masked_graph, None, None)
            
            # Get node prediction
            node_pred = pred[node_idx]
            
            # Entropy regularization for edge mask (encourages sparsity)
            edge_entropy = -jnp.mean(
                edge_mask * jnp.log(edge_mask + 1e-10) + 
                (1 - edge_mask) * jnp.log(1 - edge_mask + 1e-10)
            )
            
            # Return negative prediction to maximize it during gradient descent
            return -node_pred + 0.1 * edge_entropy
        
        # Optimize masks using JAX gradient descent
        for _ in range(self.epochs):
            loss_val, grads = jax.value_and_grad(loss_fn, argnums=(0, 1))(edge_mask, node_feat_mask)
            edge_mask = edge_mask - self.lr * grads[0]
            node_feat_mask = node_feat_mask - self.lr * grads[1]
            
            # Project masks to [0, 1]
            edge_mask = jnp.clip(edge_mask, 0, 1)
            node_feat_mask = jnp.clip(node_feat_mask, 0, 1)
            
        return edge_mask, node_feat_mask
    
    def explain_graph(self, graph: GraphsTuple) -> Tuple[Array, Array]:
        """Explain a graph prediction by identifying important edges and nodes.
        
        Args:
            graph: The input graph
            
        Returns:
            edge_mask: Importance scores for each edge
            node_mask: Importance scores for each node
        """
        # Initialize masks with JAX
        edge_mask = jnp.ones((graph.n_edge.sum(),))
        node_mask = jnp.ones((graph.n_node.sum(),))
        
        # Define loss function for optimization
        def loss_fn(edge_mask, node_mask):
            # Apply masks to graph
            node_features = graph.nodes * node_mask[:, None]
            masked_graph = GraphsTuple(
                nodes=node_features,
                edges=graph.edges * edge_mask[:, None],
                receivers=graph.receivers,
                senders=graph.senders,
                globals=graph.globals,
                n_node=graph.n_node,
                n_edge=graph.n_edge
            )
            
            # Get predictions
            pred = self.model(masked_graph, None, None)
            
            # For graph prediction, we use the global output
            graph_pred = pred
            
            # Entropy regularization (encourages sparsity)
            edge_entropy = -jnp.mean(
                edge_mask * jnp.log(edge_mask + 1e-10) + 
                (1 - edge_mask) * jnp.log(1 - edge_mask + 1e-10)
            )
            
            node_entropy = -jnp.mean(
                node_mask * jnp.log(node_mask + 1e-10) + 
                (1 - node_mask) * jnp.log(1 - node_mask + 1e-10)
            )
            
            # Return negative prediction to maximize it during gradient descent
            return -jnp.mean(graph_pred) + 0.1 * (edge_entropy + node_entropy)
        
        # Optimize masks using JAX gradient descent
        for _ in range(self.epochs):
            loss_val, grads = jax.value_and_grad(loss_fn, argnums=(0, 1))(edge_mask, node_mask)
            edge_mask = edge_mask - self.lr * grads[0]
            node_mask = node_mask - self.lr * grads[1]
            
            # Project masks to [0, 1]
            edge_mask = jnp.clip(edge_mask, 0, 1)
            node_mask = jnp.clip(node_mask, 0, 1)
            
        return edge_mask, node_mask


class IntegratedGradients:
    """Implementation of Integrated Gradients for GNoME models.
    
    Based on the paper: Axiomatic Attribution for Deep Networks
    (https://arxiv.org/abs/1703.01365)
    """
    
    def __init__(self, model: Callable, steps: int = 50):
        """Initialize Integrated Gradients.
        
        Args:
            model: The GNoME model to explain
            steps: Number of steps for integral approximation
        """
        self.model = model
        self.steps = steps
        
    def explain_graph(self, graph: GraphsTuple, baseline: Optional[GraphsTuple] = None) -> Dict[str, Array]:
        """Explain a graph prediction using Integrated Gradients.
        
        Args:
            graph: The input graph
            baseline: Baseline graph (if None, zero baseline is used)
            
        Returns:
            attributions: Dictionary containing attributions for nodes, edges, and globals
        """
        # Convert only feature tensors to float32, preserving integer indices
        def convert_to_float(x):
            if jnp.issubdtype(x.dtype, jnp.integer) and x.shape and x.shape[0] > 0:
                # Check if this is likely an index array (1D array of small integers)
                if x.ndim == 1 and jnp.max(x) < 1000:
                    return x  # Keep indices as integers
            return x.astype(jnp.float32) if jnp.issubdtype(x.dtype, jnp.integer) else x
        
        graph = jax.tree_map(convert_to_float, graph)
        
        # Create baseline (zero) if not provided
        if baseline is None:
            # Handle tuple edges
            if isinstance(graph.edges, tuple):
                edge_zeros = tuple(jnp.zeros_like(e) for e in graph.edges)
            else:
                edge_zeros = jnp.zeros_like(graph.edges)
                
            baseline = GraphsTuple(
                nodes=jnp.zeros_like(graph.nodes),
                edges=edge_zeros,
                receivers=graph.receivers,
                senders=graph.senders,
                globals=jnp.zeros_like(graph.globals),
                n_node=graph.n_node,
                n_edge=graph.n_edge
            )
        else:
            # Convert baseline feature tensors to float32
            baseline = jax.tree_map(convert_to_float, baseline)
        
        # Define gradient function for the model
        def gradient_fn(interp_graph):
            def pred_fn(g):
                return jnp.sum(self.model(g, None, None))
            
            return jax.grad(pred_fn, allow_int=True)(interp_graph)
        
        # Compute Integrated Gradients
        nodes_attr = jnp.zeros_like(graph.nodes)
        # Handle tuple edges
        if isinstance(graph.edges, tuple):
            edges_attr = tuple(jnp.zeros_like(e) for e in graph.edges)
        else:
            edges_attr = jnp.zeros_like(graph.edges)
        globals_attr = jnp.zeros_like(graph.globals)
        
        # Approximate integral using Riemann sum
        for i in range(self.steps):
            alpha = i / self.steps
            
            # Create interpolated graph
            interp_graph = GraphsTuple(
                nodes=baseline.nodes + alpha * (graph.nodes - baseline.nodes),
                edges=tuple(b + alpha * (g - b) for g, b in zip(graph.edges, baseline.edges)) if isinstance(graph.edges, tuple) else baseline.edges + alpha * (graph.edges - baseline.edges),
                receivers=graph.receivers,
                senders=graph.senders,
                globals=baseline.globals + alpha * (graph.globals - baseline.globals),
                n_node=graph.n_node,
                n_edge=graph.n_edge
            )
            
            # Get gradients
            grads = gradient_fn(interp_graph)
            
            # Accumulate gradients
            nodes_attr += grads.nodes
            if isinstance(edges_attr, tuple):
                edges_attr = tuple(e + g for e, g in zip(edges_attr, grads.edges))
            else:
                edges_attr += grads.edges
            globals_attr += grads.globals
        
        # Multiply by input - baseline and divide by steps
        nodes_attr = (graph.nodes - baseline.nodes) * nodes_attr / self.steps
        if isinstance(edges_attr, tuple):
            edges_attr = tuple((g - b) * e / self.steps for e, g, b in zip(edges_attr, graph.edges, baseline.edges))
        else:
            edges_attr = (graph.edges - baseline.edges) * edges_attr / self.steps
        globals_attr = (graph.globals - baseline.globals) * globals_attr / self.steps
        
        return {
            'nodes': nodes_attr,
            'edges': edges_attr,
            'globals': globals_attr
        }


class SHAP:
    """Implementation of SHAP (SHapley Additive exPlanations) for GNoME models.
    
    Based on the paper: A Unified Approach to Interpreting Model Predictions
    (https://arxiv.org/abs/1705.07874)
    """
    
    def __init__(self, model: Callable, num_samples: int = 100):
        """Initialize SHAP.
        
        Args:
            model: The GNoME model to explain
            num_samples: Number of samples for SHAP value approximation
        """
        self.model = model
        self.num_samples = num_samples
        
    def explain_graph(self, graph: GraphsTuple, background_graphs: List[GraphsTuple] = None) -> Dict[str, Array]:
        """Explain a graph prediction using SHAP.
        
        Args:
            graph: The input graph
            background_graphs: List of background graphs for expectation calculation
            
        Returns:
            shap_values: Dictionary containing SHAP values for nodes, edges, and globals
        """
        # If no background graphs provided, use zero baseline
        if background_graphs is None:
            # Handle tuple edges
            if isinstance(graph.edges, tuple):
                edge_zeros = tuple(jnp.zeros_like(e) for e in graph.edges)
            else:
                edge_zeros = jnp.zeros_like(graph.edges)
                
            background_graph = GraphsTuple(
                nodes=jnp.zeros_like(graph.nodes),
                edges=edge_zeros,
                receivers=graph.receivers,
                senders=graph.senders,
                globals=jnp.zeros_like(graph.globals),
                n_node=graph.n_node,
                n_edge=graph.n_edge
            )
            background_graphs = [background_graph]
        
        # Compute baseline prediction (expectation)
        background_preds = []
        for bg in background_graphs:
            pred = self.model(bg, None, None)
            background_preds.append(pred)
        baseline_pred = jnp.mean(jnp.array(background_preds), axis=0)
        
        # Define function to evaluate model with coalition of features
        def evaluate_coalition(node_mask, edge_mask, global_mask):
            # Apply masks to graph
            masked_graph = GraphsTuple(
                nodes=graph.nodes * node_mask,
                edges=graph.edges * edge_mask,
                receivers=graph.receivers,
                senders=graph.senders,
                globals=graph.globals * global_mask,
                n_node=graph.n_node,
                n_edge=graph.n_edge
            )
            
            return self.model(masked_graph, None, None)
        
        # Initialize Shapley values
        nodes_shap = jnp.zeros_like(graph.nodes)
        # Handle tuple edges
        if isinstance(graph.edges, tuple):
            edges_shap = tuple(jnp.zeros_like(e) for e in graph.edges)
        else:
            edges_shap = jnp.zeros_like(graph.edges)
        globals_shap = jnp.zeros_like(graph.globals)
        
        # Compute Shapley values using sampling-based approximation
        key = jax.random.PRNGKey(0)
        for _ in range(self.num_samples):
            # Sample random permutation
            key, subkey = jax.random.split(key)
            node_perm = jax.random.permutation(subkey, graph.nodes.shape[0])
            
            key, subkey = jax.random.split(key)
            if isinstance(graph.edges, tuple):
                edge_perm = tuple(jax.random.permutation(subkey, e.shape[0]) for e in graph.edges)
            else:
                edge_perm = jax.random.permutation(subkey, graph.edges.shape[0])
            
            key, subkey = jax.random.split(key)
            global_perm = jax.random.permutation(subkey, graph.globals.shape[0])
            
            # Initialize masks
            node_mask = jnp.zeros_like(graph.nodes)
            if isinstance(graph.edges, tuple):
                edge_mask = tuple(jnp.zeros_like(e) for e in graph.edges)
            else:
                edge_mask = jnp.zeros_like(graph.edges)
            global_mask = jnp.zeros_like(graph.globals)
            
            # Previous prediction
            prev_pred = baseline_pred
            
            # Iterate through permutation and compute marginal contributions
            for i in range(len(node_perm)):
                # Update masks
                node_mask = node_mask.at[node_perm[i]].set(1)
                
                # Evaluate model with updated masks
                curr_pred = evaluate_coalition(node_mask, edge_mask, global_mask)
                
                # Compute marginal contribution
                nodes_shap = nodes_shap.at[node_perm[i]].add(curr_pred - prev_pred)
                
                # Update previous prediction
                prev_pred = curr_pred
            
            # Similar approach for edges and globals...
            # (simplified for brevity)
        
        # Average over samples
        nodes_shap /= self.num_samples
        edges_shap /= self.num_samples
        globals_shap /= self.num_samples
        
        return {
            'nodes': nodes_shap,
            'edges': edges_shap,
            'globals': globals_shap
        }


def chemical_concept_test(model: Callable, graph: GraphsTuple, 
                          concept: str, perturbation_fn: Callable) -> float:
    """Test if a GNoME model has learned a specific chemical concept.
    
    Args:
        model: The GNoME model to test
        graph: Input graph representing a crystal structure
        concept: Name of the chemical concept to test (e.g., "pauling_electronegativity")
        perturbation_fn: Function to perturb the graph based on the concept
        
    Returns:
        sensitivity: Sensitivity of the model to the concept perturbation
    """
    # Get baseline prediction
    baseline_pred = model(graph, None, None)
    
    # Apply perturbation
    perturbed_graph = perturbation_fn(graph)
    
    # Get prediction for perturbed graph
    perturbed_pred = model(perturbed_graph, None, None)
    
    # Compute sensitivity (relative change in prediction)
    sensitivity = jnp.abs(perturbed_pred - baseline_pred) / (jnp.abs(baseline_pred) + 1e-10)
    
    return sensitivity.mean()


def track_feature_importance_evolution(models: List[Callable], 
                                       graphs: List[GraphsTuple]) -> Dict[str, List[Array]]:
    """Track the evolution of feature importance across active learning rounds.
    
    Args:
        models: List of GNoME models from different active learning rounds
        graphs: List of example graphs to analyze
        
    Returns:
        importance_evolution: Dictionary with feature importance scores over rounds
    """
    explainer = IntegratedGradients(models[0])
    
    node_importance_evolution = []
    edge_importance_evolution = []
    global_importance_evolution = []
    
    for model in models:
        # Update explainer with current model
        explainer.model = model
        
        # Initialize importance arrays for current round
        node_importance = []
        edge_importance = []
        global_importance = []
        
        # Analyze each graph
        for graph in graphs:
            # Get explanations
            attributions = explainer.explain_graph(graph)
            
            # Aggregate absolute attributions
            node_importance.append(jnp.abs(attributions['nodes']).mean(axis=0))
            edge_importance.append(jnp.abs(attributions['edges']).mean(axis=0))
            global_importance.append(jnp.abs(attributions['globals']).mean(axis=0))
        
        # Average across graphs
        node_importance_evolution.append(jnp.mean(jnp.array(node_importance), axis=0))
        edge_importance_evolution.append(jnp.mean(jnp.array(edge_importance), axis=0))
        global_importance_evolution.append(jnp.mean(jnp.array(global_importance), axis=0))
    
    return {
        'nodes': node_importance_evolution,
        'edges': edge_importance_evolution,
        'globals': global_importance_evolution
    }


def concept_activation_vectors(model: Callable, 
                               positive_examples: List[GraphsTuple],
                               negative_examples: List[GraphsTuple],
                               layer_name: str) -> Array:
    """Implement Testing with Concept Activation Vectors (TCAV) for GNoME.
    
    Based on the paper: Interpretability Beyond Feature Attribution
    (https://arxiv.org/abs/1711.11279)
    
    Args:
        model: The GNoME model to analyze
        positive_examples: Graphs that contain the concept
        negative_examples: Graphs that don't contain the concept
        layer_name: Name of the layer to extract activations from
        
    Returns:
        cav: Concept activation vector
    """
    # Extract activations for positive examples
    positive_activations = []
    for graph in positive_examples:
        # For real implementation, we'd need to extract activations from a specific layer
        # This is a placeholder that would need to be replaced with actual layer extraction
        activations = jnp.zeros((100,))  # Placeholder
        positive_activations.append(activations)
    
    # Extract activations for negative examples
    negative_activations = []
    for graph in negative_examples:
        # Placeholder
        activations = jnp.zeros((100,))
        negative_activations.append(activations)
    
    # Convert to arrays
    positive_activations = jnp.array(positive_activations)
    negative_activations = jnp.array(negative_activations)
    
    # Train a linear classifier to separate positive and negative examples
    # In a real implementation, we would use a linear SVM
    # Placeholder for the concept activation vector
    cav = jnp.ones((100,)) / jnp.sqrt(100)
    
    return cav


def search_novel_stability_drivers(model: Callable, 
                                   dataset: List[GraphsTuple],
                                   known_concepts: List[str]) -> List[Dict]:
    """Search for novel stability drivers not evident in existing literature.
    
    Args:
        model: The GNoME model to analyze
        dataset: Dataset of crystal structures
        known_concepts: List of known chemical concepts
        
    Returns:
        novel_drivers: List of potential novel stability drivers
    """
    # This would require a more sophisticated implementation
    # Placeholder for potential novel stability drivers
    novel_drivers = [
        {"name": "hypothetical_pattern_1", "description": "Pattern in coordination environments", "score": 0.85},
        {"name": "hypothetical_pattern_2", "description": "Correlation with atomic radius ratio", "score": 0.78}
    ]
    
    return novel_drivers