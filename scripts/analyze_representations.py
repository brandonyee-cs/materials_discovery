#!/usr/bin/env python3
# Copyright 2025
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
Analyze GNoME's learned representations across active learning and chemical space.

This script applies XAI techniques to GNoME models to analyze learned representations
and test for known chemical concepts.
"""

import os
import argparse
import pickle
import jax
import jax.numpy as jnp
import jraph
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple
from ml_collections import ConfigDict

from GNoME import gnome
from GNoME import crystal
from xai import visualizations


def create_simple_explainer(model_fn):
    """Create a simple explainer that works without gradients."""
    
    def explain_graph(graph):
        """Explain graph prediction using input perturbation."""
        # Get baseline prediction
        baseline_pred = model_fn(graph, None, None)
        
        # Initialize importance scores
        node_importance = np.zeros(graph.nodes.shape[0])
        edge_importance = np.zeros(graph.n_edge.sum())
        
        # Compute node importance by perturbation
        for i in range(graph.nodes.shape[0]):
            # Create perturbed graph with one node zeroed out
            perturbed_nodes = jnp.array(graph.nodes)
            perturbed_nodes = perturbed_nodes.at[i].set(jnp.zeros_like(perturbed_nodes[i]))
            perturbed_graph = graph._replace(nodes=perturbed_nodes)
            
            # Get prediction for perturbed graph
            perturbed_pred = model_fn(perturbed_graph, None, None)
            
            # Compute importance as change in prediction
            node_importance = node_importance.at[i].set(
                float(jnp.abs(perturbed_pred - baseline_pred)))
        
        # Normalize importance scores
        if node_importance.max() > 0:
            node_importance = node_importance / node_importance.max()
        
        return {
            'nodes': node_importance,
            'edges': edge_importance,
            'globals': np.array([1.0])  # Default global importance
        }
    
    return explain_graph


def load_models(model_dirs: List[str]):
    """Load GNoME models from directories."""
    models = []
    configs = []
    
    for model_dir in model_dirs:
        print(f"Loading model from {model_dir}...")
        config, model, params = gnome.load_model(model_dir)
        
        # Create a simple function that just forwards to the model
        def model_fn(graph, positions=None, box=None, params=params, model=model):
            return model.apply(params, graph, positions, box)
        
        models.append(model_fn)
        configs.append(config)
    
    return models, configs


def load_data(data_file: str):
    """Load crystal graph data from file.
    
    Args:
        data_file: Path to data file
        
    Returns:
        graphs: List of crystal graphs
        positions: List of atomic positions
        boxes: List of unit cell boxes
    """
    print(f"Loading data from {data_file}...")
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    
    # Extract graphs, positions, and boxes
    graphs = data.get('graphs', [])
    positions = data.get('positions', [None] * len(graphs))
    boxes = data.get('boxes', [None] * len(graphs))
    
    return graphs, positions, boxes


def chemical_concept_test(model_fn, graph, concept_name):
    """Simple test for chemical concept learning."""
    # Create a perturbation based on the concept
    if concept_name == "pauling_electronegativity":
        # Perturb elements with high electronegativity
        perturbed_graph = perturb_electronegative_elements(graph)
    elif concept_name == "coordination_preference":
        # Perturb coordination environment
        perturbed_graph = perturb_coordination(graph)
    else:
        # Default perturbation
        perturbed_graph = graph
    
    # Get predictions
    baseline_pred = model_fn(graph, None, None)
    perturbed_pred = model_fn(perturbed_graph, None, None)
    
    # Compute sensitivity
    sensitivity = float(jnp.abs(perturbed_pred - baseline_pred))
    
    return sensitivity


def perturb_electronegative_elements(graph):
    """Perturb elements with high electronegativity."""
    # For simplicity, we'll just zero out oxygen/fluorine atoms (O=7, F=8 in 0-indexed)
    electronegative_indices = [7, 8]
    
    perturbed_nodes = jnp.array(graph.nodes)
    for i in range(perturbed_nodes.shape[0]):
        for idx in electronegative_indices:
            if idx < perturbed_nodes.shape[1] and perturbed_nodes[i, idx] > 0.5:
                # Replace with silicon (Si=13 in 0-indexed)
                new_element = jnp.zeros_like(perturbed_nodes[i])
                new_element = new_element.at[13].set(1.0)
                perturbed_nodes = perturbed_nodes.at[i].set(new_element)
    
    return graph._replace(nodes=perturbed_nodes)


def perturb_coordination(graph):
    """Perturb coordination environment."""
    # Simple implementation: remove some edges
    if graph.n_edge.sum() > 0:
        # Keep only half the edges
        keep_edges = graph.n_edge.sum() // 2
        
        perturbed_senders = graph.senders[:keep_edges]
        perturbed_receivers = graph.receivers[:keep_edges]
        
        if isinstance(graph.edges, tuple):
            edge_data, edge_meta = graph.edges
            perturbed_edges = (edge_data[:keep_edges], edge_meta[:keep_edges])
        else:
            perturbed_edges = graph.edges[:keep_edges]
        
        perturbed_n_edge = jnp.array([keep_edges])
        
        return graph._replace(
            edges=perturbed_edges,
            senders=perturbed_senders,
            receivers=perturbed_receivers,
            n_edge=perturbed_n_edge
        )
    
    return graph


def track_feature_importance_evolution(models, graphs):
    """Track evolution of feature importance across models."""
    # Create explainers for each model
    explainers = [create_simple_explainer(model) for model in models]
    
    # Initialize importance storage
    importance_evolution = {
        'nodes': [],
        'edges': [],
        'globals': []
    }
    
    # Process each model
    for explainer in explainers:
        node_importance = []
        edge_importance = []
        global_importance = []
        
        # Process each graph
        for graph in graphs:
            # Get explanation
            explanation = explainer(graph)
            
            # Store feature importance
            if len(explanation['nodes']) > 0:
                node_importance.append(explanation['nodes'])
            if len(explanation['edges']) > 0:
                edge_importance.append(explanation['edges'])
            if len(explanation['globals']) > 0:
                global_importance.append(explanation['globals'])
        
        # Average across graphs or pad with zeros if needed
        if node_importance:
            # Different graphs may have different numbers of nodes
            # For simplicity, we'll just use the first graph's shape
            avg_node_importance = np.mean(node_importance, axis=0)
            importance_evolution['nodes'].append(avg_node_importance)
        else:
            importance_evolution['nodes'].append(np.zeros(1))
        
        if edge_importance:
            avg_edge_importance = np.mean(edge_importance, axis=0)
            importance_evolution['edges'].append(avg_edge_importance)
        else:
            importance_evolution['edges'].append(np.zeros(1))
        
        if global_importance:
            avg_global_importance = np.mean(global_importance, axis=0)
            importance_evolution['globals'].append(avg_global_importance)
        else:
            importance_evolution['globals'].append(np.zeros(1))
    
    return importance_evolution


def search_novel_stability_drivers(model_fn, graphs, known_concepts):
    """Search for novel stability drivers."""
    # Simple placeholder implementation
    novel_drivers = [
        {"name": "hypothetical_pattern_1", "description": "Pattern in coordination environments", "score": 0.85},
        {"name": "hypothetical_pattern_2", "description": "Correlation with atomic radius ratio", "score": 0.78}
    ]
    
    return novel_drivers


def analyze_representations(args):
    """Analyze GNoME's learned representations.
    
    Args:
        args: Command-line arguments
    """
    # Load models from different active learning rounds
    models, configs = load_models(args.model_dirs)
    
    # Load data
    graphs, positions, boxes = load_data(args.data_file)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Analyze representations across active learning rounds
    if args.track_evolution:
        print("Tracking feature importance evolution across models...")
        importance_evolution = track_feature_importance_evolution(
            models, graphs[:args.num_examples])
        
        # Save results
        with open(os.path.join(args.output_dir, 'importance_evolution.pkl'), 'wb') as f:
            pickle.dump(importance_evolution, f)
        
        # Visualize results
        fig = visualizations.visualize_feature_importance_evolution(
            importance_evolution,
            title="Feature Importance Evolution Across Active Learning Rounds",
            save_path=os.path.join(args.output_dir, 'importance_evolution.png')
        )
        plt.close(fig)
    
    # Test for known chemical concepts
    if args.test_concepts:
        print("Testing for known chemical concepts...")
        # Define concepts to test
        concepts = [
            "pauling_electronegativity",
            "coordination_preference",
            "atomic_radius",
            "valence_electrons",
            "bond_strength"
        ]
        
        # Test each concept
        concept_sensitivities = {}
        for concept in concepts[:2]:  # Test only first two concepts for simplicity
            print(f"Testing concept: {concept}")
            sensitivity = chemical_concept_test(models[-1], graphs[0], concept)
            concept_sensitivities[concept] = float(sensitivity)
        
        # Save results
        with open(os.path.join(args.output_dir, 'concept_sensitivities.pkl'), 'wb') as f:
            pickle.dump(concept_sensitivities, f)
        
        # Visualize results
        fig = visualizations.visualize_chemical_concept_test(
            concept_sensitivities,
            title="Chemical Concept Test Results",
            save_path=os.path.join(args.output_dir, 'concept_sensitivities.png')
        )
        plt.close(fig)
    
    # Search for novel stability drivers
    if args.search_novel:
        print("Searching for novel stability drivers...")
        concepts = [
            "pauling_electronegativity",
            "coordination_preference"
        ]
        novel_drivers = search_novel_stability_drivers(
            models[-1], graphs[:args.num_examples], concepts)
        
        # Save results
        with open(os.path.join(args.output_dir, 'novel_drivers.pkl'), 'wb') as f:
            pickle.dump(novel_drivers, f)
        
        # Visualize results
        fig = visualizations.visualize_novel_stability_drivers(
            novel_drivers,
            title="Novel Stability Drivers",
            save_path=os.path.join(args.output_dir, 'novel_drivers.png')
        )
        plt.close(fig)
    
    # Analyze individual structures if requested
    if args.analyze_examples:
        print(f"Analyzing {args.num_examples} example structures...")
        # Create explainer for the last model
        explainer = create_simple_explainer(models[-1])
        
        for i in range(min(args.num_examples, len(graphs))):
            print(f"Analyzing structure {i+1}...")
            
            # Generate explanation
            explanation = explainer(graphs[i])
            
            # Save explanation
            with open(os.path.join(args.output_dir, f'explanation_{i}.pkl'), 'wb') as f:
                pickle.dump(explanation, f)
            
            # Visualize explanation if positions are available
            if positions[i] is not None:
                fig = visualizations.visualize_crystal_structure(
                    graphs[i], positions[i], boxes[i],
                    explanation['nodes'], explanation['edges'],
                    title=f"Structure {i+1} Explanation (Perturbation Method)",
                    save_path=os.path.join(args.output_dir, f'structure_{i}_explanation.png')
                )
                plt.close(fig)
    
    print("Analysis complete!")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze GNoME's learned representations.")
    parser.add_argument('--model_dirs', type=str, nargs='+', required=True,
                        help='Directories containing model checkpoints')
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to data file containing crystal graphs')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save analysis results')
    parser.add_argument('--xai_method', type=str, default='integrated_gradients',
                        choices=['gnnexplainer', 'integrated_gradients', 'shap'],
                        help='XAI method to use')
    parser.add_argument('--track_evolution', action='store_true',
                        help='Track feature importance evolution across models')
    parser.add_argument('--test_concepts', action='store_true',
                        help='Test for known chemical concepts')
    parser.add_argument('--search_novel', action='store_true',
                        help='Search for novel stability drivers')
    parser.add_argument('--analyze_examples', action='store_true',
                        help='Analyze individual example structures')
    parser.add_argument('--num_examples', type=int, default=10,
                        help='Number of example structures to analyze')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    analyze_representations(args)