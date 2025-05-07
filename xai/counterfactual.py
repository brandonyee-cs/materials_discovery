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
Implementation of counterfactual explanation methods for GNoME models.

This module provides algorithms for minimal perturbations (e.g., substitutions, 
displacements) that alter stability predictions, using gradient-based or 
XAI-guided discrete search.
"""

import jax
import jax.numpy as jnp
import jraph
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from functools import partial

from gnome import crystal, gnome, gnn

Array = jnp.ndarray
GraphsTuple = jraph.GraphsTuple


class GradientBasedCounterfactual:
    """Generate counterfactual explanations using gradient-based optimization."""
    
    def __init__(self, model: Callable, lr: float = 0.01, max_iter: int = 100):
        """Initialize the counterfactual generator.
        
        Args:
            model: The GNoME model
            lr: Learning rate for optimization
            max_iter: Maximum number of iterations
        """
        self.model = model
        self.lr = lr
        self.max_iter = max_iter
        
    def generate_atomic_substitution(self, 
                                     graph: GraphsTuple, 
                                     target_stability: float,
                                     allowed_elements: List[int] = None) -> Tuple[GraphsTuple, List[Tuple[int, int]]]:
        """Generate counterfactual by substituting atoms.
        
        Args:
            graph: The input crystal graph
            target_stability: Target stability value
            allowed_elements: List of allowed element indices
            
        Returns:
            cf_graph: Counterfactual graph
            substitutions: List of (atom_idx, new_element) tuples
        """
        # Create a copy of the graph
        cf_graph = jraph.GraphsTuple(
            nodes=jnp.copy(graph.nodes),
            edges=jnp.copy(graph.edges),
            receivers=graph.receivers,
            senders=graph.senders,
            globals=jnp.copy(graph.globals),
            n_node=graph.n_node,
            n_edge=graph.n_edge
        )
        
        # Define loss function for optimization
        def loss_fn(nodes):
            g = cf_graph._replace(nodes=nodes)
            pred = self.model(g, None, None)
            return jnp.mean((pred - target_stability) ** 2)
        
        # Get gradient function
        grad_fn = jax.grad(loss_fn)
        
        # Initialize substitutions list
        substitutions = []
        
        # Perform gradient descent to find minimum perturbation
        for _ in range(self.max_iter):
            # Get gradients
            grads = grad_fn(cf_graph.nodes)
            
            # Find atom with highest gradient magnitude
            atom_idx = jnp.argmax(jnp.sum(jnp.abs(grads), axis=1))
            
            # Determine new element (simplified; in practice we'd need to make discrete choices)
            if allowed_elements is not None:
                # Project to allowed elements
                element_scores = jnp.zeros((len(allowed_elements),))
                for i, elem in enumerate(allowed_elements):
                    # Create one-hot encoding for this element
                    elem_encoding = jnp.zeros_like(cf_graph.nodes[atom_idx])
                    elem_encoding = elem_encoding.at[elem].set(1.0)
                    
                    # Estimate score
                    test_nodes = cf_graph.nodes.at[atom_idx].set(elem_encoding)
                    test_g = cf_graph._replace(nodes=test_nodes)
                    pred = self.model(test_g, None, None)
                    element_scores = element_scores.at[i].set(-jnp.mean((pred - target_stability) ** 2))
                
                # Choose best element
                best_elem_idx = jnp.argmax(element_scores)
                new_element = allowed_elements[best_elem_idx]
            else:
                # Choose element with gradient in opposite direction
                new_element = jnp.argmin(grads[atom_idx])
            
            # Update node features (one-hot encoding)
            new_node_features = jnp.zeros_like(cf_graph.nodes[atom_idx])
            new_node_features = new_node_features.at[new_element].set(1.0)
            cf_graph = cf_graph._replace(nodes=cf_graph.nodes.at[atom_idx].set(new_node_features))
            
            # Record substitution
            substitutions.append((int(atom_idx), int(new_element)))
            
            # Check if we've reached the target
            pred = self.model(cf_graph, None, None)
            if jnp.abs(pred - target_stability) < 0.01:
                break
        
        return cf_graph, substitutions
    
    def generate_atomic_displacement(self, 
                                    graph: GraphsTuple, 
                                    positions: Array,
                                    box: Array,
                                    target_stability: float,
                                    max_displacement: float = 0.2) -> Tuple[Array, List[Tuple[int, Array]]]:
        """Generate counterfactual by displacing atoms.
        
        Args:
            graph: The input crystal graph
            positions: Atomic positions
            box: Unit cell box
            target_stability: Target stability value
            max_displacement: Maximum allowed displacement magnitude
            
        Returns:
            new_positions: Counterfactual atomic positions
            displacements: List of (atom_idx, displacement_vector) tuples
        """
        # Create a copy of the positions
        new_positions = jnp.copy(positions)
        
        # Define loss function for optimization
        def loss_fn(pos):
            # We would need to regenerate the graph with new positions
            # For simplicity, we're assuming the model can take positions directly
            pred = self.model(graph, pos, box)
            return jnp.mean((pred - target_stability) ** 2)
        
        # Get gradient function
        grad_fn = jax.grad(loss_fn)
        
        # Initialize displacements list
        displacements = []
        
        # Perform gradient descent to find minimum perturbation
        for _ in range(self.max_iter):
            # Get gradients
            grads = grad_fn(new_positions)
            
            # Normalize gradients
            grad_norms = jnp.sqrt(jnp.sum(grads ** 2, axis=1, keepdims=True))
            normalized_grads = grads / (grad_norms + 1e-10)
            
            # Find atom with highest gradient magnitude
            atom_idx = jnp.argmax(grad_norms)
            
            # Compute displacement (limit by max_displacement)
            displacement = -normalized_grads[atom_idx] * min(self.lr, max_displacement)
            
            # Update position
            new_positions = new_positions.at[atom_idx].add(displacement)
            
            # Record displacement
            displacements.append((int(atom_idx), displacement))
            
            # Check if we've reached the target
            pred = self.model(graph, new_positions, box)
            if jnp.abs(pred - target_stability) < 0.01:
                break
        
        return new_positions, displacements
    
    def evaluate_dft_feasibility(self, 
                                graph: GraphsTuple, 
                                mlip_model: Callable) -> float:
        """Evaluate energetic feasibility of modification using MLIPs pre-DFT.
        
        Args:
            graph: The counterfactual crystal graph
            mlip_model: Machine learning interatomic potential model
            
        Returns:
            energy: Estimated energy of the structure
        """
        # For a real implementation, we would use the MLIP to compute energy
        # This is a placeholder
        energy = mlip_model(graph, None, None)
        
        return energy


class DiscreteSearchCounterfactual:
    """Generate counterfactual explanations using discrete search strategies."""
    
    def __init__(self, model: Callable, max_iter: int = 100):
        """Initialize the counterfactual generator.
        
        Args:
            model: The GNoME model
            max_iter: Maximum number of iterations
        """
        self.model = model
        self.max_iter = max_iter
        
    def greedy_atomic_substitution(self, 
                                  graph: GraphsTuple, 
                                  target_stability: float,
                                  allowed_elements: List[int] = None) -> Tuple[GraphsTuple, List[Tuple[int, int]]]:
        """Generate counterfactual by greedily substituting atoms.
        
        Args:
            graph: The input crystal graph
            target_stability: Target stability value
            allowed_elements: List of allowed element indices
            
        Returns:
            cf_graph: Counterfactual graph
            substitutions: List of (atom_idx, new_element) tuples
        """
        # Create a copy of the graph
        cf_graph = jraph.GraphsTuple(
            nodes=jnp.copy(graph.nodes),
            edges=jnp.copy(graph.edges),
            receivers=graph.receivers,
            senders=graph.senders,
            globals=jnp.copy(graph.globals),
            n_node=graph.n_node,
            n_edge=graph.n_edge
        )
        
        # Define elements to consider
        if allowed_elements is None:
            allowed_elements = list(range(cf_graph.nodes.shape[1]))
        
        # Initialize substitutions list
        substitutions = []
        
        # Perform greedy search
        for _ in range(self.max_iter):
            best_score = float('inf')
            best_substitution = None
            
            # Try substituting each atom with each allowed element
            for atom_idx in range(cf_graph.nodes.shape[0]):
                original_atom = cf_graph.nodes[atom_idx]
                
                for elem in allowed_elements:
                    # Skip if this is already the current element
                    if original_atom[elem] > 0.5:
                        continue
                    
                    # Create one-hot encoding for this element
                    elem_encoding = jnp.zeros_like(original_atom)
                    elem_encoding = elem_encoding.at[elem].set(1.0)
                    
                    # Create test graph with substituted atom
                    test_nodes = cf_graph.nodes.at[atom_idx].set(elem_encoding)
                    test_g = cf_graph._replace(nodes=test_nodes)
                    
                    # Evaluate
                    pred = self.model(test_g, None, None)
                    score = jnp.abs(pred - target_stability)
                    
                    if score < best_score:
                        best_score = score
                        best_substitution = (atom_idx, elem)
            
            # Apply best substitution
            if best_substitution is not None:
                atom_idx, new_elem = best_substitution
                
                # Update node features
                new_node_features = jnp.zeros_like(cf_graph.nodes[atom_idx])
                new_node_features = new_node_features.at[new_elem].set(1.0)
                cf_graph = cf_graph._replace(nodes=cf_graph.nodes.at[atom_idx].set(new_node_features))
                
                # Record substitution
                substitutions.append((int(atom_idx), int(new_elem)))
            
            # Check if we've reached the target
            if best_score < 0.01:
                break
        
        return cf_graph, substitutions
    
    def evolutionary_search(self, 
                           graph: GraphsTuple, 
                           target_stability: float,
                           population_size: int = 10,
                           generations: int = 10,
                           mutation_rate: float = 0.1) -> Tuple[GraphsTuple, List[Tuple[int, int]]]:
        """Generate counterfactual using evolutionary search.
        
        Args:
            graph: The input crystal graph
            target_stability: Target stability value
            population_size: Size of the population
            generations: Number of generations
            mutation_rate: Probability of mutation
            
        Returns:
            cf_graph: Counterfactual graph
            substitutions: List of (atom_idx, new_element) tuples
        """
        # This is a placeholder implementation
        # In a real implementation, we would maintain a population of candidate solutions
        
        # Create a copy of the graph for best solution
        best_graph = jraph.GraphsTuple(
            nodes=jnp.copy(graph.nodes),
            edges=jnp.copy(graph.edges),
            receivers=graph.receivers,
            senders=graph.senders,
            globals=jnp.copy(graph.globals),
            n_node=graph.n_node,
            n_edge=graph.n_edge
        )
        
        # Placeholder for substitutions
        substitutions = [(0, 1), (3, 5)]  # Example substitutions
        
        return best_graph, substitutions


def connect_to_synthesizability(counterfactual: GraphsTuple, 
                               original: GraphsTuple) -> Dict[str, Union[float, str]]:
    """Connect counterfactual explanations to synthesizability.
    
    Args:
        counterfactual: Counterfactual crystal graph
        original: Original crystal graph
        
    Returns:
        synthesis_guide: Dictionary with synthesis suggestions
    """
    # This would require domain-specific knowledge
    # Placeholder for synthesis guidance
    synthesis_guide = {
        "feasibility_score": 0.75,
        "suggested_method": "Solid-state reaction",
        "precursors": ["Li2CO3", "Mn2O3"],
        "temperature": "800°C",
        "doping_strategy": "5% Fe substitution at Mn sites"
    }
    
    return synthesis_guide


def analyze_counterfactual_patterns(counterfactuals: List[GraphsTuple],
                                   originals: List[GraphsTuple]) -> Dict:
    """Analyze patterns in counterfactual explanations.
    
    Args:
        counterfactuals: List of counterfactual crystal graphs
        originals: List of original crystal graphs
        
    Returns:
        patterns: Dictionary with identified patterns
    """
    # This would require sophisticated pattern recognition
    # Placeholder for pattern analysis
    patterns = {
        "common_substitutions": {"Mn -> Fe": 0.45, "O -> S": 0.32},
        "structural_changes": ["coordination_increase", "bond_shortening"],
        "electronic_effects": ["band_gap_widening", "orbital_hybridization"]
    }
    
    return patterns