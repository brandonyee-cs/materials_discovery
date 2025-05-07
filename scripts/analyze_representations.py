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

This script applies XAI techniques (GNNExplainer, Integrated Gradients, SHAP) to GNoME
models to analyze learned representations and test for known chemical concepts.
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

from GNoME import gnome
from xai import representation_analysis
from xai import visualizations


def load_models(model_dirs: List[str]):
    """Load GNoME models from directories.
    
    Args:
        model_dirs: List of directories containing model checkpoints
        
    Returns:
        models: List of loaded models
        configs: List of model configurations
    """
    models = []
    configs = []
    
    for model_dir in model_dirs:
        print(f"Loading model from {model_dir}...")
        config, model, params = gnome.load_model(model_dir)
        # Create a callable model function that uses the loaded parameters
        def model_fn(graph, positions=None, box=None):
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
    
    # Select XAI method
    if args.xai_method == 'gnnexplainer':
        explainer_class = representation_analysis.GNNExplainer
    elif args.xai_method == 'integrated_gradients':
        explainer_class = representation_analysis.IntegratedGradients
    elif args.xai_method == 'shap':
        explainer_class = representation_analysis.SHAP
    else:
        raise ValueError(f"Unknown XAI method: {args.xai_method}")
    
    # Analyze representations across active learning rounds
    if args.track_evolution:
        print("Tracking feature importance evolution across models...")
        importance_evolution = representation_analysis.track_feature_importance_evolution(
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
        
        # Define perturbation functions for each concept
        def perturb_electronegativity(graph):
            """Perturb electronegativity by swapping elements."""
            # This is a simplified placeholder
            perturbed_graph = jraph.GraphsTuple(
                nodes=jnp.copy(graph.nodes),
                edges=jnp.copy(graph.edges),
                receivers=graph.receivers,
                senders=graph.senders,
                globals=jnp.copy(graph.globals),
                n_node=graph.n_node,
                n_edge=graph.n_edge
            )
            return perturbed_graph
        
        # Similar placeholders for other concepts
        perturbation_fns = {
            "pauling_electronegativity": perturb_electronegativity,
            # Add other perturbation functions here
        }
        
        # Test each concept
        concept_sensitivities = {}
        for concept in concepts:
            if concept in perturbation_fns:
                print(f"Testing concept: {concept}")
                sensitivity = representation_analysis.chemical_concept_test(
                    models[-1], graphs[0], concept, perturbation_fns[concept])
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
        novel_drivers = representation_analysis.search_novel_stability_drivers(
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
        for i in range(min(args.num_examples, len(graphs))):
            print(f"Analyzing structure {i+1}...")
            # Initialize explainer with the latest model
            explainer = explainer_class(models[-1])
            
            # Generate explanation
            if args.xai_method == 'gnnexplainer':
                edge_mask, node_mask = explainer.explain_graph(graphs[i])
                explanation = {"edge_mask": edge_mask, "node_mask": node_mask}
            elif args.xai_method == 'integrated_gradients':
                explanation = explainer.explain_graph(graphs[i])
            elif args.xai_method == 'shap':
                explanation = explainer.explain_graph(graphs[i], graphs[:5])
            
            # Save explanation
            with open(os.path.join(args.output_dir, f'explanation_{i}.pkl'), 'wb') as f:
                pickle.dump(explanation, f)
            
            # Visualize explanation if positions are available
            if positions[i] is not None:
                # Extract node and edge importance from explanation
                if 'nodes' in explanation:
                    node_importance = jnp.abs(explanation['nodes']).mean(axis=1)
                elif 'node_mask' in explanation:
                    node_importance = explanation['node_mask']
                else:
                    node_importance = None
                
                if 'edges' in explanation:
                    edge_importance = jnp.abs(explanation['edges']).mean(axis=1)
                elif 'edge_mask' in explanation:
                    edge_importance = explanation['edge_mask']
                else:
                    edge_importance = None
                
                fig = visualizations.visualize_crystal_structure(
                    graphs[i], positions[i], boxes[i],
                    node_importance, edge_importance,
                    title=f"Structure {i+1} Explanation ({args.xai_method})",
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