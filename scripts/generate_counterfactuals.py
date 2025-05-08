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
Generate counterfactual explanations for GNoME models.

This script generates counterfactual explanations by identifying minimal perturbations
that alter stability predictions, using gradient-based or XAI-guided discrete search.
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
from xai import counterfactual
from xai import visualizations


def load_model(model_dir: str):
    """Load GNoME model from directory.
    
    Args:
        model_dir: Directory containing model checkpoint
        
    Returns:
        model_fn: Callable model function
        config: Model configuration
    """
    print(f"Loading model from {model_dir}...")
    config, model, params = gnome.load_model(model_dir)
    # Create a callable model function that uses the loaded parameters
    def model_fn(graph, positions=None, box=None):
        return model.apply(params, graph, positions, box)
    
    return model_fn, config


def load_mlip_model(model_dir: str):
    """Load MLIP model from directory.
    
    Args:
        model_dir: Directory containing model checkpoint
        
    Returns:
        model_fn: Callable model function
    """
    print(f"Loading MLIP model from {model_dir}...")
    # This is a placeholder - in a real implementation,
    # we would load the MLIP model similarly to the GNoME model
    def mlip_model_fn(graph, positions=None, box=None):
        # Placeholder that returns a random energy
        return jnp.array([-5.0 + 0.1 * jax.random.normal(jax.random.PRNGKey(0), ())])
    
    return mlip_model_fn


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


def generate_counterfactuals(args):
    """Generate counterfactual explanations.
    
    Args:
        args: Command-line arguments
    """
    # Load model
    model_fn, config = load_model(args.model_dir)
    
    # Load MLIP model if specified
    mlip_model_fn = None
    if args.mlip_dir:
        mlip_model_fn = load_mlip_model(args.mlip_dir)
    
    # Load data
    graphs, positions, boxes = load_data(args.data_file)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Select counterfactual generator
    if args.method == 'gradient':
        cf_generator = counterfactual.GradientBasedCounterfactual(
            model_fn, lr=args.learning_rate, max_iter=args.max_iterations)
    elif args.method == 'discrete':
        cf_generator = counterfactual.DiscreteSearchCounterfactual(
            model_fn, max_iter=args.max_iterations)
    else:
        raise ValueError(f"Unknown counterfactual method: {args.method}")
    
    # Process each structure
    for i in range(min(args.num_examples, len(graphs))):
        print(f"Processing structure {i+1}...")
        
        # Get target stability (opposite of current prediction)
        current_stability = float(model_fn(graphs[i], positions[i], boxes[i]))
        target_stability = -current_stability
        
        # Generate counterfactual
        if args.transformation == 'substitution':
            allowed_elements = list(range(94))  # All elements
            if positions[i] is None:
                # Compositional graph
                cf_graph, substitutions = cf_generator.generate_atomic_substitution(
                    graphs[i], target_stability, allowed_elements)
                
                # Save results
                with open(os.path.join(args.output_dir, f'cf_substitution_{i}.pkl'), 'wb') as f:
                    pickle.dump({
                        'original_graph': graphs[i],
                        'cf_graph': cf_graph,
                        'substitutions': substitutions,
                        'current_stability': current_stability,
                        'target_stability': target_stability,
                        'cf_stability': float(model_fn(cf_graph, None, None))
                    }, f)
                
                # Visualize results (if positions were available)
                if positions[i] is not None:
                    # Extract atom indices that were changed
                    changed_atoms = [sub[0] for sub in substitutions]
                    
                    fig = visualizations.visualize_counterfactual_comparison(
                        graphs[i], positions[i],
                        cf_graph, positions[i],  # Using the same positions
                        boxes[i],
                        highlighted_changes=changed_atoms,
                        title=f"Counterfactual Substitution for Structure {i+1}",
                        save_path=os.path.join(args.output_dir, f'cf_substitution_{i}.png')
                    )
                    plt.close(fig)
                
                # Evaluate DFT feasibility if MLIP model is available
                if mlip_model_fn is not None:
                    energy = cf_generator.evaluate_dft_feasibility(cf_graph, mlip_model_fn)
                    print(f"Estimated energy for counterfactual: {energy}")
            
        elif args.transformation == 'displacement':
            if positions[i] is not None:
                # Structure with positions
                new_positions, displacements = cf_generator.generate_atomic_displacement(
                    graphs[i], positions[i], boxes[i], target_stability, args.max_displacement)
                
                # Save results
                with open(os.path.join(args.output_dir, f'cf_displacement_{i}.pkl'), 'wb') as f:
                    pickle.dump({
                        'original_graph': graphs[i],
                        'original_positions': positions[i],
                        'cf_positions': new_positions,
                        'displacements': displacements,
                        'current_stability': current_stability,
                        'target_stability': target_stability,
                        'cf_stability': float(model_fn(graphs[i], new_positions, boxes[i]))
                    }, f)
                
                # Visualize results
                # Extract atom indices that were changed
                changed_atoms = [disp[0] for disp in displacements]
                
                fig = visualizations.visualize_counterfactual_comparison(
                    graphs[i], positions[i],
                    graphs[i], new_positions,  # Using the same graph, different positions
                    boxes[i],
                    highlighted_changes=changed_atoms,
                    title=f"Counterfactual Displacement for Structure {i+1}",
                    save_path=os.path.join(args.output_dir, f'cf_displacement_{i}.png')
                )
                plt.close(fig)
                
                # Evaluate DFT feasibility if MLIP model is available
                if mlip_model_fn is not None:
                    energy = mlip_model_fn(graphs[i], new_positions, boxes[i])
                    print(f"Estimated energy for counterfactual: {energy}")
            else:
                print(f"Skipping displacement for structure {i+1} (no positions available)")
    
    # Analyze counterfactual patterns if multiple examples are processed
    if args.num_examples > 1:
        print("Analyzing counterfactual patterns...")
        # Load all generated counterfactuals
        counterfactuals = []
        originals = []
        
        if args.transformation == 'substitution':
            for i in range(min(args.num_examples, len(graphs))):
                file_path = os.path.join(args.output_dir, f'cf_substitution_{i}.pkl')
                if os.path.exists(file_path):
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                        counterfactuals.append(data['cf_graph'])
                        originals.append(data['original_graph'])
        
        # Analyze patterns
        if counterfactuals and originals:
            patterns = counterfactual.analyze_counterfactual_patterns(counterfactuals, originals)
            
            # Save patterns
            with open(os.path.join(args.output_dir, 'cf_patterns.pkl'), 'wb') as f:
                pickle.dump(patterns, f)
    
    print("Counterfactual generation complete!")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate counterfactual explanations.")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing model checkpoint')
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to data file containing crystal graphs')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save counterfactuals')
    parser.add_argument('--method', type=str, default='gradient',
                        choices=['gradient', 'discrete'],
                        help='Counterfactual generation method')
    parser.add_argument('--transformation', type=str, default='substitution',
                        choices=['substitution', 'displacement'],
                        help='Type of transformation to apply')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate for gradient-based method')
    parser.add_argument('--max_iterations', type=int, default=100,
                        help='Maximum number of iterations')
    parser.add_argument('--max_displacement', type=float, default=0.2,
                        help='Maximum atomic displacement (Angstroms)')
    parser.add_argument('--num_examples', type=int, default=5,
                        help='Number of examples to process')
    parser.add_argument('--mlip_dir', type=str, default=None,
                        help='Directory containing MLIP model checkpoint')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    generate_counterfactuals(args)