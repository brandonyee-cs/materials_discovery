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
Refine crystal structure prediction using XAI insights.

This script enhances SAPS and AIRSS by prioritizing XAI-identified stability motifs
in candidate generation and integrating stability gradients into iterative CSP.
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
from xai import iterative_refinement
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


def load_uncertainty_model(model_dir: str):
    """Load uncertainty model from directory.
    
    Args:
        model_dir: Directory containing model checkpoint
        
    Returns:
        model_fn: Callable model function
    """
    print(f"Loading uncertainty model from {model_dir}...")
    # This is a placeholder - in a real implementation,
    # we would load the uncertainty model similarly to the GNoME model
    def uncertainty_model_fn(graph, positions=None, box=None):
        # Placeholder that returns a random uncertainty
        return jnp.array([0.1 * jax.random.normal(jax.random.PRNGKey(0), ())])
    
    return uncertainty_model_fn


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


def refine_candidates(args):
    """Refine crystal structure prediction using XAI insights.
    
    Args:
        args: Command-line arguments
    """
    # Load model
    model_fn, config = load_model(args.model_dir)
    
    # Load uncertainty model if specified
    uncertainty_model_fn = None
    if args.uncertainty_model_dir:
        uncertainty_model_fn = load_uncertainty_model(args.uncertainty_model_dir)
    
    # Load data
    graphs, positions, boxes = load_data(args.data_file)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.method == 'saps':
        # Enhance SAPS with XAI
        xai_saps = iterative_refinement.XAIGuidedSAPS(
            model_fn, xai_method=args.xai_method)
        
        # Extract stability motifs from stable structures
        print("Extracting stability motifs from stable structures...")
        motifs = xai_saps.extract_stability_motifs(graphs[:args.num_examples])
        
        # Save motifs
        with open(os.path.join(args.output_dir, 'stability_motifs.pkl'), 'wb') as f:
            pickle.dump(motifs, f)
        
        # Visualize motifs
        fig = visualizations.visualize_stability_motifs(
            motifs,
            title="XAI-Identified Stability Motifs",
            save_path=os.path.join(args.output_dir, 'stability_motifs.png')
        )
        plt.close(fig)
        
        # Generate candidates with XAI guidance
        print("Generating candidates with XAI guidance...")
        composition = jnp.ones((94,))  # Placeholder for composition
        candidates = xai_saps.generate_candidates(
            composition, motifs, num_candidates=args.num_candidates)
        
        # Save candidates
        with open(os.path.join(args.output_dir, 'saps_candidates.pkl'), 'wb') as f:
            pickle.dump(candidates, f)
    
    elif args.method == 'airss':
        # Enhance AIRSS with XAI
        xai_airss = iterative_refinement.XAIGuidedAIRSS(
            model_fn, xai_method=args.xai_method)
        
        # Extract stability patterns from stable structures
        print("Analyzing stable structures...")
        stable_structures = [(graphs[i], positions[i], boxes[i]) 
                            for i in range(min(args.num_examples, len(graphs)))
                            if positions[i] is not None]
        
        patterns = xai_airss.analyze_stable_structures(stable_structures)
        
        # Save patterns
        with open(os.path.join(args.output_dir, 'stability_patterns.pkl'), 'wb') as f:
            pickle.dump(patterns, f)
        
        # Generate candidates with XAI guidance
        print("Generating candidates with XAI guidance...")
        composition = jnp.ones((94,))  # Placeholder for composition
        candidates = xai_airss.generate_candidates(
            composition, patterns, num_candidates=args.num_candidates)
        
        # Save candidates
        with open(os.path.join(args.output_dir, 'airss_candidates.pkl'), 'wb') as f:
            pickle.dump(candidates, f)
    
    elif args.method == 'basin_hopping':
        # Run XAI-guided basin hopping
        print("Running XAI-guided basin hopping...")
        
        # Choose a structure to refine
        i = 0
        while i < len(graphs) and positions[i] is None:
            i += 1
        
        if i < len(graphs) and positions[i] is not None:
            # Initialize basin hopping
            basin_hopping = iterative_refinement.XAIGuidedBasinHopping(
                model_fn, temperature=args.temperature, step_size=args.step_size)
            
            # Run basin hopping
            best_graph, best_positions, best_box, best_energy = basin_hopping.run_basin_hopping(
                graphs[i], positions[i], boxes[i], num_steps=args.max_iterations)
            
            # Save results
            with open(os.path.join(args.output_dir, 'basin_hopping_result.pkl'), 'wb') as f:
                pickle.dump({
                    'original_graph': graphs[i],
                    'original_positions': positions[i],
                    'original_box': boxes[i],
                    'original_energy': float(model_fn(graphs[i], positions[i], boxes[i])),
                    'best_graph': best_graph,
                    'best_positions': best_positions,
                    'best_box': best_box,
                    'best_energy': float(best_energy)
                }, f)
            
            # Visualize comparison
            fig = visualizations.visualize_counterfactual_comparison(
                graphs[i], positions[i],
                best_graph, best_positions,
                boxes[i],
                title=f"Basin Hopping Optimization",
                save_path=os.path.join(args.output_dir, 'basin_hopping_result.png')
            )
            plt.close(fig)
    
    elif args.method == 'uncertainty_guided':
        if uncertainty_model_fn is not None:
            # Run uncertainty-guided exploration
            print("Running uncertainty-guided exploration...")
            
            # Initialize uncertainty-guided exploration
            explorer = iterative_refinement.UncertaintyGuidedExploration(
                model_fn, uncertainty_model_fn)
            
            # Choose a structure to analyze
            i = 0
            while i < len(graphs) and positions[i] is None:
                i += 1
            
            if i < len(graphs) and positions[i] is not None:
                # Identify high uncertainty regions
                atom_uncertainties, bond_uncertainties = explorer.identify_high_uncertainty_regions(
                    graphs[i], positions[i], boxes[i])
                
                # Get modification directions
                directions = explorer.get_modification_directions(graphs[i])
                
                # Save results
                with open(os.path.join(args.output_dir, 'uncertainty_analysis.pkl'), 'wb') as f:
                    pickle.dump({
                        'graph': graphs[i],
                        'positions': positions[i],
                        'box': boxes[i],
                        'atom_uncertainties': atom_uncertainties,
                        'bond_uncertainties': bond_uncertainties,
                        'directions': directions
                    }, f)
                
                # Visualize s
                fig = visualizations.visualize_crystal_structure(
                    graphs[i], positions[i], boxes[i],
                    atom_uncertainties, bond_uncertainties,
                    title="Uncertainty Analysis",
                    save_path=os.path.join(args.output_dir, 'uncertainty_analysis.png')
                )
                plt.close(fig)
            
            # Explore novel structures
            print("Exploring novel structures...")
            composition = jnp.ones((94,))  # Placeholder for composition
            candidates = explorer.explore_novel_structures(
                composition, num_candidates=args.num_candidates)
            
            # Save candidates
            with open(os.path.join(args.output_dir, 'uncertainty_candidates.pkl'), 'wb') as f:
                pickle.dump(candidates, f)
    
    print("Refinement complete!")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Refine crystal structure prediction using XAI insights.")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing model checkpoint')
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to data file containing crystal graphs')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save refinement results')
    parser.add_argument('--method', type=str, default='saps',
                        choices=['saps', 'airss', 'basin_hopping', 'uncertainty_guided'],
                        help='Refinement method')
    parser.add_argument('--xai_method', type=str, default='integrated_gradients',
                        choices=['gnnexplainer', 'integrated_gradients', 'shap'],
                        help='XAI method to use')
    parser.add_argument('--num_examples', type=int, default=10,
                        help='Number of example structures to analyze')
    parser.add_argument('--num_candidates', type=int, default=100,
                        help='Number of candidates to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature parameter for basin hopping')
    parser.add_argument('--step_size', type=float, default=0.1,
                        help='Step size for basin hopping')
    parser.add_argument('--max_iterations', type=int, default=100,
                        help='Maximum number of iterations')
    parser.add_argument('--uncertainty_model_dir', type=str, default=None,
                        help='Directory containing uncertainty model checkpoint')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    refine_candidates(args)