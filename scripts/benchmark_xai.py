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
Benchmark XAI methods for materials science.

This script compares XAI methods' fidelity, interpretability, and cost on GNoME's
dataset, with metrics for explanation actionability.
"""

import os
import argparse
import pickle
import time
import jax
import jax.numpy as jnp
import jraph
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple

from GNoME import gnome
from xai import representation_analysis
from xai import benchmarking
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


def load_csp_model(model_dir: str):
    """Load crystal structure prediction model from directory.
    
    Args:
        model_dir: Directory containing model checkpoint
        
    Returns:
        model_fn: Callable model function
    """
    print(f"Loading CSP model from {model_dir}...")
    # This is a placeholder - in a real implementation,
    # we would load the CSP model similarly to the GNoME model
    def csp_model_fn(graph, positions=None, box=None):
        # Placeholder that returns a random structure
        return jnp.array([0.0])
    
    return csp_model_fn


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


def benchmark_xai_methods(args):
    """Benchmark XAI methods for materials science.
    
    Args:
        args: Command-line arguments
    """
    # Load model
    model_fn, config = load_model(args.model_dir)
    
    # Load CSP model if specified
    csp_model_fn = None
    if args.csp_model_dir:
        csp_model_fn = load_csp_model(args.csp_model_dir)
    
    # Load data
    graphs, positions, boxes = load_data(args.data_file)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Define metrics based on arguments
    metrics = ['fidelity', 'sparsity', 'stability', 'computational_cost']
    if args.csp_model_dir:
        metrics.append('actionability')
    
    # Initialize benchmark framework
    benchmark = benchmarking.XAIBenchmark(model_fn, csp_model_fn, metrics)
    
    # Define XAI methods to benchmark
    explainers = {}
    
    if 'gnnexplainer' in args.methods:
        explainer = representation_analysis.GNNExplainer(model_fn)
        explainers['GNNExplainer'] = lambda g: explainer.explain_graph(g)
    
    if 'integrated_gradients' in args.methods:
        explainer = representation_analysis.IntegratedGradients(model_fn)
        explainers['IntegratedGradients'] = lambda g: explainer.explain_graph(g)
    
    if 'shap' in args.methods:
        explainer = representation_analysis.SHAP(model_fn)
        explainers['SHAP'] = lambda g: explainer.explain_graph(g, graphs[:5])
    
    # Define reference for comparison (optional)
    reference = {
        'max_acceptable_time': 60.0  # seconds
    }
    
    # Select a subset of graphs for benchmarking
    benchmark_graphs = graphs[:args.num_examples]
    
    # Run benchmark
    print(f"Benchmarking {len(explainers)} XAI methods on {len(benchmark_graphs)} examples...")
    results = benchmark.compare_methods(explainers, benchmark_graphs, reference)
    
    # Save benchmark results
    with open(os.path.join(args.output_dir, 'benchmark_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    # Visualize benchmark results
    fig = visualizations.visualize_benchmark_results(
        results,
        title="XAI Method Benchmark Results",
        save_path=os.path.join(args.output_dir, 'benchmark_results.png')
    )
    plt.close(fig)
    
    # Define best practices based on benchmark results
    if len(explainers) > 1:
        print("Defining best practices based on benchmark results...")
        best_practices = benchmarking.define_xai_best_practices(results)
        
        # Save best practices
        with open(os.path.join(args.output_dir, 'best_practices.pkl'), 'wb') as f:
            pickle.dump(best_practices, f)
        
        # Print best practices
        print("\nBest Practices:")
        print(f"  For high fidelity: {best_practices['recommended_methods']['for_high_fidelity']}")
        print(f"  For interpretability: {best_practices['recommended_methods']['for_interpretability']}")
        print(f"  For low cost: {best_practices['recommended_methods']['for_low_cost']}")
        print("\nGuidelines:")
        for guideline in best_practices['guidelines']:
            print(f"  - {guideline}")
    
    # Compare explanations for a specific structure if multiple methods are used
    if len(explainers) > 1 and args.compare_example >= 0 and args.compare_example < len(graphs):
        i = args.compare_example
        print(f"Comparing explanations for example {i}...")
        
        # Generate explanations for each method
        explanations = {}
        for method_name, method_fn in explainers.items():
            explanations[method_name] = method_fn(graphs[i])
        
        # Visualize comparison if positions are available
        if positions[i] is not None:
            fig = visualizations.visualize_explanation_comparison(
                explanations,
                graphs[i],
                positions[i],
                title=f"XAI Method Comparison for Structure {i}",
                save_path=os.path.join(args.output_dir, f'explanation_comparison_{i}.png')
            )
            plt.close(fig)
    
    print("Benchmarking complete!")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark XAI methods for materials science.")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing model checkpoint')
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to data file containing crystal graphs')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save benchmark results')
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['gnnexplainer', 'integrated_gradients', 'shap'],
                        help='XAI methods to benchmark')
    parser.add_argument('--num_examples', type=int, default=10,
                        help='Number of examples to use for benchmarking')
    parser.add_argument('--csp_model_dir', type=str, default=None,
                        help='Directory containing CSP model checkpoint')
    parser.add_argument('--compare_example', type=int, default=-1,
                        help='Example index to use for explanation comparison')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    benchmark_xai_methods(args)