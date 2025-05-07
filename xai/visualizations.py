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
Visualization utilities for XAI results in materials science.

This module provides tools to visualize explanations, counterfactuals,
and other XAI-related outputs for crystal structures.
"""

import jax
import jax.numpy as jnp
import jraph
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from gnome import crystal, gnome, gnn

Array = jnp.ndarray
GraphsTuple = jraph.GraphsTuple


# Periodic table data for visualization
ELEMENT_SYMBOLS = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U', 'Np', 'Pu'
]

# Element colors for visualization
ELEMENT_COLORS = {
    'H': '#FFFFFF', 'Li': '#CC80FF', 'Be': '#C2FF00', 'B': '#FFB5B5',
    'C': '#909090', 'N': '#3050F8', 'O': '#FF0D0D', 'F': '#90E050',
    'Na': '#AB5CF2', 'Mg': '#8AFF00', 'Al': '#BFA6A6', 'Si': '#F0C8A0',
    'P': '#FF8000', 'S': '#FFFF30', 'Cl': '#1FF01F', 'K': '#8F40D4',
    'Ca': '#3DFF00', 'Ti': '#BFC2C7', 'Cr': '#8A99C7', 'Mn': '#9C7AC7',
    'Fe': '#E06633', 'Co': '#F090A0', 'Ni': '#50D050', 'Cu': '#C88033',
    'Zn': '#7D80B0', 'Ga': '#C28F8F', 'Ge': '#668F8F', 'As': '#BD80E3',
    'Se': '#FFA100', 'Br': '#A62929', 'Sr': '#00FF00', 'Zr': '#94E0E0',
    'Mo': '#54B5B5', 'Ag': '#C0C0C0', 'Cd': '#FFD98F', 'In': '#A67573',
    'Sn': '#668080', 'Sb': '#9E63B5', 'Te': '#D47A00', 'I': '#940094',
    'Ba': '#00C900', 'W': '#2194D6', 'Au': '#FFD123', 'Hg': '#B8B8D0',
    'Pb': '#575961', 'Bi': '#9E4FB5'
}

# Default color for elements not in the dictionary
DEFAULT_ELEMENT_COLOR = '#777777'


def convert_to_networkx(graph: GraphsTuple, positions: Array) -> nx.Graph:
    """Convert jraph.GraphsTuple to NetworkX graph for visualization.
    
    Args:
        graph: Crystal graph in jraph format
        positions: Atomic positions
        
    Returns:
        nx_graph: NetworkX graph with node positions
    """
    # Create empty NetworkX graph
    nx_graph = nx.Graph()
    
    # Add nodes with positions and element info
    for i in range(graph.nodes.shape[0]):
        # Find element based on one-hot encoding
        element_idx = jnp.argmax(graph.nodes[i])
        element = ELEMENT_SYMBOLS[element_idx] if element_idx < len(ELEMENT_SYMBOLS) else 'X'
        
        # Add node with attributes
        nx_graph.add_node(i, 
                          pos=positions[i],
                          element=element,
                          color=ELEMENT_COLORS.get(element, DEFAULT_ELEMENT_COLOR))
    
    # Add edges
    for i in range(graph.senders.shape[0]):
        sender = graph.senders[i]
        receiver = graph.receivers[i]
        nx_graph.add_edge(sender, receiver)
    
    return nx_graph


def visualize_crystal_structure(graph: GraphsTuple, 
                               positions: Array,
                               box: Optional[Array] = None,
                               node_importance: Optional[Array] = None,
                               edge_importance: Optional[Array] = None,
                               title: str = "Crystal Structure",
                               save_path: Optional[str] = None) -> plt.Figure:
    """Visualize crystal structure with optional importance highlighting.
    
    Args:
        graph: Crystal graph
        positions: Atomic positions
        box: Unit cell box (optional)
        node_importance: Importance scores for nodes (optional)
        edge_importance: Importance scores for edges (optional)
        title: Plot title
        save_path: Path to save the figure (optional)
        
    Returns:
        fig: Matplotlib figure
    """
    # Convert to NetworkX for easier visualization
    nx_graph = convert_to_networkx(graph, positions)
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract node positions
    pos = nx.get_node_attributes(nx_graph, 'pos')
    
    # Draw nodes
    for node, (x, y, z) in pos.items():
        element = nx_graph.nodes[node]['element']
        color = nx_graph.nodes[node]['color']
        
        # Adjust size based on importance if provided
        size = 100
        if node_importance is not None:
            size = 50 + 300 * node_importance[node]
        
        # Adjust alpha based on importance if provided
        alpha = 1.0
        if node_importance is not None:
            alpha = 0.3 + 0.7 * node_importance[node]
        
        ax.scatter(x, y, z, s=size, c=color, alpha=alpha, edgecolors='black')
        
        # Add element label
        ax.text(x, y, z, element, fontsize=8)
    
    # Draw edges
    for i, (u, v) in enumerate(nx_graph.edges()):
        x = [pos[u][0], pos[v][0]]
        y = [pos[u][1], pos[v][1]]
        z = [pos[u][2], pos[v][2]]
        
        # Adjust linewidth and alpha based on importance if provided
        linewidth = 1.0
        alpha = 0.6
        if edge_importance is not None:
            linewidth = 0.5 + 3.0 * edge_importance[i]
            alpha = 0.2 + 0.8 * edge_importance[i]
        
        ax.plot(x, y, z, c='gray', linewidth=linewidth, alpha=alpha)
    
    # Draw unit cell if provided
    if box is not None:
        # Get box vectors
        a, b, c = box
        
        # Define corners of the unit cell
        corners = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1]
        ])
        
        # Transform corners to actual coordinates
        corners_transformed = np.dot(corners, box)
        
        # Define edges of the unit cell
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
        ]
        
        # Draw unit cell edges
        for start, end in edges:
            ax.plot([corners_transformed[start, 0], corners_transformed[end, 0]],
                    [corners_transformed[start, 1], corners_transformed[end, 1]],
                    [corners_transformed[start, 2], corners_transformed[end, 2]],
                    'k--', alpha=0.3)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set equal aspect ratio
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    z_lim = ax.get_zlim()
    
    x_range = max(x_lim) - min(x_lim)
    y_range = max(y_lim) - min(y_lim)
    z_range = max(z_lim) - min(z_lim)
    
    max_range = max(x_range, y_range, z_range)
    
    ax.set_xlim([(x_lim[0] + x_lim[1]) / 2 - max_range / 2,
                 (x_lim[0] + x_lim[1]) / 2 + max_range / 2])
    ax.set_ylim([(y_lim[0] + y_lim[1]) / 2 - max_range / 2,
                 (y_lim[0] + y_lim[1]) / 2 + max_range / 2])
    ax.set_zlim([(z_lim[0] + z_lim[1]) / 2 - max_range / 2,
                 (z_lim[0] + z_lim[1]) / 2 + max_range / 2])
    
    # Add colorbar for importance if provided
    if node_importance is not None or edge_importance is not None:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                  norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label="Feature Importance")
    
    # Save figure if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_counterfactual_comparison(original_graph: GraphsTuple, 
                                      original_positions: Array,
                                      cf_graph: GraphsTuple,
                                      cf_positions: Array,
                                      box: Optional[Array] = None,
                                      highlighted_changes: Optional[List[int]] = None,
                                      title: str = "Counterfactual Comparison",
                                      save_path: Optional[str] = None) -> plt.Figure:
    """Visualize comparison between original and counterfactual structures.
    
    Args:
        original_graph: Original crystal graph
        original_positions: Original atomic positions
        cf_graph: Counterfactual crystal graph
        cf_positions: Counterfactual atomic positions
        box: Unit cell box (optional)
        highlighted_changes: Indices of atoms that changed (optional)
        title: Plot title
        save_path: Path to save the figure (optional)
        
    Returns:
        fig: Matplotlib figure
    """
    # Convert to NetworkX for easier visualization
    original_nx = convert_to_networkx(original_graph, original_positions)
    cf_nx = convert_to_networkx(cf_graph, cf_positions)
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(16, 8))
    
    # Original structure
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Extract node positions for original
    pos_orig = nx.get_node_attributes(original_nx, 'pos')
    
    # Draw nodes for original
    for node, (x, y, z) in pos_orig.items():
        element = original_nx.nodes[node]['element']
        color = original_nx.nodes[node]['color']
        
        # Highlight changes if provided
        if highlighted_changes is not None and node in highlighted_changes:
            ax1.scatter(x, y, z, s=150, c=color, edgecolors='red', linewidth=2)
        else:
            ax1.scatter(x, y, z, s=100, c=color, edgecolors='black')
        
        # Add element label
        ax1.text(x, y, z, element, fontsize=8)
    
    # Draw edges for original
    for u, v in original_nx.edges():
        x = [pos_orig[u][0], pos_orig[v][0]]
        y = [pos_orig[u][1], pos_orig[v][1]]
        z = [pos_orig[u][2], pos_orig[v][2]]
        ax1.plot(x, y, z, c='gray', linewidth=1.0, alpha=0.6)
    
    # Counterfactual structure
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Extract node positions for counterfactual
    pos_cf = nx.get_node_attributes(cf_nx, 'pos')
    
    # Draw nodes for counterfactual
    for node, (x, y, z) in pos_cf.items():
        element = cf_nx.nodes[node]['element']
        color = cf_nx.nodes[node]['color']
        
        # Highlight changes if provided
        if highlighted_changes is not None and node in highlighted_changes:
            ax2.scatter(x, y, z, s=150, c=color, edgecolors='red', linewidth=2)
        else:
            ax2.scatter(x, y, z, s=100, c=color, edgecolors='black')
        
        # Add element label
        ax2.text(x, y, z, element, fontsize=8)
    
    # Draw edges for counterfactual
    for u, v in cf_nx.edges():
        x = [pos_cf[u][0], pos_cf[v][0]]
        y = [pos_cf[u][1], pos_cf[v][1]]
        z = [pos_cf[u][2], pos_cf[v][2]]
        ax2.plot(x, y, z, c='gray', linewidth=1.0, alpha=0.6)
    
    # Draw unit cell if provided
    if box is not None:
        # Draw for both subplots using similar logic as above
        # (Code omitted for brevity)
        pass
    
    # Set labels and titles
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title("Original Structure")
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title("Counterfactual Structure")
    
    fig.suptitle(title, fontsize=16)
    
    # Save figure if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_feature_importance_evolution(importance_evolution: Dict[str, List[Array]],
                                         feature_names: Optional[List[str]] = None,
                                         title: str = "Feature Importance Evolution",
                                         save_path: Optional[str] = None) -> plt.Figure:
    """Visualize the evolution of feature importance across active learning rounds.
    
    Args:
        importance_evolution: Dictionary with feature importance scores over rounds
        feature_names: Names of features (optional)
        title: Plot title
        save_path: Path to save the figure (optional)
        
    Returns:
        fig: Matplotlib figure
    """
    # Create figure
    fig, axs = plt.subplots(len(importance_evolution), 1, figsize=(10, 4 * len(importance_evolution)))
    
    # If only one type of feature, axs will not be an array
    if len(importance_evolution) == 1:
        axs = [axs]
    
    for i, (feature_type, importances) in enumerate(importance_evolution.items()):
        # Convert to numpy array
        importances = np.array(importances)
        
        # Number of rounds and features
        num_rounds, num_features = importances.shape
        
        # Create heatmap
        im = axs[i].imshow(importances.T, cmap='viridis', aspect='auto')
        
        # Set labels
        axs[i].set_xlabel('Active Learning Round')
        axs[i].set_ylabel(f'{feature_type.capitalize()} Features')
        axs[i].set_title(f'{feature_type.capitalize()} Feature Importance')
        
        # Set ticks
        axs[i].set_xticks(np.arange(num_rounds))
        axs[i].set_xticklabels([f'Round {j+1}' for j in range(num_rounds)])
        
        if feature_names is not None and len(feature_names) == num_features:
            axs[i].set_yticks(np.arange(num_features))
            axs[i].set_yticklabels(feature_names)
        
        # Add colorbar
        plt.colorbar(im, ax=axs[i], label='Feature Importance')
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_stability_motifs(motifs: Dict,
                              title: str = "Identified Stability Motifs",
                              save_path: Optional[str] = None) -> plt.Figure:
    """Visualize identified stability motifs.
    
    Args:
        motifs: Dictionary with identified stability motifs
        title: Plot title
        save_path: Path to save the figure (optional)
        
    Returns:
        fig: Matplotlib figure
    """
    # Create figure with multiple subplots based on motif types
    num_motif_types = len(motifs)
    fig, axs = plt.subplots(1, num_motif_types, figsize=(6 * num_motif_types, 6))
    
    # If only one type of motif, axs will not be an array
    if num_motif_types == 1:
        axs = [axs]
    
    # Iterate through motif types
    for i, (motif_type, values) in enumerate(motifs.items()):
        # Extract motif names and scores
        motif_names = list(values.keys())
        motif_scores = list(values.values())
        
        # Sort by score
        sorted_indices = np.argsort(motif_scores)[::-1]  # Descending order
        sorted_names = [motif_names[j] for j in sorted_indices]
        sorted_scores = [motif_scores[j] for j in sorted_indices]
        
        # Create horizontal bar chart
        axs[i].barh(sorted_names, sorted_scores, color='skyblue')
        
        # Add score values on the bars
        for j, score in enumerate(sorted_scores):
            axs[i].text(score + 0.01, j, f'{score:.2f}', va='center')
        
        # Set labels and title
        axs[i].set_xlabel('Importance Score')
        axs[i].set_title(motif_type.replace('_', ' ').title())
        
        # Adjust x-axis limits
        axs[i].set_xlim([0, 1.1 * max(sorted_scores)])
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_explanation_comparison(explanations: Dict[str, Dict],
                                    graph: GraphsTuple,
                                    positions: Array,
                                    title: str = "XAI Method Comparison",
                                    save_path: Optional[str] = None) -> plt.Figure:
    """Visualize comparison of explanations from different XAI methods.
    
    Args:
        explanations: Dictionary with explanations from different methods
        graph: Crystal graph
        positions: Atomic positions
        title: Plot title
        save_path: Path to save the figure (optional)
        
    Returns:
        fig: Matplotlib figure
    """
    # Number of methods
    num_methods = len(explanations)
    
    # Create figure
    fig = plt.figure(figsize=(6 * num_methods, 8))
    
    # Iterate through methods
    for i, (method_name, explanation) in enumerate(explanations.items()):
        # Create 3D subplot
        ax = fig.add_subplot(1, num_methods, i + 1, projection='3d')
        
        # Convert to NetworkX for easier visualization
        nx_graph = convert_to_networkx(graph, positions)
        
        # Extract node positions
        pos = nx.get_node_attributes(nx_graph, 'pos')
        
        # Extract node importance from explanation
        if 'nodes' in explanation:
            node_importance = jnp.abs(explanation['nodes']).mean(axis=1)
            # Normalize
            node_importance = node_importance / node_importance.max()
        elif 'node_mask' in explanation:
            node_importance = explanation['node_mask']
        else:
            node_importance = None
        
        # Extract edge importance from explanation
        if 'edges' in explanation:
            edge_importance = jnp.abs(explanation['edges']).mean(axis=1)
            # Normalize
            edge_importance = edge_importance / edge_importance.max()
        elif 'edge_mask' in explanation:
            edge_importance = explanation['edge_mask']
        else:
            edge_importance = None
        
        # Draw nodes
        for node, (x, y, z) in pos.items():
            element = nx_graph.nodes[node]['element']
            color = nx_graph.nodes[node]['color']
            
            # Adjust size based on importance if provided
            size = 100
            if node_importance is not None:
                size = 50 + 300 * node_importance[node]
            
            # Adjust alpha based on importance if provided
            alpha = 1.0
            if node_importance is not None:
                alpha = 0.3 + 0.7 * node_importance[node]
            
            ax.scatter(x, y, z, s=size, c=color, alpha=alpha, edgecolors='black')
            
            # Add element label
            ax.text(x, y, z, element, fontsize=8)
        
        # Draw edges
        for j, (u, v) in enumerate(nx_graph.edges()):
            x = [pos[u][0], pos[v][0]]
            y = [pos[u][1], pos[v][1]]
            z = [pos[u][2], pos[v][2]]
            
            # Adjust linewidth and alpha based on importance if provided
            linewidth = 1.0
            alpha = 0.6
            if edge_importance is not None:
                linewidth = 0.5 + 3.0 * edge_importance[j]
                alpha = 0.2 + 0.8 * edge_importance[j]
            
            ax.plot(x, y, z, c='gray', linewidth=linewidth, alpha=alpha)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(method_name)
        
        # Set equal aspect ratio
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        z_lim = ax.get_zlim()
        
        x_range = max(x_lim) - min(x_lim)
        y_range = max(y_lim) - min(y_lim)
        z_range = max(z_lim) - min(z_lim)
        
        max_range = max(x_range, y_range, z_range)
        
        ax.set_xlim([(x_lim[0] + x_lim[1]) / 2 - max_range / 2,
                    (x_lim[0] + x_lim[1]) / 2 + max_range / 2])
        ax.set_ylim([(y_lim[0] + y_lim[1]) / 2 - max_range / 2,
                    (y_lim[0] + y_lim[1]) / 2 + max_range / 2])
        ax.set_zlim([(z_lim[0] + z_lim[1]) / 2 - max_range / 2,
                    (z_lim[0] + z_lim[1]) / 2 + max_range / 2])
        
        # Add colorbar for importance if provided
        if node_importance is not None or edge_importance is not None:
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                      norm=plt.Normalize(0, 1))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, label="Feature Importance")
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_benchmark_results(benchmark_results: Dict[str, Dict[str, float]],
                               title: str = "XAI Method Benchmark Results",
                               save_path: Optional[str] = None) -> plt.Figure:
    """Visualize benchmark results for different XAI methods.
    
    Args:
        benchmark_results: Dictionary with benchmark results
        title: Plot title
        save_path: Path to save the figure (optional)
        
    Returns:
        fig: Matplotlib figure
    """
    # Extract method names and metrics
    method_names = list(benchmark_results.keys())
    all_metrics = set()
    
    for results in benchmark_results.values():
        all_metrics.update(results.keys())
    
    # Remove avg_time from metrics (will be displayed separately)
    if 'avg_time' in all_metrics:
        all_metrics.remove('avg_time')
    
    all_metrics = sorted(list(all_metrics))
    
    # Create figure with two subplots: one for metrics, one for time
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Set width of bars
    bar_width = 0.8 / len(method_names)
    
    # Set positions for bars
    positions = np.arange(len(all_metrics))
    
    # Plot metrics
    for i, method in enumerate(method_names):
        # Extract metric values for this method
        metric_values = []
        for metric in all_metrics:
            metric_values.append(benchmark_results[method].get(metric, 0))
        
        # Plot bars
        ax1.bar(positions + i * bar_width, metric_values, 
               width=bar_width, label=method)
    
    # Set labels and title for metrics subplot
    ax1.set_xlabel('Metric')
    ax1.set_ylabel('Score')
    ax1.set_title('Performance Metrics')
    ax1.set_xticks(positions + (len(method_names) - 1) * bar_width / 2)
    ax1.set_xticklabels(all_metrics)
    ax1.legend()
    
    # Plot computation time
    time_values = [benchmark_results[method].get('avg_time', 0) for method in method_names]
    ax2.bar(method_names, time_values, color='skyblue')
    
    # Set labels and title for time subplot
    ax2.set_xlabel('Method')
    ax2.set_ylabel('Average Time (s)')
    ax2.set_title('Computational Cost')
    
    # Add time values on the bars
    for i, time_val in enumerate(time_values):
        ax2.text(i, time_val + 0.05 * max(time_values), 
                f'{time_val:.2f}s', ha='center')
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def visualize_chemical_concept_test(concept_sensitivities: Dict[str, float],
                                   title: str = "Chemical Concept Test Results",
                                   save_path: Optional[str] = None) -> plt.Figure:
    """Visualize results of testing for chemical concepts in the model.
    
    Args:
        concept_sensitivities: Dictionary with concept names and sensitivity scores
        title: Plot title
        save_path: Path to save the figure (optional)
        
    Returns:
        fig: Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract concept names and sensitivities
    concepts = list(concept_sensitivities.keys())
    sensitivities = list(concept_sensitivities.values())
    
    # Sort by sensitivity
    sorted_indices = np.argsort(sensitivities)[::-1]  # Descending order
    sorted_concepts = [concepts[i] for i in sorted_indices]
    sorted_sensitivities = [sensitivities[i] for i in sorted_indices]
    
    # Create horizontal bar chart
    ax.barh(sorted_concepts, sorted_sensitivities, color='skyblue')
    
    # Add sensitivity values on the bars
    for i, sensitivity in enumerate(sorted_sensitivities):
        ax.text(sensitivity + 0.01, i, f'{sensitivity:.3f}', va='center')
    
    # Set labels and title
    ax.set_xlabel('Sensitivity')
    ax.set_ylabel('Chemical Concept')
    ax.set_title(title)
    
    # Adjust x-axis limits
    ax.set_xlim([0, 1.1 * max(sorted_sensitivities)])
    
    # Add a vertical line at a significance threshold (e.g., 0.05)
    ax.axvline(x=0.05, color='red', linestyle='--', label='Significance Threshold')
    ax.legend()
    
    # Save figure if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def visualize_novel_stability_drivers(novel_drivers: List[Dict],
                                     title: str = "Novel Stability Drivers",
                                     save_path: Optional[str] = None) -> plt.Figure:
    """Visualize novel stability drivers identified by XAI.
    
    Args:
        novel_drivers: List of dictionaries with novel stability drivers
        title: Plot title
        save_path: Path to save the figure (optional)
        
    Returns:
        fig: Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract driver names and scores
    driver_names = [driver['name'] for driver in novel_drivers]
    driver_scores = [driver['score'] for driver in novel_drivers]
    driver_descriptions = [driver['description'] for driver in novel_drivers]
    
    # Sort by score
    sorted_indices = np.argsort(driver_scores)[::-1]  # Descending order
    sorted_names = [driver_names[i] for i in sorted_indices]
    sorted_scores = [driver_scores[i] for i in sorted_indices]
    sorted_descriptions = [driver_descriptions[i] for i in sorted_indices]
    
    # Create horizontal bar chart
    bars = ax.barh(sorted_names, sorted_scores, color='skyblue')
    
    # Add score values on the bars
    for i, score in enumerate(sorted_scores):
        ax.text(score + 0.01, i, f'{score:.2f}', va='center')
    
    # Set labels and title
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Novel Driver')
    ax.set_title(title)
    
    # Adjust x-axis limits
    ax.set_xlim([0, 1.1 * max(sorted_scores)])
    
    # Add tooltips with descriptions
    tooltip_text = ""
    for name, description in zip(sorted_names, sorted_descriptions):
        tooltip_text += f"{name}: {description}\n"
    
    # Create a text box with descriptions
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.05, tooltip_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='bottom', bbox=props)
    
    # Save figure if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig