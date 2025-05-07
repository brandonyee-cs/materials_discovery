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
Benchmarking framework for XAI methods in materials science.

This module provides tools to compare XAI methods' fidelity, interpretability, 
and cost on GNoME's dataset, with metrics for explanation actionability.
"""

import jax
import jax.numpy as jnp
import jraph
import time
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from functools import partial

from GNoME import crystal, gnome, gnn
from .representation_analysis import GNNExplainer, IntegratedGradients, SHAP

Array = jnp.ndarray
GraphsTuple = jraph.GraphsTuple


class XAIMetric:
    """Base class for XAI evaluation metrics."""
    
    def __init__(self, name: str):
        """Initialize XAI metric.
        
        Args:
            name: Name of the metric
        """
        self.name = name
    
    def compute(self, explanation: Dict, reference: Optional[Dict] = None) -> float:
        """Compute metric value.
        
        Args:
            explanation: XAI explanation to evaluate
            reference: Optional reference explanation or ground truth
            
        Returns:
            score: Metric value
        """
        raise NotImplementedError("Subclasses must implement compute method")


class ExplanationFidelity(XAIMetric):
    """Measure how faithfully an explanation represents the model's behavior."""
    
    def __init__(self, model: Callable):
        """Initialize explanation fidelity metric.
        
        Args:
            model: The GNoME model
        """
        super().__init__("fidelity")
        self.model = model
    
    def compute(self, 
               explanation: Dict, 
               graph: GraphsTuple, 
               reference: Optional[Dict] = None) -> float:
        """Compute fidelity score.
        
        Args:
            explanation: XAI explanation to evaluate
            graph: Input graph
            reference: Optional reference explanation
            
        Returns:
            fidelity: Fidelity score (higher is better)
        """
        # We measure fidelity by how well the explanation predicts the effect of
        # perturbations on the model's output
        
        # Get original prediction
        original_pred = self.model(graph, None, None)
        
        # Create perturbed graph based on explanation
        # For example, if it's a node mask, we zero out the least important nodes
        if 'nodes' in explanation:
            node_importance = jnp.abs(explanation['nodes']).mean(axis=1)
            threshold = jnp.percentile(node_importance, 50)  # Zero out bottom 50%
            node_mask = (node_importance > threshold).astype(jnp.float32)
            
            perturbed_graph = jraph.GraphsTuple(
                nodes=graph.nodes * node_mask[:, None],
                edges=graph.edges,
                receivers=graph.receivers,
                senders=graph.senders,
                globals=graph.globals,
                n_node=graph.n_node,
                n_edge=graph.n_edge
            )
            
            # Get prediction for perturbed graph
            perturbed_pred = self.model(perturbed_graph, None, None)
            
            # Compute fidelity (normalized change in prediction)
            fidelity = jnp.abs(perturbed_pred - original_pred) / (jnp.abs(original_pred) + 1e-10)
            
            return float(fidelity)
        
        # If it's an edge mask, we would do something similar
        elif 'edge_mask' in explanation:
            edge_mask = explanation['edge_mask']
            threshold = jnp.percentile(edge_mask, 50)  # Zero out bottom 50%
            mask = (edge_mask > threshold).astype(jnp.float32)
            
            perturbed_graph = jraph.GraphsTuple(
                nodes=graph.nodes,
                edges=graph.edges * mask[:, None],
                receivers=graph.receivers,
                senders=graph.senders,
                globals=graph.globals,
                n_node=graph.n_node,
                n_edge=graph.n_edge
            )
            
            # Get prediction for perturbed graph
            perturbed_pred = self.model(perturbed_graph, None, None)
            
            # Compute fidelity (normalized change in prediction)
            fidelity = jnp.abs(perturbed_pred - original_pred) / (jnp.abs(original_pred) + 1e-10)
            
            return float(fidelity)
        
        return 0.0


class ExplanationSparsity(XAIMetric):
    """Measure how concise an explanation is."""
    
    def __init__(self):
        """Initialize explanation sparsity metric."""
        super().__init__("sparsity")
    
    def compute(self, explanation: Dict, reference: Optional[Dict] = None) -> float:
        """Compute sparsity score.
        
        Args:
            explanation: XAI explanation to evaluate
            reference: Optional reference explanation
            
        Returns:
            sparsity: Sparsity score (higher is better)
        """
        if 'nodes' in explanation:
            node_importance = jnp.abs(explanation['nodes']).mean(axis=1)
            # Normalize to sum to 1
            node_importance = node_importance / (jnp.sum(node_importance) + 1e-10)
            # Compute Gini coefficient as a measure of sparsity
            sorted_importance = jnp.sort(node_importance)
            n = len(sorted_importance)
            gini = 2 * jnp.sum((jnp.arange(1, n + 1) * sorted_importance)) / (n * jnp.sum(sorted_importance)) - (n + 1) / n
            return float(gini)
        
        elif 'edge_mask' in explanation:
            edge_mask = explanation['edge_mask']
            # Normalize to sum to 1
            edge_mask = edge_mask / (jnp.sum(edge_mask) + 1e-10)
            # Compute Gini coefficient
            sorted_mask = jnp.sort(edge_mask)
            n = len(sorted_mask)
            gini = 2 * jnp.sum((jnp.arange(1, n + 1) * sorted_mask)) / (n * jnp.sum(sorted_mask)) - (n + 1) / n
            return float(gini)
        
        return 0.0


class ExplanationStability(XAIMetric):
    """Measure how stable an explanation is to small perturbations in the input."""
    
    def __init__(self):
        """Initialize explanation stability metric."""
        super().__init__("stability")
    
    def compute(self, explanation1: Dict, explanation2: Dict) -> float:
        """Compute stability score.
        
        Args:
            explanation1: First XAI explanation
            explanation2: Second XAI explanation (for slightly perturbed input)
            
        Returns:
            stability: Stability score (higher is better)
        """
        if 'nodes' in explanation1 and 'nodes' in explanation2:
            # Compute similarity between explanations (cosine similarity)
            node_importance1 = jnp.abs(explanation1['nodes']).mean(axis=1)
            node_importance2 = jnp.abs(explanation2['nodes']).mean(axis=1)
            
            # Normalize
            node_importance1 = node_importance1 / (jnp.linalg.norm(node_importance1) + 1e-10)
            node_importance2 = node_importance2 / (jnp.linalg.norm(node_importance2) + 1e-10)
            
            # Compute cosine similarity
            similarity = jnp.sum(node_importance1 * node_importance2)
            
            return float(similarity)
        
        elif 'edge_mask' in explanation1 and 'edge_mask' in explanation2:
            # Compute similarity between explanations
            mask1 = explanation1['edge_mask']
            mask2 = explanation2['edge_mask']
            
            # Normalize
            mask1 = mask1 / (jnp.linalg.norm(mask1) + 1e-10)
            mask2 = mask2 / (jnp.linalg.norm(mask2) + 1e-10)
            
            # Compute cosine similarity
            similarity = jnp.sum(mask1 * mask2)
            
            return float(similarity)
        
        return 0.0


class ExplanationActionability(XAIMetric):
    """Measure how actionable an explanation is for crystal structure prediction."""
    
    def __init__(self, csp_model: Callable):
        """Initialize explanation actionability metric.
        
        Args:
            csp_model: Crystal structure prediction model
        """
        super().__init__("actionability")
        self.csp_model = csp_model
    
    def compute(self, 
               explanation: Dict, 
               graph: GraphsTuple,
               reference: Optional[Dict] = None) -> float:
        """Compute actionability score.
        
        Args:
            explanation: XAI explanation to evaluate
            graph: Input graph
            reference: Optional reference explanation
            
        Returns:
            actionability: Actionability score (higher is better)
        """
        # This would require integration with a real CSP system
        # Here we provide a placeholder
        
        # Placeholder for actionability score
        actionability = 0.75
        
        return actionability


class ComputationalCost(XAIMetric):
    """Measure the computational cost of an XAI method."""
    
    def __init__(self):
        """Initialize computational cost metric."""
        super().__init__("computational_cost")
    
    def compute(self, timings: Dict, reference: Optional[Dict] = None) -> float:
        """Compute computational cost score.
        
        Args:
            timings: Dictionary with timing information
            reference: Optional reference timings
            
        Returns:
            cost: Computational cost score (lower is better)
        """
        # We use the walltime as a measure of computational cost
        if 'walltime' in timings:
            # Normalize to a score between 0 and 1 (lower is better)
            # Assuming reference is the maximum acceptable time
            if reference is not None and 'max_acceptable_time' in reference:
                max_time = reference['max_acceptable_time']
                cost = min(1.0, timings['walltime'] / max_time)
            else:
                # If no reference, just return the raw time
                cost = timings['walltime']
            
            return float(cost)
        
        return 0.0


class XAIBenchmark:
    """Framework for benchmarking XAI methods in materials science."""
    
    def __init__(self, 
                model: Callable, 
                csp_model: Optional[Callable] = None,
                metrics: Optional[List[str]] = None):
        """Initialize XAI benchmark.
        
        Args:
            model: The GNoME model
            csp_model: Crystal structure prediction model (optional)
            metrics: List of metrics to compute
        """
        self.model = model
        self.csp_model = csp_model
        
        # Initialize metrics
        self.metrics = {}
        if metrics is None:
            metrics = ["fidelity", "sparsity", "stability", "computational_cost"]
            if csp_model is not None:
                metrics.append("actionability")
        
        for metric in metrics:
            if metric == "fidelity":
                self.metrics[metric] = ExplanationFidelity(model)
            elif metric == "sparsity":
                self.metrics[metric] = ExplanationSparsity()
            elif metric == "stability":
                self.metrics[metric] = ExplanationStability()
            elif metric == "actionability" and csp_model is not None:
                self.metrics[metric] = ExplanationActionability(csp_model)
            elif metric == "computational_cost":
                self.metrics[metric] = ComputationalCost()
    
    def evaluate_method(self, 
                       method: Callable, 
                       graphs: List[GraphsTuple],
                       reference: Optional[Dict] = None) -> Dict[str, float]:
        """Evaluate an XAI method on a set of graphs.
        
        Args:
            method: XAI method to evaluate
            graphs: List of crystal graphs
            reference: Optional reference for comparison
            
        Returns:
            results: Dictionary with evaluation results
        """
        results = {metric: [] for metric in self.metrics}
        timings = []
        
        for graph in graphs:
            # Measure time to generate explanation
            start_time = time.time()
            explanation = method(graph)
            end_time = time.time()
            walltime = end_time - start_time
            
            # Compute metrics
            for metric_name, metric in self.metrics.items():
                if metric_name == "computational_cost":
                    score = metric.compute({"walltime": walltime}, reference)
                elif metric_name == "stability":
                    # For stability, we need a slightly perturbed graph
                    # Here we just use a simplified approach
                    perturbed_graph = jraph.GraphsTuple(
                        nodes=graph.nodes + 0.01 * jax.random.normal(jax.random.PRNGKey(0), graph.nodes.shape),
                        edges=graph.edges,
                        receivers=graph.receivers,
                        senders=graph.senders,
                        globals=graph.globals,
                        n_node=graph.n_node,
                        n_edge=graph.n_edge
                    )
                    explanation2 = method(perturbed_graph)
                    score = metric.compute(explanation, explanation2)
                elif metric_name == "actionability":
                    score = metric.compute(explanation, graph, reference)
                else:
                    score = metric.compute(explanation, graph, reference)
                
                results[metric_name].append(score)
            
            timings.append(walltime)
        
        # Compute average results
        avg_results = {metric: float(np.mean(scores)) for metric, scores in results.items()}
        avg_results["avg_time"] = float(np.mean(timings))
        
        return avg_results
    
    def compare_methods(self, 
                      methods: Dict[str, Callable], 
                      graphs: List[GraphsTuple],
                      reference: Optional[Dict] = None) -> Dict[str, Dict[str, float]]:
        """Compare multiple XAI methods.
        
        Args:
            methods: Dictionary with method names and functions
            graphs: List of crystal graphs
            reference: Optional reference for comparison
            
        Returns:
            comparison: Dictionary with evaluation results for each method
        """
        comparison = {}
        
        for method_name, method_fn in methods.items():
            results = self.evaluate_method(method_fn, graphs, reference)
            comparison[method_name] = results
        
        return comparison


def define_xai_best_practices(benchmark_results: Dict[str, Dict[str, float]]) -> Dict:
    """Define best practices for XAI in materials science based on benchmark results.
    
    Args:
        benchmark_results: Dictionary with benchmark results
        
    Returns:
        best_practices: Dictionary with best practices
    """
    # This would require analysis of benchmark results
    # Here we provide a placeholder
    
    # Find best method for each metric
    best_methods = {}
    for metric in next(iter(benchmark_results.values())).keys():
        if metric == "avg_time":
            continue
        
        best_score = -float('inf')
        best_method = None
        
        for method, results in benchmark_results.items():
            if metric == "computational_cost":
                # For cost, lower is better
                if results[metric] < best_score or best_method is None:
                    best_score = results[metric]
                    best_method = method
            else:
                # For other metrics, higher is better
                if results[metric] > best_score:
                    best_score = results[metric]
                    best_method = method
        
        best_methods[metric] = (best_method, best_score)
    
    # Generate best practices based on benchmark results
    best_practices = {
        "recommended_methods": {
            "for_high_fidelity": best_methods["fidelity"][0],
            "for_interpretability": best_methods["sparsity"][0],
            "for_low_cost": min(benchmark_results.items(), key=lambda x: x[1]["avg_time"])[0]
        },
        "guidelines": [
            "Use gradient-based methods for high-fidelity explanations",
            "For interactive exploration, prefer faster methods even with slightly lower fidelity",
            "Combine multiple XAI methods for more robust interpretations"
        ],
        "recommended_visualizations": [
            "3D crystal structures with highlighted important atoms/bonds",
            "Heatmaps for feature importance",
            "Comparative visualization for counterfactuals"
        ]
    }
    
    return best_practices