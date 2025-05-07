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
XAI Module for GNoME: Explainable AI techniques for Graph Networks for Materials Exploration.

This module provides tools to make GNoME's predictions interpretable by implementing:
1. Analysis of learned representations (GNNExplainer, SHAP, Integrated Gradients)
2. Counterfactual explanations to guide synthesis
3. XAI-driven refinement for crystal structure prediction
4. Benchmarking framework for XAI methods in materials science
"""

from . import representation_analysis
from . import counterfactual
from . import iterative_refinement
from . import benchmarking
from . import visualizations

__all__ = [
    'representation_analysis',
    'counterfactual',
    'iterative_refinement',
    'benchmarking',
    'visualizations',
]