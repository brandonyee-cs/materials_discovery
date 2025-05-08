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
Load a pretrained GNoME model for XAI analysis.

This script loads a pretrained GNoME model from a checkpoint.
"""

import os
import argparse
import pickle
import json
import jax
import jax.numpy as jnp
from flax import serialization
import jraph
from ml_collections import ConfigDict

from GNoME import crystal, gnome


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Load a pretrained GNoME model for XAI analysis.")
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory containing the model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the loaded model')
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to crystal data pickle file (for feature dimensions)')
    return parser.parse_args()


def load_data(data_file):
    """Load crystal data to get feature dimensions."""
    print(f"Loading data from {data_file}...")
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    return data


def create_model_config(params):
    """Create model configuration based on loaded parameters."""
    cfg = ConfigDict()
    
    # Extract dimensions from loaded parameters
    node_embed_dim = params['params']['Node Embedding']['kernel'].shape[0]
    edge_embed_dim = params['params']['Edge Embedding']['kernel'].shape[0]
    
    # Model architecture
    cfg.model_family = 'crystal'
    cfg.graph_net_steps = 2
    cfg.mlp_width = (32, 32, 32)
    cfg.mlp_nonlinearity = 'relu'
    cfg.embedding_dim = 32
    cfg.featurizer = 'gaussian'
    
    # Set input dimensions based on loaded parameters
    cfg.node_input_dim = node_embed_dim
    cfg.edge_input_dim = edge_embed_dim
    
    # Aggregation functions
    cfg.node_aggregation = 'mean'
    cfg.edges_for_globals_aggregation = 'mean'
    cfg.readout_edges_for_globals_aggregation = 'mean'
    
    # Other options
    cfg.feature_band_limit = 0
    cfg.conditioning_band_limit = 0
    cfg.extra_scalars_for_gating = False
    cfg.residual = 'none'
    
    return cfg


def load_model(args, data):
    """Load a pretrained GNoME model."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load checkpoint
    checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint_final')
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    with open(checkpoint_path, 'rb') as f:
        checkpoint_data = f.read()
    
    # Load parameters from checkpoint
    params = serialization.from_bytes(None, checkpoint_data)[1]  # [1] gets the params from (step, params, opt_state)
    
    # Create model configuration based on loaded parameters
    cfg = create_model_config(params)
    
    # Save configuration
    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        f.write(json.dumps(cfg.to_dict()))
    
    # Create model with correct dimensions
    model = crystal.crystal_energy_model(cfg)
    
    # Save loaded model
    output_path = os.path.join(args.output_dir, 'model')
    with open(output_path, 'wb') as f:
        f.write(serialization.to_bytes(params))
    
    print(f"Model loaded successfully! Saved to {output_path}")
    print(f"Model configured with node_input_dim={cfg.node_input_dim}, edge_input_dim={cfg.edge_input_dim}")


def main(args):
    """Main function for loading a pretrained model."""
    # Load data to get feature dimensions
    data = load_data(args.data_file)
    
    # Load model
    load_model(args, data)


if __name__ == '__main__':
    args = parse_args()
    main(args) 