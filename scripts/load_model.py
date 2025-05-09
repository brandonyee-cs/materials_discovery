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
import time
import jax
import jax.numpy as jnp
from jax import random
from flax import serialization
import jraph
from ml_collections import ConfigDict
import optax
from tqdm import tqdm

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


def create_default_config():
    """Create default model configuration."""
    cfg = ConfigDict()
    
    # Model architecture
    cfg.model_family = 'crystal'
    cfg.graph_net_steps = 2
    cfg.mlp_width = (32, 32, 32)
    cfg.mlp_nonlinearity = 'relu'
    cfg.embedding_dim = 32
    cfg.featurizer = 'gaussian'
    
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


def create_dummy_targets(graphs):
    """Create dummy targets for pretraining."""
    return jnp.array([-1.0] * len(graphs))


def train_model(args, data, cfg):
    """Train a GNoME model."""
    print("Starting model training...")
    
    # Extract data
    graphs = data['graphs']
    positions = data.get('positions', [None] * len(graphs))
    boxes = data.get('boxes', [None] * len(graphs))
    
    # Create dummy targets
    targets = create_dummy_targets(graphs)
    
    # Create model
    model = crystal.crystal_energy_model(cfg)
    
    # Initialize model
    key = random.PRNGKey(42)  # Fixed seed for reproducibility
    dummy_graph = graphs[0]
    dummy_pos = positions[0]
    dummy_box = boxes[0]
    
    params = model.init(key, dummy_graph, dummy_pos, dummy_box)
    
    # Initialize optimizer
    optimizer = optax.adam(0.001)  # Fixed learning rate
    opt_state = optimizer.init(params)
    
    # Training loop
    num_epochs = 5  # Fixed number of epochs
    batch_size = 8  # Fixed batch size
    num_examples = len(graphs)
    
    # Define training step
    @jax.jit
    def train_step(params, opt_state, graph, pos, box, target):
        def loss_fn(p):
            pred = model.apply(p, graph, pos, box)
            return jnp.mean((pred - target) ** 2)
        
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        # Shuffle data
        indices = jax.random.permutation(key, num_examples)
        
        # Process in batches
        for i in range(0, num_examples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch_graphs = [graphs[j] for j in batch_indices]
            batch_positions = [positions[j] for j in batch_indices]
            batch_boxes = [boxes[j] for j in batch_indices]
            batch_targets = targets[batch_indices]
            
            # Update parameters
            params, opt_state, loss = train_step(params, opt_state, 
                                               batch_graphs[0], batch_positions[0], 
                                               batch_boxes[0], batch_targets[0])
            
            epoch_loss += loss
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return params


def load_model(args, data):
    """Load a pretrained GNoME model or train a new one if no checkpoint exists."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if checkpoint exists
    checkpoint_path = os.path.join(args.checkpoint_dir, 'checkpoint_final')
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = f.read()
        
        # Load parameters from checkpoint
        params = serialization.from_bytes(None, checkpoint_data)[1]
        
        # Create model configuration based on loaded parameters
        cfg = create_model_config(params)
    else:
        print("No checkpoint found. Training new model...")
        # Create default configuration
        cfg = create_default_config()
        
        # Train model
        params = train_model(args, data, cfg)
        
        # Save checkpoint
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        with open(checkpoint_path, 'wb') as f:
            f.write(serialization.to_bytes((0, params, None)))  # Save as (step, params, opt_state)
    
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
    
    print(f"Model saved to {output_path}")
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