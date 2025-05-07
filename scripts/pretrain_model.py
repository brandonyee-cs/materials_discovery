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
Pretrain a GNoME model for XAI analysis.

This script creates and trains a simplified GNoME model.
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
    parser = argparse.ArgumentParser(description="Pretrain a GNoME model for XAI analysis.")
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to crystal data pickle file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the trained model')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()


def load_data(data_file):
    """Load crystal data."""
    print(f"Loading data from {data_file}...")
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    return data


def create_dummy_targets(graphs):
    """Create dummy target values for training."""
    # In a real scenario, we would use actual formation energies
    # Here we create random values for demonstration
    key = random.PRNGKey(0)
    keys = random.split(key, len(graphs))
    targets = []
    
    for i, key in enumerate(keys):
        # Create a random value between -5 and 0
        target = -5.0 + 5.0 * random.uniform(key)
        targets.append(target)
    
    return jnp.array(targets)


def create_model_config():
    """Create model configuration."""
    cfg = ConfigDict()
    
    # Model architecture - simplified for speed
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
    
    # Training settings
    cfg.schedule = 'constant'
    cfg.learning_rate = 0.001
    cfg.l2_regularization = 0.0
    
    return cfg


def train_model(args, data, cfg):
    """Train a GNoME model."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        f.write(json.dumps(json.dumps(cfg.to_dict())))
    
    # Extract data
    graphs = data['graphs']
    positions = data.get('positions', [None] * len(graphs))
    boxes = data.get('boxes', [None] * len(graphs))
    
    # Create dummy targets
    targets = create_dummy_targets(graphs)
    
    # Create model
    model = crystal.crystal_energy_model(cfg)
    
    # Initialize model
    key = random.PRNGKey(args.seed)
    dummy_graph = graphs[0]
    dummy_pos = positions[0]
    dummy_box = boxes[0]
    
    params = model.init(key, dummy_graph, dummy_pos, dummy_box)
    
    # Initialize optimizer
    optimizer = optax.adam(args.learning_rate)
    opt_state = optimizer.init(params)
    
    # Training loop
    print(f"Training for {args.epochs} epochs...")
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
    
    for epoch in range(args.epochs):
        start_time = time.time()
        epoch_loss = 0.0
        
        # Process each example
        for i in tqdm(range(num_examples), desc=f"Epoch {epoch+1}/{args.epochs}"):
            params, opt_state, loss = train_step(
                params, opt_state, graphs[i], positions[i], boxes[i], targets[i])
            epoch_loss += loss
        
        # Average loss
        epoch_loss /= num_examples
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss:.6f}, Time: {elapsed:.2f}s")
    
    # Save model
    checkpoint_path = os.path.join(args.output_dir, 'checkpoint_final')
    with open(checkpoint_path, 'wb') as f:
        f.write(serialization.to_bytes((0, params, opt_state)))
    
    print(f"Model training complete! Saved to {checkpoint_path}")


def main(args):
    """Main function for model pretraining."""
    # Load data
    data = load_data(args.data_file)
    
    # Create model configuration
    cfg = create_model_config()
    
    # Update with command-line arguments
    cfg.learning_rate = args.learning_rate
    
    # Train model
    train_model(args, data, cfg)


if __name__ == '__main__':
    args = parse_args()
    main(args)