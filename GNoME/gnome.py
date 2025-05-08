# Copyright 2023 Google LLC
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

import json
import os
from typing import NamedTuple, Tuple

import e3nn_jax as e3nn
from flax import serialization
import flax.linen as nn
import jax
from jax import eval_shape
from jax import random
from jax.tree_util import tree_map
from jax.core import ShapedArray
import jax.numpy as jnp
from jax_md import util
import jraph
from ml_collections import ConfigDict
import optax

from . import nequip


f32 = jnp.float32
i32 = jnp.int32

GraphsTuple = jraph.GraphsTuple

IrrepsArray = e3nn.IrrepsArray

NUM_ELEMENTS = 94

PyTree = util.PyTree

def ensure_config_defaults(cfg: ConfigDict) -> ConfigDict:
  """Ensure config has all required keys with default values."""
  # List of required keys with their default values
  defaults = {
    'epochs': 1,
    'epoch_size': 100,
    'train_batch_size': 1,
    'warmup_steps': 0,
    'l2_regularization': 0.0,
    'schedule': 'constant',
    'learning_rate': 0.001,
  }
  
  # Add any missing keys
  for key, default_value in defaults.items():
    if not hasattr(cfg, key):
      cfg[key] = default_value
  
  return cfg

def model_from_config(cfg: ConfigDict) -> nn.Module:
  model_family = cfg.get('model_family', 'nequip')
  if model_family == 'nequip':
    return nequip.model_from_config(cfg)
  elif model_family == 'crystal':
    from . import crystal
    return crystal.crystal_energy_model(cfg)
  else:
    raise ValueError(f'Unrecognized model family: {model_family}')

def minimum_batch_size(cfg: ConfigDict) -> int:
  if not hasattr(cfg, 'train_batch_size'):
    return 1
  if isinstance(cfg.train_batch_size, int):
    return cfg.train_batch_size
  return min(cfg.train_batch_size)


class ScaleLROnPlateau(NamedTuple):
  step_size: jnp.ndarray
  minimum_loss: jnp.ndarray
  steps_without_reduction: jnp.ndarray
  max_steps_without_reduction: jnp.ndarray
  reduction_factor: jnp.ndarray


def scale_lr_on_plateau(
    initial_step_size: float,
    max_steps_without_reduction: int,
    reduction_factor: float,
) -> optax.GradientTransformation:
  def init_fn(params):
    del params
    return ScaleLROnPlateau(
        initial_step_size,
        jnp.inf,
        0,
        max_steps_without_reduction,
        reduction_factor,
    )

  def update_fn(updates, state, params=None):
    del params
    updates = jax.tree_util.tree_map(lambda g: g * state.step_size, updates)
    return updates, state

  return optax.GradientTransformation(init_fn, update_fn)

def optimizer(cfg: ConfigDict) -> optax.OptState:
  epoch_size = cfg.epoch_size if hasattr(cfg, 'epoch_size') else -1
  batch_size = minimum_batch_size(cfg)
  
  # Add default value for epochs
  epochs = cfg.epochs if hasattr(cfg, 'epochs') else 1
  total_steps = epochs * (epoch_size // batch_size)
  
  # Add default value for warmup_steps
  warmup_steps = cfg.get('warmup_steps', 0)

  if cfg.schedule == 'constant':
    schedule = cfg.learning_rate
  elif cfg.schedule == 'linear_decay':
    schedule = optax.polynomial_schedule(cfg.learning_rate, 0.0, 1, total_steps)
  elif cfg.schedule == 'cosine_decay':
    schedule = optax.cosine_decay_schedule(cfg.learning_rate, total_steps)
  elif cfg.schedule == 'warmup_cosine_decay':
    schedule = optax.warmup_cosine_decay_schedule(
        1e-7, cfg.learning_rate, warmup_steps, total_steps
    )
  elif cfg.schedule == 'scale_on_plateau':
    max_plateau_steps = cfg.max_lr_plateau_epochs // cfg.epochs_per_eval
    return optax.chain(
        optax.scale_by_adam(),
        scale_lr_on_plateau(-cfg.learning_rate, max_plateau_steps, 0.8),
    )
  else:
    raise ValueError(f'Unknown learning rate schedule, "{cfg.schedule}".')

  if not hasattr(cfg, 'l2_regularization') or cfg.l2_regularization == 0.0:
    return optax.adam(schedule)

  return optax.adamw(schedule, weight_decay=cfg.l2_regularization)


def load_model(directory: str) -> Tuple[ConfigDict, nn.Module, PyTree]:
  with open(os.path.join(directory, 'config.json'), 'r') as f:
    c = json.loads(json.loads(f.read()))
    c = ConfigDict(c)
    
  # Ensure config has all required keys
  c = ensure_config_defaults(c)

  # Now initialize the model and the optimizer functions.
  model = model_from_config(c)
  opt_init, _ = optimizer(c)
  
  # Create dummy data for initialization
  graph = GraphsTuple(
      ShapedArray((1, NUM_ELEMENTS), f32),  # Nodes     (nodes, features)
      (ShapedArray((1, 3), f32), ShapedArray((1, 3), f32)),  # (edge_features, translations)
      ShapedArray((1,), i32),  # senders   (edges,)
      ShapedArray((1,), i32),  # receivers (edges,)
      ShapedArray((1, 1), f32),  # globals   (graphs,)
      ShapedArray((1,), i32),  # n_node    (graphs,)
      ShapedArray((1,), i32),  # n_edge    (graphs,)
  )
  positions = ShapedArray((1, 3), f32)  # Single atom at origin
  box = ShapedArray((3, 3), f32)  # Identity matrix for box

  def init_opt_and_model(graph, positions, box):
    key = random.PRNGKey(0)
    params = model.init(key, graph, positions, box)
    state = opt_init(params)
    return params, state

  abstract_params, abstract_state = eval_shape(init_opt_and_model, graph, positions, box)

  # Now that we have the structure, load the data using FLAX checkpointing.
  ckpt_data = (0, abstract_params, abstract_state)

  checkpoints = [c for c in os.listdir(directory) if 'checkpoint' in c]
  assert len(checkpoints) == 1

  checkpoint = os.path.join(directory, checkpoints[0])

  with open(checkpoint, 'rb') as f:
    ckpt = serialization.from_bytes(ckpt_data, f.read())

  params = tree_map(lambda x: x.astype(f32), ckpt[1])
  return c, model, params