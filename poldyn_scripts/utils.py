import jax.numpy as np
from jax import device_put
from jax import tree_map
from jax.config import config

import warnings; warnings.simplefilter('ignore')
config.update('jax_enable_x64', True)

import pickle

import warnings
warnings.simplefilter("ignore")

import jax_md
import jax.numpy as jnp
import jax

from jax import vmap
from jax import jit
from jax import random
from jax import grad
import jax.numpy as jnp
from jax import tree_util

from flax import serialization
import e3nn_jax as e3nn
import numpy as np

from jax_md import space
from jax_md import energy
from jax_md import simulate
from jax_md import quantity
from jax_md.nn import util as nn_util
from jax_md.nn import nequip

import pickle


def make_config():
    # Design a Nequip architecture
    config = nequip.default_config()
    # Spherical harmonic representations
    config.sh_irreps = '1x0e + 1x1e'
    # Hidden representation representations
    config.hidden_irreps = '32x0e + 4x1e'
    config.radial_net_n_hidden = 16
    # Total number of graph net iterations
    config.graph_net_steps = 3
    
    # Dataset shift (average) and scale (std deviation)
    config.shift = np.zeros(94)
    config.scale = np.zeros(94)
    
    config.n_neighbors = 10.
    config.scalar_mlp_std = 4.
    return config


def init_params():
    key = random.PRNGKey(0)
    params = init_fn(key, jnp.array(test_position)[0,0], neighbor=nbrs, atoms = test_atoms[0][0])
    return params


def energy_and_nbrs(params, box, position, atoms):
  # Updates neighbors lists before computing the energy
  _nbrs = nbrs.update(position, box=box)
  return energy_fn(params, position, _nbrs, box=box, atoms=atoms)

def energy_loss_fn(params, box, position, energy, atoms):
  # Energies are batched over multiple particle positions
  E = vmap(energy_and_nbrs, (None, None, 0))
  predicted_energy = E(params, box, position, atoms=atoms)
  return jnp.mean((predicted_energy - energy) ** 2)

def force_loss_fn(params, box, position, force, atoms):
  # We want the gradient with respect to the position, not the parameters or box.
  F = vmap(grad(energy_and_nbrs, argnums=2), (None, None, 0, 0))
  return jnp.mean((F(params, box, position, atoms) + force) ** 2)

@jit
def loss_fn(params, box, position, energy, force, atoms):
  # A common optimization is to weight these loss functions to energies
  # and forcess are learned appropriately
  return (energy_loss_fn(params, box, position, energy, atoms) + 
          82944*force_loss_fn(params, box, position, force, atoms))
