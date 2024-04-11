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
from jax_md.nn import nequip_pol

import pickle
import optax

#from utils import *

def make_config(energy, force, occ, atoms, distinguish_charge_state=False):
    # Design a Nequip architecture
    config = nequip_pol.default_config()
    # Spherical harmonic representations
    config.sh_irreps = '1x0e + 1x1e'
    # Hidden representation representations
    config.hidden_irreps = '32x0e + 4x1e'
    config.radial_net_n_hidden = 16
    # Total number of graph net iterations
    config.graph_net_steps = 3

    # Dataset shift (average) and scale (std deviation)
    config.shift = (energy/64).mean()
    config.scale = (force.flatten()).std()

    # this block only distinguishes elements but not their charge state
    nonzero_indices = np.where(atoms.reshape(-1,94)[:,:83] == 1)
    indices = np.maximum.reduceat(nonzero_indices[1], np.unique(nonzero_indices[0], return_index=True)[1])

    occ_flat = occ.reshape(-1,2)
    tmp_shift = np.zeros(94)
    tmp_scale = np.zeros(94)

    for index in np.unique(indices):
        tmp_shift[index] = np.mean(occ_flat[indices==index])
        tmp_scale[index] = np.std(occ_flat[indices==index])

    indices = np.argmax(atoms[0,0], axis=-1)
    config.shift_occ = tmp_shift[indices]
    config.scale_occ = tmp_scale[indices]

    config.n_neighbors = 18.
    config.scalar_mlp_std = 4.
    return config

# load data

with open('data_mgo.pkl', 'rb') as f:
    train_data, val_data, test_data = pickle.load(f)

train_box, train_position, train_energy, train_force, train_atoms, train_occ = train_data
val_box, val_position, val_energy, val_force, val_atoms, val_occ = val_data
test_box, test_position, test_energy, test_force, test_atoms, test_occ = test_data


#initialize stuff
config = make_config(train_energy, train_force, train_occ, train_atoms)

model =  nequip_pol.model_from_config(config)
displacement, shift = space.periodic_general(train_box)
featurizer = jax_md._nn.util.neighbor_list_featurizer(displacement)

neighbor_list, init_fn, energy_fn = energy.nequip_neighbor_list(
    displacement, 
    train_box, 
    config, 
    fractional_coordinates=True, 
    disable_cell_list=True
)

nbrs = neighbor_list.allocate(jnp.array(test_position)[0][0])


# loss and stuff 

def en_occ_n_nbrs(params, box, position, atoms):
    # Updates neighbors lists before computing the energy
    _nbrs = nbrs.update(position, box=box)
    predicted_energy, predicted_occs = energy_fn(params, position, _nbrs, box=box, atoms=atoms)
    return predicted_energy[0,0], predicted_occs

def en_n_nbrs(params, box, position, atoms):
    # Updates neighbors lists before computing the energy
    _nbrs = nbrs.update(position, box=box)
    predicted_energy, _ = energy_fn(params, position, _nbrs, box=box, atoms=atoms)
    return predicted_energy[0,0]

def occ_n_nbrs(params, box, position, atoms):
    # Updates neighbors lists before computing the energy
    _nbrs = nbrs.update(position, box=box)
    _, predicted_occs = energy_fn(params, position, _nbrs, box=box, atoms=atoms)
    return predicted_occs


def occ_loss_fn(params, box, position, occs, atoms):
    # Energies are batched over multiple particle positions
    E = vmap(occ_n_nbrs, (None, None, 0))
    predicted_occs = E(params, box, position, atoms=atoms)
    return jnp.mean((predicted_occs[...,0] - occs[...,0]) ** 2 +
                    (predicted_occs[...,1] - occs[...,1]) ** 2 +
                    ((predicted_occs[...,0]+predicted_occs[...,1]) - (occs[...,0] + occs[...,1])) ** 2 +
                    ((predicted_occs[...,0]-predicted_occs[...,1]) - (occs[...,0] - occs[...,1])) ** 2)

def energy_loss_fn(params, box, position, energy, atoms):
  # Energies are batched over multiple particle positions
  E = vmap(en_n_nbrs, (None, None, 0))
  predicted_energy = E(params, box, position, atoms=atoms)
  return jnp.mean((predicted_energy - energy) ** 2)

def force_loss_fn(params, box, position, force, atoms):
  # We want the gradient with respect to the position, not the parameters or box.
  F = vmap(grad(en_n_nbrs, argnums=2), (None, None, 0, 0))
  return jnp.mean((F(params, box, position, atoms) + force) ** 2)

@jit
def loss_fn(params, box, position, energy, force, occs, atoms):
  # A common optimization is to weight these loss functions to energies
  # and forcess are learned appropriately
  return (energy_loss_fn(params, box, position, energy, atoms) + 
          64**2*force_loss_fn(params, box, position, force, atoms)+
          64**2*occ_loss_fn(params, box, position, occs, atoms))


# either random init
key = random.PRNGKey(0)
params = init_fn(key, jnp.array(test_position)[0,0], neighbor=nbrs, atoms = test_atoms[0][0])

#or start from previous
#with open('params_tmp.pkl', 'rb') as f:
#  params, config = pickle.load(f)

# training

opt = optax.adam(5e-4)
opt_state = opt.init(params)

# training
@jit
def update(params, opt_state, box, position, energy, force, occ, atoms):
  grad_fn = grad(loss_fn)
  # Update the step based on gradient information
  updates, opt_state = opt.update(
      grad_fn(params, box, position, energy, force, occ, atoms), opt_state)
  return optax.apply_updates(params, updates), opt_state
lowest_loss = np.inf

# Iterate through training data (epochs)
for i in range(5000):
  if i % 1 == 0:
    train_loss = loss_fn(params, train_box, train_position[0:50,0], train_energy[0:50,0], train_force[0:50,0], train_occ[0:50,0], train_atoms[0:50,0], )
    val_loss = loss_fn(params, val_box, val_position[:,0], val_energy[:,0], val_force[:,0], val_occ[:,0], val_atoms[:,0])
    
    
    print(f'Epoch {i}; Train Loss: {train_loss}. '
          f'Test Loss: {val_loss}')
    
    
    if val_loss < lowest_loss:
      lowest_loss = val_loss
      with open('params_tmp.pkl', 'wb') as f:
        pickle.dump([params, config], f)

  
  # Iterate over minibatches of data
  for p, e, f, o, a in zip(train_position, train_energy, train_force, train_occ, train_atoms):
    params, opt_state = update(params, opt_state, train_box, p, e, f, o, a)
