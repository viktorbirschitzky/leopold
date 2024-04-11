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

from functools import partial

# load data
with open('data_mgo.pkl', 'rb') as f:
    train_data, val_data, test_data = pickle.load(f)

train_box, train_position, train_energy, train_force, train_atoms, train_occ = train_data
val_box, val_position, val_energy, val_force, val_atoms, val_occ = val_data
test_box, test_position, test_energy, test_force, test_atoms, test_occ = test_data

#initialize stuff
with open('params/params_mgo.pkl', 'rb') as f:
  params, config = pickle.load(f)

model =  nequip.model_from_config(config)
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


# MD parameters
# Unit system such that eV, atomic mass units, and angstrom are 1.
kb = 8.61733362e-5        # boltzmann constant in units of eV/K
fs = 0.09822693530999717  # femtoseconds in units of mass, energy, length == 1

dt = 1 * fs
kT = 300 * kb

m_Mg = 24.305
m_O = 15.999

masses = np.array(32*[m_Mg] + 32*[m_O])

# some functions for energy evaluation and MD
def energy(params, pos, **kwargs):
    return energy_fn(params, pos, **kwargs)[0][0,0]

def energy_occ(params, pos, **kwargs):
    return energy_fn(params, pos, **kwargs)

E = partial(energy, params, box=train_box)

E_occ = partial(energy_occ, params, box=train_box)

F = quantity.force(E)
init_fn, step_fn = simulate.nvt_nose_hoover(E, shift, dt=dt, kT=kT)

def get_en_occ(params, box, position, atoms):
    # Updates neighbors lists before computing the energy
    _nbrs = nbrs.update(position, box=box)
    predicted_energy, predicted_occs = energy_fn(params, position, _nbrs, box=box, atoms=atoms)
    return predicted_energy[0,0], predicted_occs

@jit
def update(state, nbrs, atoms):
    state = step_fn(state, neighbor=nbrs, atoms=atoms)
    nbrs = nbrs.update(state.position)
    en, occ  = E_occ(state.position, neighbor=nbrs, atoms=atoms)
    return state, nbrs, en, occ

def update_atoms(atoms, mag):
    new_atoms = atoms.copy()
    new_atoms[:,-10] = 0
    new_atoms[np.argmax(mag),-10] = 1
    return new_atoms

steps = 1000000

states = []
energies = []
occs = []

atom = train_atoms[0,0].copy()

key = random.PRNGKey(0)
state = init_fn(key, 
                jnp.array(train_position[0,0]), 
                mass=masses, 
                neighbor=nbrs, 
                atoms=atom)

for i in range(steps):
    state, nbrs, en, occ = update(state, nbrs, atom)
    energies.append(en), occs.append(occ)
    mag = occ[...,0] - occ[...,1]
    atom = update_atoms(atom, mag)
    if i%100 == 0: 
        pols = np.argsort(mag)[-2:]
        print(i, pols, mag[pols], en[0,0])

res = {'states': states,
       'energies': np.array(energies),
       'occs': np.array(occs)}

with open('results/MD_run.pkl', 'wb') as f:
    pickle.dump(res, f)
