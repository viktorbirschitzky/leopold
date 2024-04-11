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
import sys

def setup(lower, upper, batch_size_test):
    with open('data/bulk_300K_processed.dat', 'rb') as f:
        data = pickle.load(f)

    positions, forces, occ, atoms, cell, energies = data

    print(energies.shape)

    # if settting up new data, check if coordinates are fractional!!!!
    positions = positions/np.diag(cell)

    # do batching of data
    batch_size = 1 
    n = len(positions[1])

    step = 1000

    train_box = cell
    train_position = positions[lower:upper:step].reshape((-1, batch_size, n, 3))
    train_energy = energies[lower:upper:step].reshape((-1, batch_size,1))
    train_occ = occ[lower:upper:step].reshape((-1, batch_size,n,2))
    train_force = forces[lower:upper:step].reshape((-1, batch_size, n, 3))
    train_atoms = atoms[lower:upper:step].reshape((-1, batch_size, n, 94))

    train_data = (train_box, train_position, train_energy, train_force, train_atoms, train_occ)

    print(train_energy.shape, np.histogram(train_energy))

    step = 1003

    val_box = cell
    val_position = positions[lower:upper:step].reshape((-1, batch_size, n, 3))
    val_energy = energies[lower:upper:step].reshape((-1, batch_size,1))
    val_occ = occ[lower:upper:step].reshape((-1, batch_size,n,2))
    val_force =  forces[lower:upper:step].reshape((-1, batch_size, n, 3))
    val_atoms = atoms[lower:upper:step].reshape((-1, batch_size, n, 94))

    val_data = (val_box, val_position, val_energy, val_force, val_atoms, val_occ)

    print(val_energy.shape, np.histogram(val_energy))

    test_box = cell
    test_position = positions[lower:upper:].reshape((-1, batch_size_test, n, 3))
    test_energy = energies[lower:upper:].reshape((-1, batch_size_test,1))
    test_occ = occ[lower:upper:].reshape((-1, batch_size_test,n,2))
    test_force =  forces[lower:upper:].reshape((-1, batch_size_test, n, 3))
    test_atoms = atoms[lower:upper:].reshape((-1, batch_size_test, n, 94))

    test_data = (test_box, test_position, test_energy, test_force, test_atoms, test_occ)

    print(test_energy.shape, np.histogram(test_energy))

    with open('data_mgo.pkl', 'wb') as f:
        pickle.dump((train_data, val_data, test_data), f)

    return

if __name__=='__main__':
    lower, upper, batch = [int(i) for i in sys.argv[1:4]]
    setup(lower, upper, batch)
