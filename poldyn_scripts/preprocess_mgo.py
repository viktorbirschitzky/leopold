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

def setup():
    with open('data/bulk_300K_2U.dat', 'rb') as f:
        res = pickle.load(f)

    def polaron_embedding(atoms, mags, n_pols=1):
        pol = np.argsort(mags, axis=1)[:,-n_pols:]
        embedding = np.zeros((atoms.shape[0], atoms.shape[1],94))
        embedding[atoms=='Mg_pv', 11] = 1
        embedding[atoms=='O', 7] = 1

        for idx, pol_idx in enumerate(pol):
            embedding[idx, pol_idx, -10] = 1

        return embedding

    atoms = polaron_embedding(res['element'], np.abs(res['mag'][...,-1]))

    tr_up = np.trace(res['up'], axis1=-2, axis2=-1)[...,np.newaxis]
    tr_down = np.trace(res['down'], axis1=-2, axis2=-1)[...,np.newaxis]
    traces = np.concatenate((tr_up,tr_down), axis=-1)
    data = [res['pos'], res['forces'], traces[:-1], atoms, res['cell'][0], res['energies']]
    for key in data:
        print(key.shape)

    with open('data/bulk_300K_processed.dat', 'wb') as f:
        pickle.dump(data, f)
    return

if __name__=='__main__':
    setup()
