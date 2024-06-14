# loads full trajectory from h5 and gets all indices of timesteps where polaron idx changed
# and vector connecting hopping sites for analysis of hopping types 
import numpy as np
import tables as tb
import pickle
import matplotlib.pyplot as plt

from jax_md import space
from jax import vmap

import sys

start, end = int(sys.argv[1]), int(sys.argv[2])

# load stuff
file = tb.open_file('/data/viktor/polaron_dynamics/leopold/TiO_0_600.h5', 'r')
cell = file.root.cell.read()
print('Reading positions')
position = file.root.frames.read(field='positions')
print('Reading Occs')
toccup = file.root.frames.read(field='toccups')

# define some stuff
displacement, shift = space.periodic_general(cell)
mag = toccup[...,0]-toccup[...,1]

# get polaron idxs at each step
pol_idx = np.argmax(mag, axis=1)

# get timesteps where pol_idx changed; ie hops
hops = (pol_idx[1:] != pol_idx[:-1]).nonzero()[0]

# get displacement vector of polarons in pbc
hop_res = []
for hop in hops:
    hop_res.append(displacement(position[hop,pol_idx[hop]], position[hop+1, pol_idx[hop+1]]))
hop_res = np.array(hop_res)

# store displacements
with open('hops.pkl', 'wb') as f:
    pickle.dump([hops, hop_res],f)

