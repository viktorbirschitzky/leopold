# Script for analyzing specific hops
# Plots occupation from MD run for given start and end
# Writes POSCARs from start to end for comparative DFT calculation
import numpy as np
import tables as tb
import pickle
import matplotlib.pyplot as plt
from ase.io import write, read
from ase import Atoms
from ase.units import kB
import os

from jax_md import space
from jax import vmap

import sys

start, end = int(sys.argv[1]), int(sys.argv[2])
data_path = 'jumps/'

# load stuff
file = tb.open_file('/data/viktor/polaron_dynamics/leopold/TiO_0_600.h5', 'r')
cell = file.root.cell.read()
print('Reading positions')
position = file.root.frames.read(start, end, field='positions')
print('Reading Occs')
toccup = file.root.frames.read(start, end, field='toccups')

# define some stuff
displacement, shift = space.periodic_general(cell)
mag = toccup[...,0]-toccup[...,1]
tot = toccup[...,0]+toccup[...,1]

plot = input("Plotting? (y/n): ").strip().lower() == "y"
if plot:
    plt.figure(figsize=(12,8))
    plt.subplot(2,2,1)
    plt.plot(toccup[:,:96,0])
    plt.subplot(2,2,2)
    plt.plot(toccup[:,:96,1])
    plt.subplot(2,2,3)
    plt.plot(tot[:,:96])
    plt.subplot(2,2,4)
    plt.plot(mag[:,:96])
    plt.tight_layout()
    plt.show()


positi = file.root.frames.read(start, end, field='positions')
polaro = file.root.frames.read(start, end, field='polaron')
forces = file.root.frames.read(start, end, field='forces')
toccup = file.root.frames.read(start, end, field='toccups')
energy = file.root.frames.read(start, end, field='energy')

jump = []

species = file.root.species.read()
# Write atoms
for (pos, pol, fos, toc, ene) in zip(positi, polaro, forces, toccup, energy):
    atom = Atoms(numbers=species, scaled_positions=pos, cell=cell, pbc=(True, True, True))
    atom.calc = None

    atom.info['energy'] = ene
    atom.arrays['forces'] = fos
    atom.arrays['pol_state'] = pol.astype(np.int32)
    atom.arrays['toccup'] = toc
    atom.arrays['magmom'] = np.diff(toc, axis=-1).flatten()

    jump.append(atom)

write_files = input("Writing? (y/n): ").strip().lower() == "y"
if write_files:
    jump_path = os.path.join(data_path, f'jump_{start}_{end}')

    # Write entire database for simplicity
    os.makedirs(jump_path, exist_ok=True)
    #open(os.path.join(jump_path, 'database.xyz'), 'w').close()

    # Write the single POSCARS
    write(os.path.join(jump_path, 'jump.xyz'), jump, format='extxyz')
    for j, atom in enumerate(jump):
        # Create folder
        pos_path = os.path.join(jump_path, f'poscar_{j}')
        os.makedirs(pos_path, exist_ok=True)

        write(os.path.join(pos_path, 'POSCAR'), atom, format='vasp')
    print('Done writintg')

