"""Preprocess our MgO data into train and test sets written as extended xyz files"""

import pickle
import numpy as np

from ase.io import write
from ase import Atoms

from tqdm import tqdm

# ----DATA loading

with open("./data/bulk_300K_2U.dat", "rb") as file:
    # Dictionary with the data of all the run:
    # - pos: [n_frame, n_atoms, 3]
    # - forces: [n_frame, n_atoms, 3]
    # - energies: [n_frame]
    # - mag: [n_frame, n_atoms, 4]
    # - charge: [n_frame, n_atoms, 4]
    # - timing: [n_frame]
    # - up: [n_frame, n_atoms, 3, 3]
    # - down: [n_frame, n_atoms, 3, 3]
    # - cell: [n_frame, 3, 3]
    # - element: [n_frame, n_atoms]
    data = pickle.load(file)


# ----Gain the interesting data
n_frames = len(data["element"])
print(f"Loaded the trajectory with {n_frames} frames!")

# Set correct atomic labels
elements = np.where(data["element"] == "Mg_pv", "Mg", "O")

# Evaluate traces of occupation matrix
tr_up = np.trace(data["up"], axis1=-2, axis2=-1)[..., np.newaxis]
tr_down = np.trace(data["down"], axis1=-2, axis2=-1)[..., np.newaxis]
t_occup = np.append(tr_up, tr_down, axis=2)

# Get which element has the polaron at each step
# the -1 is since we have only one polaron in this dataset
# TODO: Not sure this is clean and necessary (Luca Leoni)
pol_idx = np.argsort(np.abs(data["mag"][..., -1]), axis=1)[:, -1:]

# 1 if the element has the polaron and 0 otherwise
polaron = np.zeros_like(elements[:-1], dtype=np.int64)
_, cols = np.indices(polaron.shape)
polaron[cols == pol_idx] = 1

# ----Create the whole trajectory data
traj: list[Atoms] = []
for i in tqdm(range(n_frames - 1)):
    atoms = Atoms(
        symbols=elements[i],
        positions=data["pos"][i],
        cell=data["cell"][0],
        pbc=(True, True, True),
    )

    # To be sure
    atoms.calc = None

    # Global properties
    atoms.info["energy"] = data["energies"][i]

    # Per atom properties
    atoms.arrays["pol_state"] = polaron[i]
    atoms.arrays["forces"] = data["forces"][i]
    atoms.arrays["magmoms"] = data["mag"][i]
    atoms.arrays["toccup"] = t_occup[i]

    traj.append(atoms)

# 80% split between train and test part
split_idx = int(n_frames * 0.8)
step = 2

write("data/MgO_train.xyz", traj[:split_idx:step], format="extxyz")
write("data/MgO_test.xyz", traj[split_idx:], format="extxyz")

print(f"Created Train set with {int(split_idx / step)} configurations")
print(f"Created Test set with {n_frames - split_idx} configurations")

# Write all the trajectory to see
write("data/MgO_full.xyz", traj, format="extxyz")
