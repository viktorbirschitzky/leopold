"""Utils function to use the leopold model"""

# ---- IMPORTS

# Jax
import jax.numpy as jnp
import jax.random as jrn

from jax import value_and_grad

# Jax MD
from jax_md import nn, space, partition

# ASE
from ase.io import read

# Types
from ase import Atoms
from jax import Array
from ml_collections import ConfigDict
from typing import Tuple, List, Callable, NamedTuple
from flax.typing import VariableDict

# ---- DATABASE FUNCTIONS


class AtomsData(NamedTuple):
    energies: Array  # [..., ]
    cell: Array  # [..., 3,3]
    positions: Array  # [..., 3]
    forces: Array  # [..., 3]
    species: Array  # [..., n_species+1]
    toccup: Array  # [..., 2]
    atom_num: Array  # [n_species]


Dataset = List[AtomsData]


def get_all(batch: Dataset, key: str) -> Array:
    val = []
    for data in batch:
        val.extend(data._asdict()[key])

    return jnp.array(val)


def batch_data(data: AtomsData, batch_size: int) -> Dataset:
    data_dict = data._asdict()
    n_batch = 1 + (len(data.species) // batch_size)

    batched_data = [dict() for _ in range(n_batch)]
    for key in data_dict.keys():
        vals = jnp.split(
            data_dict[key],
            [(i + 1) * batch_size for i in range(n_batch - 1)],
            axis=0,
        )

        for batch, val in zip(batched_data, vals):
            batch[key] = val

    return [AtomsData(**batch) for batch in batched_data]


def split_data(data: AtomsData, rateo: float) -> Tuple[AtomsData, AtomsData]:
    data_dict = data._asdict()
    first_dict, second_dict = dict(), dict()

    split_idx = int(len(data.species) * rateo)
    for key in data_dict.keys():
        first_dict[key], second_dict[key] = jnp.split(
            data_dict[key], [split_idx], axis=0
        )

    return (AtomsData(**first_dict), AtomsData(**second_dict))


def shuffle_data(rng: Array, data: AtomsData) -> AtomsData:
    permutation = jnp.arange(len(data.species))
    permutation = jrn.permutation(rng, permutation)

    dict_data = {key: val[permutation] for key, val in data._asdict().items()}

    return AtomsData(**dict_data)


def get_data_from_xyz(file: str) -> AtomsData:
    atoms = read(file, index=":", format="extxyz")
    if not isinstance(atoms, list):
        atoms = [atoms]

    # Informations about the chemical species
    elements = jnp.unique_values(
        jnp.array([z for atom in atoms for z in atom.get_atomic_numbers()])
    )

    # Collect all the other informations
    energies, positions, forces, toccup, cell, species = [], [], [], [], [], []
    for atom in atoms:
        energies.append(atom.get_potential_energy())
        positions.append(atom.get_scaled_positions())
        forces.append(atom.get_forces())
        cell.append(atom.get_cell().array)

        toccup.append(atom.arrays["toccup"])

        atom_num = jnp.array([atom.get_atomic_numbers()])
        _species = jnp.where(atom_num.T == elements, 1, 0)
        species.append(
            jnp.append(_species, jnp.array([atom.arrays["pol_state"]]).T, axis=1)
        )

    return AtomsData(
        energies=jnp.array(energies),
        cell=jnp.array(cell),
        positions=jnp.array(positions),
        forces=jnp.array(forces),
        toccup=jnp.array(toccup),
        species=jnp.array(species),
        atom_num=jnp.repeat(jnp.array([elements]), len(atoms), axis=0),
    )


def get_atoms_from_data(data: AtomsData) -> List[Atoms]:
    atoms = []

    for i in range(len(data.energies)):
        symbols, pol_state = data.species[i, :, :-1], data.species[i, :, -1]
        atom = Atoms(
            numbers=jnp.matmul(symbols, data.atom_num[0]),
            scaled_positions=data.positions[0],
            cell=data.cell[i],
            pbc=(True, True, True),
        )

        atom.calc = None

        atom.info["energy"] = float(data.energies[i])
        atom.arrays["forces"] = data.forces[i]
        atom.arrays["pol_state"] = pol_state
        atom.arrays["toccup"] = data.toccup[i]

        atoms.append(atom)

    return atoms


# ---- MODEL FUNCTION


def get_model(example_batch: AtomsData, cfg: ConfigDict, **nl_kwargs):
    displacement, _ = space.periodic_general(example_batch.cell[0])
    model = nn.nequip_pol.model_from_config(cfg)

    neighbor = partition.neighbor_list(
        displacement,
        example_batch.cell[0],
        cfg.r_max,  # pyright: ignore
        format=partition.Sparse,
        **nl_kwargs,
    ).allocate(example_batch.positions[0])

    featurizer = nn.util.neighbor_list_featurizer(displacement)

    def init_fn(key: Array, position: Array, cell: Array, atoms: Array):
        graph = featurizer(atoms, position, neighbor.update(position, box=cell))
        return model.init(key, graph)

    def apply_fn(params: VariableDict, position: Array, cell: Array, atoms: Array):
        graph = featurizer(atoms, position, neighbor.update(position, box=cell))
        energy, magmom = model.apply(params, graph)

        return energy[0, 0], magmom[:-1]  # pyright: ignore

    return init_fn, value_and_grad(apply_fn, argnums=1, has_aux=True)


def evaluate_model(params, model: Callable, data: Dataset):
    mod_e, mod_f, mod_o = [], [], []

    for batch in data:
        (energy, toccup), forces = model(
            params,
            batch.positions,
            batch.cell,
            batch.species,
        )

        mod_e.extend(energy)
        mod_f.extend(-forces)
        mod_o.extend(toccup)
    mod_e, mod_f, mod_o = jnp.array(mod_e), jnp.array(mod_f), jnp.array(mod_o)

    data_e = get_all(data, "energies")
    data_f = get_all(data, "forces")
    data_o = get_all(data, "toccup")

    return (
        jnp.sqrt(jnp.square(mod_e - data_e).mean()),
        jnp.sqrt(jnp.square(mod_f - data_f).mean()),
        jnp.sqrt(jnp.square(mod_o - data_o).mean()),
    )
