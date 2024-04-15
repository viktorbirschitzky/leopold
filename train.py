"""Train script for the leopold architecture"""

# ---- IMPORTS

# Generics
import yaml
import logging
import sys
import os

# Jax
import jax.numpy as jnp
import jax.random as jrn

# Jax MD
from jax_md import nn

# ASE
from ase.io import read

# Types
from jax import Array
from dataclasses import dataclass, asdict
from ml_collections import ConfigDict
from argparse import ArgumentParser, Namespace
from typing import Union, Optional, Tuple, List


# ---- HELPER FUNCTIONS
def arg_parse() -> Namespace:
    parser = ArgumentParser()

    # Main argument
    parser.add_argument("database", nargs="+")

    # Directory informations
    parser.add_argument("--name", type=str, default="POL_DYN")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--chekpoints_dir", type=str, default="checkpoints")
    parser.add_argument("--logs_dir", type=str, default="logs")

    # Model specifics
    parser.add_argument("--configuration", type=str, default=None)
    parser.add_argument(
        "--default_dtype", type=str, choices=["float64", "float32"], default="float64"
    )

    # General options
    parser.add_argument("--log_level", help="log level", type=str, default="INFO")

    return parser.parse_args()


def default_config() -> ConfigDict:
    config = ConfigDict()

    config.graph_net_steps = 5
    config.nonlinearities = {"e": "raw_swish", "o": "tanh"}
    config.use_sc = True
    config.n_elements = 94
    config.hidden_irreps = "128x0e + 64x1e + 4x2e"
    config.sh_irreps = "1x0e + 1x1e + 1x2e"
    config.num_basis = 8
    config.r_max = 5.0
    config.radial_net_nonlinearity = "raw_swish"
    config.radial_net_n_hidden = 64
    config.radial_net_n_layers = 2

    # average number of neighbors per atom, used to divide activations are sum
    # in the nequip convolution, helpful for internal normalization.
    config.n_neighbors = 10.0

    # Standard deviation used for the initializer of the weight matrix in the
    # radial scalar MLP
    config.scalar_mlp_std = 4.0

    return config


def setup_logger(
    level: Union[int, str] = logging.INFO,
    tag: Optional[str] = None,
    directory: Optional[str] = None,
):
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Create general formatting
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create standard output Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Create file handler
    if (directory is not None) and (tag is not None):
        os.makedirs(directory, exist_ok=True)

        fh = logging.FileHandler(os.path.join(directory, tag + ".log"))
        fh.setFormatter(formatter)

        logger.addHandler(fh)


@dataclass
class AtomsData:
    cell: Array  # [..., 3,3]
    positions: Array  # [..., 3]
    forces: Array  # [..., 3]
    species: Array  # [..., n_species+1]
    toccup: Array  # [..., 2]


def batch_data(data: AtomsData, batch: int) -> List[AtomsData]:
    return [data]


def split_data(data: AtomsData, rateo: float) -> Tuple[AtomsData, AtomsData]:
    first_dict, second_dict = asdict(data), asdict(data)

    split_idx = int(len(data.species) * rateo)
    for key in first_dict.keys():
        first, second = jnp.split(first_dict[key], [split_idx], axis=0)

        first_dict[key], second_dict[key] = first, second

    return (AtomsData(**first_dict), AtomsData(**second_dict))


def shuffle_data(rng: Array, data: AtomsData) -> AtomsData:
    permutation = jnp.arange(len(data.species))
    permutation = jrn.permutation(rng, permutation)

    return AtomsData(
        cell=data.cell[permutation],
        positions=data.positions[permutation],
        forces=data.forces[permutation],
        species=data.species[permutation],
        toccup=data.toccup[permutation],
    )


def get_data_from_xyz(file: str) -> AtomsData:
    atoms = read(file, index=":", format="extxyz")
    if not isinstance(atoms, list):
        atoms = [atoms]

    # Informations about the chemical species
    elements = set(atoms[0].get_atomic_numbers())
    atom_num = jnp.array(atoms[0].get_atomic_numbers())

    species = []
    for z in elements:
        species.append(jnp.where(atom_num == z, 1, 0).reshape(1, len(atom_num), 1))
    species = jnp.repeat(jnp.concat(species, axis=2), len(atoms), axis=0)

    # If polaronic states are present add also those
    if atoms[0].arrays.get("pol_state", None) is not None:
        pol_state = []
        for atom in atoms:
            pol_state.append(
                jnp.array(atom.arrays["pol_state"]).reshape(len(atom_num), 1)
            )
        pol_state = jnp.array(pol_state)

        species = jnp.concat((species, pol_state), axis=2)

    # Collect all the other informations
    positions, forces, toccup, pol_state, cell = [], [], [], [], []
    for atom in atoms:
        positions.append(atom.get_positions())
        forces.append(atom.get_forces())
        cell.append(atom.get_cell().array)

        toccup.append(atom.arrays["toccup"])
        pol_state.append(jnp.array(atom.arrays["pol_state"]).reshape(len(atom), 1))

    return AtomsData(
        cell=jnp.array(cell),
        positions=jnp.array(positions),
        forces=jnp.array(forces),
        toccup=jnp.array(toccup),
        species=species,
    )


# ---- REAL APPLICATION
def main():
    args = arg_parse()

    # ---- General setting

    # Getting model tag
    tag = f"{args.name}-{args.seed}"

    # Default dtype
    if args.default_dtype == "float64":
        import jax

        jax.config.update("jax_enable_x64", True)

    # Random number generator
    rngKey = jrn.key(args.seed)

    # Getting model configuration and saving
    config = default_config()
    if args.configuration is not None:
        with open(args.configuration, "r") as f:
            config.update(**yaml.safe_load(f))

    # Logging
    setup_logger(args.log_level, tag, args.logs_dir)

    # ---- Load Database
    data = get_data_from_xyz(args.database[0])
    data = shuffle_data(rngKey, data)
    train, eval = split_data(data, 0.8)


if __name__ == "__main__":
    main()
