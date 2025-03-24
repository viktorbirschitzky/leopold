"""Evaluation script for the leopold architecture"""

# ---- IMPORTS

# Generics
import pickle
import os

# Jax
import jax.numpy as jnp
from jax import jit, vmap

# ASE
from ase.io import write

# Utils
from leopold.utils import get_atoms_from_data, get_data_from_xyz, batch_data
from leopold.utils import get_model

# Types
from argparse import ArgumentParser, Namespace
from leopold.utils import AtomsData


# ---- HELPER FUNCTIONS
def arg_parse() -> Namespace:
    parser = ArgumentParser()

    # Main argument
    parser.add_argument(
        "data_path",
        help="Path to the .xyz file containing the configuration to evaluate",
        type=str,
    )
    parser.add_argument(
        "model_path",
        help="Path to the .pkl file containing the checkpoint of the model you want to use",
        type=str,
    )

    # Options
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument(
        "--output",
        help="Path to the output file where to save the results",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--default_dtype", type=str, choices=["float64", "float32"], default="float64"
    )

    return parser.parse_args()


# ---- REAL APPLICATION
def main():
    args = arg_parse()

    # read data
    data = get_data_from_xyz(args.data_path)
    data = batch_data(data, args.batch_size)

    # Read given checkpoint
    with open(args.model_path, "rb") as f:
        config, params, _ = pickle.load(f)

    _, apply_fn = get_model(
        data[0],
        config,
        fractional_coordinates=True,
        disable_cell_list=True,
    )

    compiled_vect_model = jit(vmap(apply_fn, (None, 0, 0, 0)))

    print("Model created")

    # Real evaluation
    result = dict()
    for batch in data:
        (energies, toccup), forces = compiled_vect_model(
            params,
            batch.positions,
            batch.cell,
            batch.species,
        )

        for key in ["positions", "cell", "species", "atom_num"]:
            if result.get(key, None) is None:
                result[key] = batch._asdict()[key]
            else:
                result[key] = jnp.append(result[key], batch._asdict()[key], axis=0)

        if result.get("energies", None) is None:
            result["energies"] = energies
        else:
            result["energies"] = jnp.append(result["energies"], energies, axis=0)

        if result.get("toccup", None) is None:
            result["toccup"] = toccup
        else:
            result["toccup"] = jnp.append(result["toccup"], toccup, axis=0)

        if result.get("forces", None) is None:
            result["forces"] = -forces
        else:
            result["forces"] = jnp.append(result["forces"], -forces, axis=0)

    result = AtomsData(**result)

    print("Model evaluated")

    out_path = args.output
    if out_path is None:
        out_path = os.path.basename(args.model_path).split(".pkl")[0] + ".xyz"

    write(out_path, get_atoms_from_data(result), format="extxyz")


if __name__ == "__main__":
    main()
