"""Script to validate an md run done with Leopold using a committee of models"""

# ---- IMPORTS

# Generics
import os
import pickle

# Math
from math import floor

# Jax
from jax import jit, vmap

# Tables
import tables as tb

# Utils
from utils import batch_data, get_data_from_hdf5, get_model

# Types
from argparse import ArgumentParser, Namespace
from tqdm import tqdm


# ---- HELPER FUNCTIONS
def arg_parse() -> Namespace:
    parser = ArgumentParser()

    # Main argument
    parser.add_argument(
        "data_path",
        help="Path to the .h5 file containing the configuration to evaluate",
        type=str,
    )
    parser.add_argument(
        "models_path",
        help="Path to the .pkl files containing the checkpoint of the models you want to use",
        type=str,
        nargs="+",
    )

    # Options
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--chunk_size", type=int, default=100_000)
    parser.add_argument("--compression_level", type=int, default=5)
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

    # ---- GENERAL SETTINGS

    # Construct the name tag
    tag = os.path.basename(args.data_path).split("-")[0]

    # Default dtype
    if args.default_dtype == "float64":
        import jax

        jax.config.update("jax_enable_x64", True)

    # Define output name
    output = args.output
    if output is None:
        output = os.path.dirname(args.data_path)
        output = os.path.join(output, tag + "-committee.h5")

    print(f"Results will be printed at {output}")
    # ---- OPEN DATA FILES

    # Open the input file
    if not tb.is_hdf5_file(args.data_path):
        raise ValueError("Data given are not a Leopold Trajectory")

    input = tb.open_file(args.data_path)

    # Open the output file
    output = tb.open_file(
        output,
        mode="w",
        title="MD trajectory committee",
        filters=tb.Filters(complevel=args.compression_level, complib="zlib"),
    )

    # ---- DEFINE THE DATASTRUSCTURE AND MODELS

    # Collect number of atoms and a sample batch
    nAtoms = len(input.root.species.read())
    data = get_data_from_hdf5(input, beg=0, end=args.chunk_size)
    data = batch_data(data, args.batch_size)

    # Creating the data structure
    class Frame(tb.IsDescription):
        energy = tb.Float32Col(pos=0)
        forces = tb.Float32Col(shape=(nAtoms, 3), pos=1)
        toccup = tb.Float32Col(shape=(nAtoms, 2), pos=2)

    # Create a table for every model
    print("Reading models: ", end="")

    models, tables = [], []
    for model in args.models_path:
        """
        In our experiments the models were uniquelly identified by their seed,
        therefore we use them to define the tables in the committee file
        """
        # Read checkpoint
        with open(model, "rb") as f:
            config, param, _ = pickle.load(f)

        # Collect model seed
        key = os.path.basename(model).split("-")[-1].split(".")[0]
        tables.append(output.create_table("/", key, Frame, title=f"Model {key}"))

        # Save model
        _, apply_fn = get_model(
            data[0],
            config,
            fractional_coordinates=True,
            disable_cell_list=True,
        )

        models.append(
            jit(vmap(lambda pos, cell, atom: apply_fn(param, pos, cell, atom)))
        )

        print(key, end="\n" if model == args.models_path[-1] else ", ")

    # ---- EVALUATE THE MODELS

    # Get the numbers of chunks
    nChunk = floor(input.root.frames.nrows / args.chunk_size)

    for i in range(nChunk + 1):
        print(f"\nEvaluate chunk {i:>3d}:")
        for i, (model, table) in enumerate(zip(models, tables)):
            frame = table.row
            for batch in tqdm(data, desc=f"Model {i}"):
                (energies, toccup), forces = model(
                    batch.positions,
                    batch.cell,
                    batch.species,
                )

                for e, o, f in zip(energies, toccup, forces):
                    frame["energy"] = e
                    frame["toccup"] = o
                    frame["forces"] = -f

                    frame.append()

            table.flush()

        # Read next chunk
        if i != nChunk:
            data = get_data_from_hdf5(
                input, beg=(i + 1) * args.chunk_size, end=(i + 2) * args.chunk_size
            )
            data = batch_data(data, args.batch_size)

    # ---- CLOSE
    input.close()
    output.close()


if __name__ == "__main__":
    main()
