"""Train script for the leopold architecture"""

# ---- IMPORTS

# Generics
import yaml
import logging
import sys
import os
import pickle

# Jax
import jax.numpy as jnp
import jax.random as jrn

from jax import vmap, jit, value_and_grad, tree_util
from jax.lax import fori_loop

# Jax MD
from jax_md import nn, space, partition

# ASE
from ase.io import read

# Optax
import optax

# Types
from jax import Array
from ml_collections import ConfigDict
from argparse import ArgumentParser, Namespace
from typing import Union, Optional, Tuple, List, Callable, NamedTuple


# ---- HELPER FUNCTIONS
def arg_parse() -> Namespace:
    parser = ArgumentParser()

    # Main argument
    parser.add_argument("database", nargs="+")

    # Database Menagement
    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=20)

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

    # Train specifics
    parser.add_argument("--energy_weight", type=float, default=1.0)
    parser.add_argument("--forces_weight", type=float, default=64**2)
    parser.add_argument("--toccup_weight", type=float, default=64**2)
    parser.add_argument("--max_epoch", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    # General options
    parser.add_argument("--log_level", help="log level", type=str, default="INFO")

    return parser.parse_args()


def default_config() -> ConfigDict:
    config = ConfigDict()

    config.graph_net_steps = 3
    config.nonlinearities = {"e": "raw_swish", "o": "tanh"}
    config.use_sc = True
    config.n_elements = 3
    config.hidden_irreps = "32x0e + 4x1e"
    config.sh_irreps = "1x0e + 1x1e"
    config.num_basis = 8
    config.r_max = 5.0
    config.radial_net_nonlinearity = "raw_swish"
    config.radial_net_n_hidden = 16
    config.radial_net_n_layers = 2

    # average number of neighbors per atom, used to divide activations are sum
    # in the nequip convolution, helpful for internal normalization.
    config.n_neighbors = 18.0

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


# ---- DATABASE FUNCTIONS


class AtomsData(NamedTuple):
    energies: Array  # [..., ]
    cell: Array  # [..., 3,3]
    positions: Array  # [..., 3]
    forces: Array  # [..., 3]
    species: Array  # [..., n_species+1]
    toccup: Array  # [..., 2]


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
    elements = set(atoms[0].get_atomic_numbers())
    atom_num = jnp.array(atoms[0].get_atomic_numbers())

    species = []
    for z in elements:
        species.append(jnp.where(atom_num == z, 1, 0).reshape(1, len(atom_num), 1))
    species = jnp.repeat(jnp.concat(species, axis=2), len(atoms), axis=0)

    # Collect all the other informations
    energies, positions, forces, toccup, pol_state, cell = [], [], [], [], [], []
    for atom in atoms:
        energies.append(atom.get_potential_energy() / len(atom))
        positions.append(atom.get_scaled_positions())
        forces.append(atom.get_forces())
        cell.append(atom.get_cell().array)

        toccup.append(atom.arrays["toccup"])
        pol_state.append(jnp.array(atom.arrays["pol_state"]).reshape(len(atom), 1))

    return AtomsData(
        energies=jnp.array(energies),
        cell=jnp.array(cell),
        positions=jnp.array(positions),
        forces=jnp.array(forces),
        toccup=jnp.array(toccup),
        species=jnp.concat((species, jnp.array(pol_state)), axis=2),
    )


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

    def apply_fn(params, position, cell, atoms):
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


# ---- REAL APPLICATION
def main():
    args = arg_parse()

    # ---- GENERAL SETTING

    # Getting model tag
    tag = f"{args.name}-{args.seed}"

    # Logging
    setup_logger(args.log_level, tag, args.logs_dir)

    # Default dtype
    if args.default_dtype == "float64":
        import jax

        jax.config.update("jax_enable_x64", True)
        logging.info("Using float64 for training")
    else:
        logging.info("Using float32 for training")

    # Random number generator
    rngKey = jrn.key(args.seed)

    # Getting model configuration and saving
    config = default_config()
    if args.configuration is not None:
        with open(args.configuration, "r") as f:
            config.update(**yaml.safe_load(f))
    logging.info(f"Read configuration from {args.configuration}")

    # ---- LOAD DATABASE

    # Only training set is given
    if len(args.database) < 2:
        # Read data from first file
        data = get_data_from_xyz(args.database[0])
        logging.info(f"Read {len(data.species)} configurations from {args.database[0]}")

        # Splitting train and validation at random
        logging.info(f"Using {args.val_split * 100}% for validation")
        rngKey, key = jrn.split(rngKey)
        data = shuffle_data(key, data)
        train, eval = split_data(data, 1 - args.val_split)

        logging.info(f"Created train set with {len(train.species)} configurations")
        logging.info(f"Created validation set with {len(eval.species)} configurations")

        # Batching the data
        train, eval = (
            batch_data(train, args.batch_size),
            batch_data(eval, args.batch_size),
        )

        # No test is given
        logging.info("No data for testing were given")
        test = None
    # Training set and Test set are given
    elif len(args.database) < 3:
        # Read data from first file
        data = get_data_from_xyz(args.database[0])
        logging.info(f"Read {len(data.species)} configurations from {args.database[0]}")

        # Splitting train and validation at random
        logging.info(f"Using {args.val_split * 100}% for validation")
        rngKey, key = jrn.split(rngKey)
        data = shuffle_data(key, data)
        train, eval = split_data(data, 1 - args.val_split)

        logging.info(f"Created train set with {len(train.species)} configurations")
        logging.info(f"Created validation set with {len(eval.species)} configurations")

        # Read second file as test data
        test = get_data_from_xyz(args.database[1])
        logging.info(
            f"Loaded {len(test.species)} configurations from {args.database[1]} used for testing"
        )

        # Batching the data
        train = batch_data(train, args.batch_size)
        eval = batch_data(eval, args.batch_size)
        test = batch_data(test, args.batch_size)
    # Training set, Validation set and Test set are given
    else:
        # Read every file as a different set
        train = get_data_from_xyz(args.database[0])
        eval = get_data_from_xyz(args.database[1])
        test = get_data_from_xyz(args.database[2])

        logging.info(
            f"Loaded {len(train.species)} configurations from {args.database[0]} used for training"
        )
        logging.info(
            f"Loaded {len(eval.species)} configurations from {args.database[1]} used for validation"
        )
        logging.info(
            f"Loaded {len(test.species)} configurations from {args.database[2]} used for testing"
        )

        # Batching the data
        train = batch_data(train, args.batch_size)
        eval = batch_data(eval, args.batch_size)
        test = batch_data(test, args.batch_size)

    # ---- MODEL CONSTRUCTION

    # Update the number of atomic species
    config.n_elements = train[0].species.shape[-1]

    # Computing shift and scale for energies
    config.shift = get_all(train, "energies").mean()
    config.scale = get_all(train, "forces").flatten().std()

    logging.info(
        f"Using an energy scaling and shift of {config.scale:.2f} and {config.shift:.2f}"
    )

    # Compute species dependent shift and scale for toccup
    toccup = get_all(train, "toccup")
    species = get_all(train, "species")

    toccup_shift, toccup_scale = [], []
    for z in species.T[:-1]:
        toccup_shift.append(toccup[z.T == 1].mean(0))
        toccup_scale.append(toccup[z.T == 1].std(0))
    config.shift_occ = jnp.array(toccup_shift)
    config.scale_occ = jnp.array(toccup_scale)

    logging.info("Using species-dependent occupation matrix scaling and shift:")
    for i, (scale, shift) in enumerate(zip(toccup_scale, toccup_shift)):
        logging.info(f"Species {i}: scale = {scale}   shift = {shift}")

    # Model function creation
    init_fn, apply_fn = get_model(
        train[0],
        config,
        fractional_coordinates=True,
        disable_cell_list=True,
    )

    compiled_vect_model = jit(vmap(apply_fn, (None, 0, 0, 0)))

    rngKey, key = jrn.split(rngKey)
    params = init_fn(key, train[0].positions[0], train[0].cell[0], train[0].species[0])

    logging.info(
        f"Initialized model with {sum(x.size for x in tree_util.tree_leaves(params))} parameters"
    )

    # ---- TRAINING

    opt = optax.adam(args.learning_rate)
    opt_state = opt.init(params)

    # Define the loss function
    @jit
    def loss_fn(params, batch: AtomsData):
        (energy, toccup), forces = compiled_vect_model(
            params,
            batch.positions,
            batch.cell,
            batch.species,
        )

        e_loss = jnp.square(energy - batch.energies).mean()
        f_loss = jnp.square(forces + batch.forces).sum(-1).mean()
        o_loss = jnp.mean(
            # Normal value
            jnp.square(toccup - batch.toccup).sum(-1)
            # Sum of occup
            + jnp.square(toccup.sum(-1) - batch.toccup.sum(-1))
            # Difference of occup
            + jnp.square(jnp.diff(toccup, axis=-1) - jnp.diff(batch.toccup, axis=-1))[
                :, :, 0
            ]
        )

        loss = (
            args.energy_weight * e_loss
            + args.forces_weight * f_loss
            + args.toccup_weight * o_loss
        )

        return loss, (e_loss, f_loss, o_loss)

    # Define the training function
    @jit
    def update(params, opt_state, batch: AtomsData):
        grad_fn = value_and_grad(loss_fn, has_aux=True)

        (loss, (e_loss, f_loss, o_loss)), params_grad = grad_fn(params, batch)

        updates, opt_state = opt.update(params_grad, opt_state)

        return (
            optax.apply_updates(params, updates),
            opt_state,
            (loss, e_loss, f_loss, o_loss),
        )

    # Start training loop
    logging.info("Starting training")

    lowest_loss, patience_count = jnp.inf, 0
    for i in range(args.max_epoch):
        if patience_count == args.patience:
            logging.info("Too many iterations wihtout improvement, stopping training")
            break

        # Using lax for training set since its usually big
        # def _func(j, var):
        #     params, opt_state, train_loss = var
        #
        #     params, opt_state, losses = update(params, opt_state, train[j])
        #
        #     return params, opt_state, train_loss.at[j].set(losses[0])
        #
        # params, opt_state, train_loss = fori_loop(
        #     0,
        #     len(train),
        #     _func,
        #     (params, opt_state, jnp.zeros(len(train))),
        # )

        train_loss = jnp.array([])
        for batch in train:
            params, opt_state, losses = update(params, opt_state, batch)

            train_loss = jnp.append(train_loss, losses[0])

        train_loss = train_loss.mean()

        # Validation loss
        valid_loss = jnp.array([])
        for batch in eval:
            losses = loss_fn(params, batch)

            valid_loss = jnp.append(valid_loss, losses[0])
        valid_loss = valid_loss.mean()

        logging.info(
            f"Epoch {i:4} ==> Train Loss: {train_loss:9.4f}  Validation Loss: {valid_loss:9.4f}"
        )

        if valid_loss < lowest_loss:
            lowest_loss = valid_loss
            patience_count = 0

            os.makedirs(args.chekpoints_dir, exist_ok=True)
            with open(os.path.join(args.chekpoints_dir, tag + ".pkl"), "wb") as f:
                pickle.dump([config, params, opt_state], f)
        else:
            patience_count += 1

    # ---- FINAL EVALUATION
    logging.info("Training complete, evaluating the model")

    # Load best parameters
    with open(os.path.join(args.chekpoints_dir, tag + ".pkl"), "rb") as f:
        _, params, _ = pickle.load(f)

    # Evalutate the model on Train
    train_rmse_e, train_rmse_f, train_rmse_o = evaluate_model(
        params, compiled_vect_model, train
    )

    logging.info(
        f"Validation set: RMSE energy = {train_rmse_e * 1e3:5.3f} mEv / atom   RMSE forces = {train_rmse_f * 1e3:5.3f} mEv/A   RMSE toccup = {train_rmse_o:5.3f} a.u."
    )

    # Evalutate the model on Valid
    val_rmse_e, val_rmse_f, val_rmse_o = evaluate_model(
        params, compiled_vect_model, eval
    )

    logging.info(
        f"Validation set: RMSE energy = {val_rmse_e * 1e3:5.3f} mEv / atom   RMSE forces = {val_rmse_f * 1e3:5.3f} mEv/A   RMSE toccup = {val_rmse_o:5.3f} a.u."
    )

    # Evaluate the model on Test, if present
    if test is not None:
        test_rmse_e, test_rmse_f, test_rmse_o = evaluate_model(
            params, compiled_vect_model, test
        )

        logging.info(
            f"Test set:       RMSE energy = {test_rmse_e * 1e3:5.3f} mEv / atom   RMSE forces = {test_rmse_f * 1e3:5.3f} mEv/A   RMSE toccup = {test_rmse_o:5.3f} a.u."
        )


if __name__ == "__main__":
    main()
