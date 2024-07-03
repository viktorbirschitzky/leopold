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

# Optax
import optax

# Utils
from utils import get_data_from_xyz, shuffle_data, get_all, batch_data, split_data
from utils import get_model, evaluate_model, get_average_num_neighbour, apply_transform

# Types
from ml_collections import ConfigDict
from argparse import ArgumentParser, Namespace
from typing import Union, Optional
from utils import AtomsData


# ---- HELPER FUNCTIONS
def arg_parse() -> Namespace:
    parser = ArgumentParser()

    # Main argument
    parser.add_argument("database", nargs="+")

    # Database Menagement
    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=20)

    # Directory informations
    parser.add_argument("--name", type=str, default="LEOPOLD")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    parser.add_argument("--logs_dir", type=str, default="logs")

    # Model specifics
    parser.add_argument("--configuration", type=str, default=None)
    parser.add_argument(
        "--default_dtype", type=str, choices=["float64", "float32"], default="float64"
    )
    parser.add_argument("--predict_magmom", action="store_true")
    parser.add_argument("--occup_clipping", action="store_true")

    # Train specifics
    parser.add_argument("--energy_weight", type=float, default=1.0)
    parser.add_argument("--forces_weight", type=float, default=None)
    parser.add_argument("--toccup_weight", type=float, default=None)
    parser.add_argument("--max_epoch", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=5e-4)

    # General options
    parser.add_argument("--log_level", help="log level", type=str, default="INFO")
    parser.add_argument(
        "--restart", help="restart from existing checkpoint", action="store_true"
    )
    parser.add_argument(
        "--reset_opt",
        help="doesn't load opt state in restart if present",
        action="store_true",
    )

    return parser.parse_args()


def default_config() -> ConfigDict:
    config = ConfigDict()

    config.graph_net_steps = 3
    config.nonlinearities = {"e": "raw_swish", "o": "tanh"}
    config.use_sc = True
    config.n_elements = 3
    config.hidden_irreps = "42x0e + 8x1e"
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

    # Set no clipping for the magnetization
    config.occup_clipping = False

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
    if args.configuration is not None:
        logging.info(f"Read configuration from {args.configuration}")
    else:
        logging.info("Using default configuration")

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

    # If direct prediction of Magnetization is requested modify data
    if args.predict_magmom:
        logging.info("Magnetization will be predicted directly!")

        def _fun(toccup):
            return jnp.dstack((jnp.diff(toccup, axis=-1)[..., 0], toccup.sum(-1)))

        train = apply_transform(train, "toccup", _fun)
        eval = apply_transform(eval, "toccup", _fun)
        if test is not None:
            test = apply_transform(eval, "toccup", _fun)

    # Save predict_magmom in configuration
    config.predict_magmom = args.predict_magmom

    # Batching
    train = batch_data(train, args.batch_size)
    eval = batch_data(eval, args.batch_size)
    if test is not None:
        test = batch_data(test, args.batch_size)

    # ---- MODEL CONSTRUCTION

    # Update the number of atomic species
    config.n_elements = train[0].species.shape[-1]

    # Compute average number of neighbors
    posis = get_all(train, "positions")

    avg_neighbors = []

    for p in posis:
        avg_neighbors.append(
            get_average_num_neighbour(train[0].cell[0], p, config.r_max)
        )

    config.n_neighbors = float(jnp.mean(jnp.array(avg_neighbors)))

    logging.info(
        f"The average number of neighbors in the dataset is {config.n_neighbors:.2f}"
    )

    # Computing shift and scale for energies per atom
    config.shift = get_all(train, "energies").mean() / train[0].species.shape[1]
    config.scale = get_all(train, "forces").flatten().std()

    logging.info(
        f"Using a per atom energy scaling and shift of {config.scale:.2f} and {config.shift:.2f}"
    )

    # Compute species dependent shift and scale for toccup
    toccup = get_all(train, "toccup")
    species = get_all(train, "species")

    toccup_shift, toccup_scale = [], []
    for z in species.T[:-1]:
        if not config.get("occup_clipping", False):
            toccup_shift.append([toccup[z.T == 1].mean()])
            toccup_scale.append([toccup[z.T == 1].std()])
        else:
            toccup_shift.append(toccup[z.T == 1].min(0))
            toccup_scale.append(toccup[z.T == 1].max(0) - toccup[z.T == 1].min(0))

    config.scale_occ = jnp.array(toccup_scale)
    config.shift_occ = jnp.array(toccup_shift)

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

    opt = optax.chain(
        optax.adam(args.learning_rate),
        optax.contrib.reduce_on_plateau(
            0.5, patience=5, accumulation_size=len(train)
        ),  # Use the learning rate from the scheduler.
    )

    opt_state = opt.init(params)

    # Set the weight for forces and toccup
    if args.forces_weight is None:
        args.forces_weight = train[0].species.shape[-2] ** 2

    if args.toccup_weight is None:
        args.toccup_weight = train[0].species.shape[-2] ** 2

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
        f_loss = jnp.square(forces + batch.forces).mean()

        if args.predict_magmom:
            o_loss = jnp.mean(
                jnp.square(toccup - batch.toccup).sum(-1)
                + jnp.square(toccup.sum(1) - batch.toccup.sum(1))
            )
        else:
            o_loss = (
                jnp.mean(
                    # Normal value
                    jnp.square(toccup - batch.toccup).sum(-1)
                    # Sum of occup
                    + jnp.square(toccup.sum(-1) - batch.toccup.sum(-1))
                    # Difference of occup
                    + jnp.square(
                        jnp.diff(toccup, axis=-1) - jnp.diff(batch.toccup, axis=-1)
                    )[..., 0]
                )
                + jnp.square(
                    # Sum of all magnetizations
                    jnp.diff(toccup, axis=-1).sum(1)
                    - jnp.diff(batch.toccup, axis=-1).sum(1)
                ).mean()
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

        updates, opt_state = opt.update(
            params_grad, opt_state, params, value=jnp.float32(loss)
        )

        return (
            optax.apply_updates(params, updates),
            opt_state,
            (loss, e_loss, f_loss, o_loss),
        )

    # Load existing parameters if wanted
    # TODO: Aloowing for selection of checkpoint, probably by simply adding an
    # argument that defines a string to sum when you open the file
    check_path = os.path.join(args.checkpoints_dir, tag)
    if args.restart and os.path.isfile(check_path + ".pkl"):
        with open(check_path + ".pkl", "rb") as f:
            if not args.reset_opt:
                _, params, opt_state = pickle.load(f)
            else:
                _, params, _ = pickle.load(f)

        logging.info(f"Reinitialized the training from checkpoint {check_path}")

    # Start training loop
    logging.info("Starting training")

    logging.info(
        f"{'Total':>35s} {'Energy':>11s} {'Forces':>12s} {'Toccup':>12s}  {'Total':>27s} {'Energy':>11s} {'Forces':>12s} {'Toccup':>12s} | {'Learning rate':>12s}"
    )

    lowest_loss, patience_count = jnp.inf, 0
    for i in range(args.max_epoch):
        if patience_count == args.patience:
            logging.info("Too many iterations wihtout improvement, stopping training")
            break

        # Train loop and loss
        train_loss = {
            "total": jnp.array([]),
            "energy": jnp.array([]),
            "force": jnp.array([]),
            "toccup": jnp.array([]),
        }
        for batch in train:
            params, opt_state, losses = update(params, opt_state, batch)

            for key, loss in zip(train_loss.keys(), losses):
                train_loss[key] = jnp.append(train_loss[key], loss)

        for key in train_loss.keys():
            train_loss[key] = train_loss[key].mean()

        # Validation loss
        valid_loss = {
            "total": jnp.array([]),
            "energy": jnp.array([]),
            "force": jnp.array([]),
            "toccup": jnp.array([]),
        }
        for batch in eval:
            losses = loss_fn(params, batch)

            # Flatten the tuple
            losses = (losses[0], losses[1][0], losses[1][1], losses[1][2])

            for key, loss in zip(valid_loss.keys(), losses):
                valid_loss[key] = jnp.append(valid_loss[key], loss)

        for key in valid_loss.keys():
            valid_loss[key] = valid_loss[key].mean()

        # Loss logging
        train_log, valid_log = "", ""
        for (key, tval), (_, vval) in zip(train_loss.items(), valid_loss.items()):
            train_log += f"{tval:>12.8f} " if key != "total" else f"{tval:>13.8f}"
            valid_log += f"{vval:>12.8f} " if key != "total" else f"{vval:>13.8f}"

        logging.info(
            f"Epoch {i:4} ==> Train: {train_log}   Validation: {valid_log}| {args.learning_rate * optax.tree_utils.tree_get(opt_state, 'scale'):12.8f}"
        )

        # Saving checkpoints
        os.makedirs(args.checkpoints_dir, exist_ok=True)
        with open(check_path + "L.pkl", "wb") as f:
            pickle.dump([config, params, opt_state], f)

        if valid_loss["total"] < lowest_loss:
            lowest_loss = valid_loss["total"]
            patience_count = 0

            with open(check_path + ".pkl", "wb") as f:
                pickle.dump([config, params, opt_state], f)
        else:
            patience_count += 1

    # ---- FINAL EVALUATION
    logging.info("Training complete, evaluating the model")

    # Load best parameters
    with open(os.path.join(args.checkpoints_dir, tag + ".pkl"), "rb") as f:
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
