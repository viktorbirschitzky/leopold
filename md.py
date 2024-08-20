"""Script to perform molecular dynamics using leopold"""

# ---- IMPORTS

# Generics
import logging
import pickle
import sys
import os

# Jax
import jax.numpy as jnp
import jax.random as jrn

from jax import jit

# Jax MD
from jax_md import simulate, space
from jax_md.quantity import kinetic_energy, temperature

# ASE
from ase.io import read
from ase import units

# Utils
from utils import get_model, get_data_from_atoms, construct_traj_writer

# Types
from jax import Array
from argparse import ArgumentParser, Namespace
from jax_md.simulate import NVTNoseHooverState
from typing import Union, Optional


# ---- HELPER FUNCTIONS
def arg_parse() -> Namespace:
    parser = ArgumentParser()

    # Main argument
    parser.add_argument("conf_path", type=str)
    parser.add_argument("model_path", type=str)

    # Simulation options
    parser.add_argument("--time_step", type=float, default=1)
    parser.add_argument("--temperature", type=float, default=300)
    parser.add_argument("--num_steps", type=int, default=10_000_000)
    parser.add_argument("--num_update", type=int, default=20)

    # Logging options
    parser.add_argument("--log_interval", type=int, default=5)
    parser.add_argument("--log_level", help="log level", type=str, default="INFO")

    # Saving options
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--compression_level", type=int, default=5)
    parser.add_argument("--reduced_frame", action="store_true")

    # General options
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="md")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--default_dtype", type=str, choices=["float64", "float32"], default="float64"
    )

    return parser.parse_args()


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

        fh = logging.FileHandler(os.path.join(directory, tag + ".log"), mode="a")

        # Specific formatting for file
        formatter = logging.Formatter(
            "%(asctime)s %(message)s",
            datefmt="%Y-%m-%d  %H:%M:%S",
        )
        fh.setFormatter(formatter)

        logger.addHandler(fh)


# ---- REAL APPLICATION
def main():
    args = arg_parse()

    # ---- GENERAL SETTING

    # Getting model tag and define output path
    tag = os.path.basename(args.model_path).split(".pkl")[0]
    if args.name is not None:
        tag = args.name

    out_path = os.path.join(args.out_dir, tag)

    # Default dtype
    if args.default_dtype == "float64":
        import jax

        jax.config.update("jax_enable_x64", True)

    # Logging
    setup_logger(args.log_level, tag, args.out_dir)

    # Add incipit in log file
    with open(out_path + ".log", "w") as f:
        f.write(
            f"Model: {args.model_path}     dtype: {args.default_dtype}     seed: {args.seed}\n"
        )

        f.write(
            f"{'Date':<10s} {'Real Time':<10s} {'MD Time[ps]':>12s} "
            f"{'Etot[eV]':>12s} {'Epot[eV]':>12s} {'Temp[K]':>12s} "
            f"{'Polaron[idx]':>14s} {'Pol. Mag.[a.u.]':>17s} "
            f"{'2˚ Pol.[idx]':>14s} {'2˚ Mag.[a.u.]':>17s}\n"
        )

    # ---- DATA READING

    # Read the starting configuration
    atom = read(args.conf_path)

    if isinstance(atom, list):
        atom = atom[0]

    # ---- MODEL CONSTRUCTION
    with open(args.model_path, "rb") as f:
        config, params, _ = pickle.load(f)

    # Construct data with it
    data = get_data_from_atoms([atom])

    # Use the data for the model
    _, apply_fn = get_model(
        data,
        config,
        fractional_coordinates=True,
        disable_cell_list=True,
    )

    # Needed for jax_md workflow
    def energy_fn(pos, **kwargs):
        (energy, _), _ = apply_fn(params, pos, data.cell[0], **kwargs)

        return energy

    # Solving backward compatibilty
    if not hasattr(config, "preidct_magmom"):
        config.predict_magmom = False

    # ---- DEFINE SIMULATION
    _, shift = space.periodic_general(data.cell[0])
    init_fn, step_fn = simulate.nvt_nose_hoover(
        energy_fn,
        shift,
        args.time_step * units.fs,
        args.temperature * units.kB,
    )

    @jit
    def update(state: NVTNoseHooverState, atoms: Array):
        state = step_fn(state, atoms=atoms)  # pyright: ignore

        (energy, toccup), _ = apply_fn(params, state.position, data.cell[0], atoms)

        return state, energy, toccup

    # ---- INITIALIZATION
    state = init_fn(
        jrn.PRNGKey(args.seed),
        data.positions[0],
        mass=atom.get_masses(),
        atoms=data.species[0],
    )

    # Setup trajectory writer
    writer = construct_traj_writer(tag, data.positions.shape[1], args)

    # Save the unit cell of the system
    writer.file.create_array(
        "/", "cell", data.cell[0].__array__(), title="Simulation unit cell"
    )

    # Save species informations
    writer.file.create_array(
        "/",
        "species",
        jnp.matmul(data.species[0, :, :-1], data.atom_num[0]).__array__(),
        title="Atomic numbers of the simulation's atoms",
    )

    # ---- RUN SIMULATION

    # Starting species
    atoms = data.species

    # Loop
    for i in range(args.num_steps):
        # Take a step
        state, energy, toccup = update(state, atoms[0])

        # Control if run is stable
        if jnp.isnan(energy):
            break

        # Modify Polaron position
        if not config.predict_magmom:
            magmom = jnp.diff(toccup, axis=-1)
        else:
            magmom = toccup[:, 0:1]

        pol_state = jnp.argsort(magmom.flatten())
        atoms = atoms.at[:, :, -1].set(0)
        atoms = atoms.at[:, pol_state[0], -1].set(1)

        # Compute interesting quantites
        temp = temperature(velocity=state.velocity, mass=state.mass) / units.kB

        # Save data
        if args.reduced_frame:
            writer(polaron=atoms[..., -1], positions=state.position)
        else:
            writer(
                energy=energy,
                temperature=temp,
                polaron=atoms[..., -1],
                toccups=toccup,
                positions=state.position,
                forces=state.force,
            )

        # Log results
        if i % args.log_interval == 0:
            logging.info(
                f"{i*args.time_step*1e-3:13.3f} "
                f"{kinetic_energy(velocity=state.velocity, mass=state.mass) + energy:12.3f} "
                f"{energy:12.3f} "
                f"{temp:12.3f} "
                f"{pol_state[0]:14d} {magmom[pol_state[0]].flatten()[0]:17.3f}"
                f"{pol_state[1]:15d} {magmom[pol_state[1]].flatten()[0]:17.3f}"
                f"{magmom.sum():17.3f}"
            )
    # Flush unwritten data
    writer.flush()

    # Close trajectory file
    writer.close()


if __name__ == "__main__":
    main()
