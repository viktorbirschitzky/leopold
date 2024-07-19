"""Utils function to use the leopold model"""

# ---- IMPORTS

# Generics
import os
import pickle

# Numpy
import numpy as np

# Jax
import jax.numpy as jnp
import jax.random as jrn

from jax import value_and_grad, vmap, jit

# Jax MD
from jax_md import nn, space, partition

# Leopold
from leopold import model_from_config

# ASE
from ase.io import read
from ase.calculators.calculator import Calculator, all_changes

# Tables
import tables as tb

# Types
from ase import Atoms
from jax import Array
from ml_collections import ConfigDict
from typing import Tuple, List, Callable, NamedTuple, Union
from tables import File
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


def apply_transform(data: AtomsData, key: str, fun: Callable) -> AtomsData:
    data_dict = data._asdict()

    data_dict[key] = fun(data_dict[key])

    return AtomsData(**data_dict)


def get_all(batch: Dataset, key: str) -> Array:
    val = []
    for data in batch:
        val.extend(data._asdict()[key])

    return jnp.array(val)


def batch_data(data: AtomsData, batch_size: int) -> Dataset:
    data_dict = data._asdict()
    n_batch = 1 + (len(data.species) // batch_size)

    # Avoid create an empty batch
    if len(data.species) % batch_size == 0:
        n_batch -= 1

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


def get_data_from_atoms(atoms: list[Atoms]) -> AtomsData:
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


def get_data_from_xyz(
    file: str, beg: int = 0, end: int = -1, step: int = 1
) -> AtomsData:
    atoms = read(
        file, index=f"{beg}:{end if end != -1 else ''}:{step}", format="extxyz"
    )
    if not isinstance(atoms, list):
        atoms = [atoms]

    return get_data_from_atoms(atoms)


def get_data_from_hdf5(
    file: Union[str, File], beg: int = 0, end: int = -1, step: int = 1
) -> AtomsData:
    close: bool = False

    if isinstance(file, str):
        file, close = tb.open_file(file), True

    # First read cell
    cell = file.root.cell.read()

    # Read species
    species = file.root.species.read()
    atom_num = np.unique(species)

    # number of atoms
    natoms = len(species)

    # HotOnes encoding
    species = np.where(species.reshape(natoms, 1) == atom_num, 1, 0)

    data = {key: [] for key in AtomsData._fields}
    for frame in file.root.frames.iterrows(beg, end, step):
        data["positions"].append(frame["positions"])
        data["cell"].append(cell)
        data["energies"].append(frame["energy"])
        data["forces"].append(frame["forces"])
        data["toccup"].append(frame["toccups"])
        data["species"].append(
            np.append(species, frame["polaron"].reshape(natoms, 1), axis=1)
        )
        data["atom_num"].append(atom_num)

    data = {key: jnp.array(val) for key, val in data.items()}

    # Close file
    if close:
        file.close()

    return AtomsData(**data)


def get_atoms_from_data(data: AtomsData) -> List[Atoms]:
    atoms = []

    for i in range(len(data.energies)):
        symbols, pol_state = data.species[i, :, :-1], data.species[i, :, -1]
        atom = Atoms(
            numbers=jnp.matmul(symbols, data.atom_num[0]),
            scaled_positions=data.positions[i],
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

def get_model_inputs_from_atoms(atoms):
    # Informations about the chemical species
    elements = jnp.unique(jnp.array(atoms.get_atomic_numbers()))


    # Collect all the other informations
    #energies = atoms.info["energy"]
    positions = atoms.get_scaled_positions()
    #forces = atom.arrays["forces"]
    cell = atoms.cell.array

    #toccup = atom.arrays["toccup"]

    atom_num = jnp.array([atoms.get_atomic_numbers()])
    _species = jnp.where(atom_num.T == elements, 1, 0)
    species = jnp.append(_species, jnp.array([atoms.arrays["pol_state"]]).T, axis=1)

    return positions, cell, species

class LeopoldCalculator(Calculator):
    """Leopold inside an ASE calculator
    """
    implemented_properties = ["energy", "forces", "free_energy", "magmom"]
    def __init__(
        self,
        model_path,
        data_path
    ):
        Calculator.__init__(self)
        self.results = {}

        # Load data
        data = get_data_from_xyz(data_path)
        data = batch_data(data, 1)

        # Load model    
        with open(model_path, "rb") as f:
            config, params, _ = pickle.load(f)

        _, apply_fn = get_model(
            data[-1],
            config,
            fractional_coordinates=True,
            disable_cell_list=True,
        )

        self.compiled_model = jit(apply_fn)
        self.params = params

    def calculate(self, atoms=None, properties=["energy","free_energy","forces","magmom"], system_changes=all_changes):
        """
        Calculate properties.

        :param atoms: ase.Atoms object
        :param properties: [str], properties to be computed, used by ASE internally
        :param system_changes: [str], system changes since last calculation, used by ASE internally
        :return:
        """
        # call to base-class to set atoms attribute

        
        Calculator.calculate(self, atoms)

        # prepare data
        pos, cell, species = get_model_inputs_from_atoms(atoms)
        # predict + extract data
        (energies, toccup), forces = self.compiled_model(
            self.params,
            pos,
            cell,
            species,
        )
        
        self.results = {}
        # only store results the model actually computed to avoid KeyErrors
        self.results["energy"] = np.array(energies)
        self.results["free_energy"] = self.results["energy"]
        # "force consistant" energy
        self.results["forces"] = -np.array(forces)
        self.results["magmom"] = np.diff(toccup, axis=-1)[:,0]
        

def get_average_num_neighbour(cell: Array, positions: Array, r_max) -> float:
    distance, _ = space.periodic_general(cell)

    dist_matrix = vmap(vmap(distance, (None, 0)), (0, None))(positions, positions)
    dist_matrix = jnp.linalg.norm(dist_matrix, axis=-1)

    return (dist_matrix < r_max).sum() / len(positions) - 1


# ---- TRAJECTORY FUNCTIONS


class TrajectoryWriter:
    file: tb.File

    def __init__(
        self,
        batch_size: int,
        description: type,
        tag: str,
        out_dir: str = ".",
        compression_level: int = 5,
    ):
        # Save batch size
        self.__batch_size = batch_size
        self.__count = 0

        # Create directory
        os.makedirs(out_dir, exist_ok=True)

        # Create Filter
        self.__filter = tb.Filters(complevel=compression_level, complib="zlib")

        # Create file
        self.file = tb.open_file(
            os.path.join(out_dir, tag) + ".h5",
            mode="w",
            title="MD Trajectory",
            filters=self.__filter,
        )

        self.__table = self.file.create_table(
            self.file.root,
            "frames",
            description=description,
            title="Simulation frames",
            expectedrows=5_000_000,
        )

        self.__frame = self.__table.row

    def __call__(self, **kwarg):
        for key, item in kwarg.items():
            self.__frame[key] = item
        self.__frame.append()
        self.__count += 1

        if self.__count == self.__batch_size:
            self.__table.flush()
            self.__count = 0

    def flush(self):
        self.__table.flush()

    def close(self):
        self.file.close()


# ---- MODEL FUNCTION


def get_model(example_batch: AtomsData, cfg: ConfigDict, **nl_kwargs):
    displacement, _ = space.periodic_general(example_batch.cell[0])
    model = model_from_config(cfg)

    neighbor = partition.neighbor_list(
        displacement,
        example_batch.cell[0],
        cfg.r_max,  # pyright: ignore
        format=partition.Sparse,
        **nl_kwargs,
    ).allocate(example_batch.positions[0])  # pyright: ignore

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
