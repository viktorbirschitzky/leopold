"""Core of the Leopold model, based on jax_md implementation of nequip"""

# ---- IMPORTS

# Functools
from functools import partial

# Flax
import flax.linen as nn

# Jax
import jax.numpy as jnp
from jax import tree_util, tree_map

# Jraph
from jraph import segment_sum

# Jax MD
from jax_md import space
from jax_md.nn import nequip, util

# E3nn
from e3nn_jax import spherical_harmonics

# Types
from typing import Dict, Union, Optional
from e3nn_jax import Irreps, IrrepsArray, Irrep
from jax import Array
from ml_collections import ConfigDict
from dataclasses import field

# Helps
get_nonlinearity_by_name = util.get_nonlinearity_by_name
tree_map = partial(tree_map, is_leaf=lambda x: isinstance(x, IrrepsArray))

# ---- MODEL


class NequIPEnergyModel(nn.Module):
    """NequIP.

    Implementation follows the original paper by Batzner et al.

    nature.com/articles/s41467-022-29939-5 and partially
    https://github.com/mir-group/nequip.

      Args:
          graph_net_steps: number of NequIP convolutional layers
          use_sc: use self-connection in network (recommended)
          nonlinearities: nonlinearities to use for even/odd irreps
          n_element: number of chemical elements in input data
          hidden_irreps: irreducible representation of hidden/latent features
          sh_irreps: irreducible representations on the edges
          num_basis: number of Bessel basis functions to use
          r_max: radial cutoff used in length units
          radial_net_nonlinearity: nonlinearity to use in radial MLP
          radial_net_n_hidden: number of hidden neurons in radial MLP
          radial_net_n_layers: number of hidden layers for radial MLP
          shift: per-atom energy shift
          scale: per-atom energy scale
          n_neighbors: constant number of per-atom neighbors, used for internal
          normalization
          scalar_mlp_std: standard deviation of weight init of radial MLP

      Returns:
          Potential energy of the inputs.
    """

    graph_net_steps: int
    use_sc: bool
    nonlinearities: Union[str, Dict[str, str]]
    n_elements: int

    hidden_irreps: str
    sh_irreps: str

    num_basis: int = 8
    r_max: float = 4.0

    radial_net_nonlinearity: str = "raw_swish"
    radial_net_n_hidden: int = 64
    radial_net_n_layers: int = 2

    shift: float = 0.0
    scale: float = 1.0
    shift_occ: Optional[Array] = field(default=None)
    scale_occ: Optional[Array] = field(default=None)
    n_neighbors: float = 1.0
    scalar_mlp_std: float = 4.0

    learn_energy: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.shift_occ is None:
            self.shift_occ = jnp.zeros((self.n_elements, 2))

        if self.scale_occ is None:
            self.scale_occ = jnp.ones((self.n_elements, 2))

    @nn.compact
    def __call__(self, graph):
        r_max = jnp.float32(self.r_max)
        hidden_irreps = Irreps(self.hidden_irreps)

        # get src/dst from graph
        edge_src = graph.senders
        edge_dst = graph.receivers

        # node features
        embedding_irreps = Irreps(f"{self.n_elements}x0e")
        node_attrs = IrrepsArray(embedding_irreps, graph.nodes)

        # edge embedding
        dR = graph.edges
        scalar_dr_edge = space.distance(dR)
        edge_sh = spherical_harmonics(self.sh_irreps, dR, normalize=True)

        embedded_dr_edge = util.BesselEmbedding(
            count=self.num_basis, inner_cutoff=r_max - 0.5, outer_cutoff=r_max
        )(scalar_dr_edge)

        # embedding layer
        h_node = nequip.Linear(irreps_out=Irreps(hidden_irreps))(node_attrs)

        # convolutions
        for _ in range(self.graph_net_steps):
            h_node = nequip.NequIPConvolution(
                hidden_irreps=hidden_irreps,
                use_sc=self.use_sc,
                nonlinearities=self.nonlinearities,
                radial_net_nonlinearity=self.radial_net_nonlinearity,
                radial_net_n_hidden=self.radial_net_n_hidden,
                radial_net_n_layers=self.radial_net_n_layers,
                num_basis=self.num_basis,
                n_neighbors=self.n_neighbors,
                scalar_mlp_std=self.scalar_mlp_std,
            )(h_node, node_attrs, edge_sh, edge_src, edge_dst, embedded_dr_edge)

        # output block, two Linears that decay dimensions from h to h//2 to 1
        for mul, ir in h_node.irreps:
            if ir == Irrep("0e"):
                mul_second_to_final = mul // 2

        second_to_final_irreps = Irreps(f"{mul_second_to_final}x0e")
        final_irreps = Irreps("1x0e")
        final_irreps_mag = Irreps("2x0e")

        h_node_en = nequip.Linear(irreps_out=second_to_final_irreps)(h_node)
        atomic_output = nequip.Linear(irreps_out=final_irreps)(h_node_en).array

        # shift + scale atomic energies
        scale, shift = self.scale, self.shift
        if self.learn_energy:
            scale = self.param("energy_scale", lambda _: self.scale)
            shift = self.param("energy_shift", lambda _: self.shift)

        atomic_output = scale * atomic_output + shift

        # this aggregation follows jraph/_src/models.py
        n_graph = graph.n_node.shape[0]
        graph_idx = jnp.arange(n_graph)
        sum_n_node = tree_util.tree_leaves(
            graph.nodes, is_leaf=lambda x: isinstance(x, IrrepsArray)
        )[0].shape[0]
        node_gr_idx = jnp.repeat(
            graph_idx, graph.n_node, axis=0, total_repeat_length=sum_n_node
        )

        global_output = tree_map(
            lambda n: segment_sum(n, node_gr_idx, n_graph), atomic_output
        )

        # magnetization prediction
        h_node_mag = nequip.Linear(irreps_out=second_to_final_irreps)(h_node)
        magnetizations = nequip.Linear(irreps_out=final_irreps_mag)(h_node_mag).array

        # per species shift + scale of magnetization
        scale_occ = jnp.matmul(graph.nodes[:, :-1], self.scale_occ)
        shift_occ = jnp.matmul(graph.nodes[:, :-1], self.shift_occ)

        magnetizations = scale_occ * magnetizations + shift_occ

        return global_output, magnetizations


def model_from_config(cfg: ConfigDict) -> NequIPEnergyModel:
    """Model replication of NequIP.

    Implementation follows the original paper by Batzner et al.

    nature.com/articles/s41467-022-29939-5 and partially
    https://github.com/mir-group/nequip.
    """
    if hasattr(cfg, "scale") and hasattr(cfg, "shift"):
        scale, shift = cfg.scale, cfg.shift
    else:
        raise ValueError

    learn_energy = False
    if hasattr(cfg, "learn_energy"):
        learn_energy = cfg.learn_energy

    model = NequIPEnergyModel(
        graph_net_steps=cfg.graph_net_steps,
        use_sc=cfg.use_sc,
        nonlinearities=cfg.nonlinearities,
        n_elements=cfg.n_elements,
        hidden_irreps=cfg.hidden_irreps,
        sh_irreps=cfg.sh_irreps,
        num_basis=cfg.num_basis,
        r_max=cfg.r_max,
        radial_net_nonlinearity=cfg.radial_net_nonlinearity,
        radial_net_n_hidden=cfg.radial_net_n_hidden,
        radial_net_n_layers=cfg.radial_net_n_layers,
        shift=shift,
        scale=scale,
        shift_occ=cfg.shift_occ,
        scale_occ=cfg.scale_occ,
        n_neighbors=cfg.n_neighbors,
        scalar_mlp_std=cfg.scalar_mlp_std,
        learn_energy=learn_energy,
    )

    return model
