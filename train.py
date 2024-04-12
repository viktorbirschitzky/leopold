"""Train script for the leopold architecture"""

import jax
import jax.numpy as jnp
from jax import vmap, jit, grad, random

from jax_md import nn
from jax_md import space, energy
from jax_md.nn import nequip_pol

import numpy as np
import yaml

from argparse import ArgumentParser, Namespace

jax.config.update("jax_enable_x64", True)


def arg_parse() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("configuration", type=str)

    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)

    return parser.parse_args()
