# Leopold: LEarning Of POLaron Dynamics

## Overview

Leopold is a machine learning package designed to model small polaron dynamics at the accuracy of density functional theory (DFT). By leveraging a message-passing neural network (MPNN) trained on first-principles molecular dynamics (FPMD) data, Leopold enables nanosecond-scale simulations of polaron hopping dynamics, overcoming the timescale limitations of traditional FPMD approaches.

This repository contains the code, data processing scripts, and trained models used in the paper:

**Machine Learning Small Polaron Dynamics**\
Viktor C. Birschitzky\*, Luca Leoni\*, Michele Reticcioli, and Cesare Franchini\
*University of Vienna & Alma Mater Studiorum - Università di Bologna*

## Features

**Message-Passing Neural Network (MPNN):** Based on a modified version of the Neural Equivariant Interatomic Potential (NequIP) architecture as implemented in the [JAX-MD repository](https://github.com/google/jax-md).
- **Polaron Encoding:** Explicit charge state encoding to ensure charge conservation.
- **Occupation Prediction:** Direct prediction of site occupation to track polaron hopping.
- **Implemented in JAX:** Optimized for high-performance machine learning training and inference.

## Installation

Leopold requires Python 3.10+ and to set up the environment, run:

```sh
pip install .
```

Alternatively, install dependencies directly using:

```sh
pip install -e .
```

To create a virtual environment and install the package:

```sh
python -m venv venv
source venv/bin/activate
pip install .
```

## Usage

### Training the Model

1. Prepare the dataset following the format in `data/`.
2. Train the model using the `train.py` script. Get help with:

```sh
python src/scripts/train.py --help
```

### Running ML-Based MD Simulations

To run a polaron dynamics simulation using the trained model use the `md.py` script. Get help with: 

```sh
python src/scripts/md.py --help
```

### Relaxations using Leopold

An example notebook for relaxing structures using ASE is provided in `notebooks/relax.ipynb`

## Data

The dataset consists of first-principles molecular dynamics (FPMD) data generated using VASP. A minimal preprocessed dataset is provided in `data/MgO`. Full datasets are available at publication via a repository

## Citation

If you use Leopold in your research, please cite our paper:

```
@article{birschitzky2024mlpolaron,
  author    = {Viktor C. Birschitzky and Luca Leoni and Michele Reticcioli and Cesare Franchini},
  title     = {Machine Learning Small Polaron Dynamics},
  journal   = {arXiv preprint arXiv:2409.16179},
  year      = {2024}
}
```

## License

This project is licensed under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).


## Contact

For questions and contributions, please open an issue or contact the authors.

