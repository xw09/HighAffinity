## HighAffinity
This is the official implementation for the paper titled 'HighAffinity: Fine-tuning Boltz2 for Structure-based Protein–Peptide Binding Affinity Prediction'.

## Installation

> Note: we recommend installing boltz in a fresh python environment

Install boltz with PyPI (recommended):

```
pip install boltz -U
```

## Inference

You can run inference using Boltz with:

```
boltz predict input_path --use_msa_server
```

`input_path` should point to a YAML file, or a directory of YAML files for batched processing, describing the biomolecules you want to model and the properties you want to predict (e.g. affinity). To see all available options: `boltz predict --help` and for more information on these input formats, see our [prediction instructions](docs/prediction.md). By default, the `boltz` command will run the latest version of the model.

## Data

HighAffinity/data/Database.csv

## Train

HighAffinity/boltz/src/boltz/affinity_train.py

## Test

HighAffinity/boltz/src/boltz/affinity_test.py