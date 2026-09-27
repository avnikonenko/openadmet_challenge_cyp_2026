# OpenADMET CYP challenge

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](environment-modeling.yml)
[![PyTorch 2.3–2.5](https://img.shields.io/badge/PyTorch-2.3--2.5-EE4C2C.svg?logo=pytorch&logoColor=white)](environment-modeling.yml)
[![Chemprop 2.2.1](https://img.shields.io/badge/Chemprop-2.2.1-4C72B0.svg)](environment-modeling.yml)
[![CPU and CUDA 12.1](https://img.shields.io/badge/setup-CPU%20%7C%20CUDA%2012.1-76B900.svg?logo=nvidia&logoColor=white)](environment-modeling-cuda.yml)

The repository contains the validated exploratory analysis under `analysis/` and a
CLI-first modelling pipeline under `src/`, `models/`, `scripts/`, and `configs/`.
For the staged Slurm workflow, GPU pre-flight gate, array template, and recovery
rules, see [cluster runbook](docs/cluster.md).

## Project scope

This project develops reproducible models for the OpenADMET CYP inhibition challenge.
The main modelling task is direct-inhibition regression: predict pIC50 for CYP1A2,
CYP2C9, CYP2D6, and CYP3A4 from molecular structure. The direct-inhibition dataset
has sparse labels across CYPs, so missing targets are masked rather than treated as
inactive. The challenge definition, auxiliary assays, and submission schema are
documented in [challenge brief](docs/challenge.md).

We compare the following model families using saved chemistry-aware cross-validation
folds and the official interval-aware ST-RAE metric:

- LightGBM with Morgan (ECFP6, radius 3) fingerprints and RDKit physicochemical descriptors.
- A project-native PyTorch Chemprop-style directed message-passing neural network (D-MPNN),
  evaluated both as CYP-specific single-task models and as a masked multitask model with
  one shared encoder and separate CYP output heads.
- The official Chemprop v2 multitask baseline, kept separate from the native implementation.
- Transfer-learning D-MPNNs pretrained on fold-safe single-concentration CYP inhibition,
  then fine-tuned on direct pIC50 with either full fine-tuning or freeze-then-unfreeze.
- A GINE message-passing encoder (PyTorch Geometric `GINEConv`) as a second, genuinely
  different graph architecture, run both directly and through the same transfer sequence.
- Target-conditioned prediction, `f(molecular_embedding, CYP_embedding)`, as an
  alternative to four separate CYP heads.
- A joint multi-assay model that learns direct pIC50 and single-concentration inhibition
  together through one shared encoder, as the alternative to sequential transfer.
- Arithmetic and OOF-performance-weighted ensembles of completed model runs.

The commands and execution order for the current round are in
[next-round experiment guide](docs/experiments.md).

Random and Bemis–Murcko scaffold splits are retained as controls; the primary experiments
use saved ECFP6 cluster folds. Model selection uses training/validation data only, while the
blinded test set is reserved for final predictions.

## Data availability

This repository contains code, configurations, and derived methodology only. Challenge
CSV files, model outputs, and submissions are intentionally excluded from version control.
Obtain the data from the official
[OpenADMET CYP challenge dataset](https://huggingface.co/datasets/openadmet/cyp-challenge-train-test)
and place the checkout at `cyp-challenge-train-test/`:

```bash
git clone https://huggingface.co/datasets/openadmet/cyp-challenge-train-test cyp-challenge-train-test
```

The checkout remains an independent, ignored upstream dataset repository. Its commit
history and contributors are not part of this project's Git history; retain the source
repository and its license/provenance when using or redistributing the data.

The repository's [MIT license](LICENSE) applies to the project-authored software and
documentation only. It does **not** grant rights to the OpenADMET challenge datasets,
third-party software, pretrained checkpoints, or other externally sourced artifacts.
Those materials remain subject to their respective terms. Users are responsible for
obtaining the challenge data from the official source and confirming that their use and
redistribution comply with its current license and challenge rules.

## Documentation

- [Quick start and reproducibility](docs/quickstart.md): setup, core CLI examples, run layout, and pre-flight checks.
- [Cluster runbook](docs/cluster.md): Slurm, GPU launch, transfer sequence, and stop conditions.
- [Next-round experiments](docs/experiments.md): model variants, exact commands, and comparison order.
- [Challenge brief](docs/challenge.md): task definition, data audit, and submission contract.
- [Experiment design](docs/experiment_design.md): original pre-run hypotheses and validation plan (historical draft).

Run commands from the repository root. For a first CPU check after installing the environment and obtaining the data:

```bash
python scripts/train_lightgbm.py --config configs/lightgbm.yaml --fold 0 --seed 42 --device cpu
```
