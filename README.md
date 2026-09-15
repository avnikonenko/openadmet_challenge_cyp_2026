# OpenADMET CYP challenge

The repository contains the validated exploratory analysis under `analysis/` and a
CLI-first modelling pipeline under `src/`, `models/`, `scripts/`, and `configs/`.
For the staged Slurm workflow, GPU pre-flight gate, array template, and recovery
rules, see [instruction_to_run.md](instruction_to_run.md).

## Quick start on cluster

Create the CPU environment:

```bash
conda env create -f environment-modeling.yml
conda activate openadmet-cyp-models
```

On CUDA 12.1 cluster nodes, create the GPU environment instead:

```bash
conda env create -f environment-modeling-cuda.yml
conda activate openadmet-cyp-models-cuda
```

1. LightGBM reference:

```bash
python scripts/train_lightgbm.py --config configs/lightgbm.yaml --fold 0 --seed 42 --device cpu
```

2. Direct masked-multitask D-MPNN:

```bash
python scripts/train_multitask.py --config configs/chemprop_multitask.yaml --fold 0 --seed 42 --device cuda:0 --max-runtime-minutes 50
```

3. CYP2D6 single-task control:

```bash
python scripts/train_chemprop.py --config configs/chemprop_multitask.yaml --cyp CYP2D6 --fold 0 --seed 42 --device cuda:0
```

4. Fold-safe single-concentration pretraining:

```bash
python scripts/pretrain_single_conc.py --config configs/transfer_singleconc.yaml --fold 0 --seed 42 --device cuda:0 --max-runtime-minutes 50
```

5. Full fine-tuning to direct pIC50:

```bash
python scripts/finetune_pic50.py --config configs/transfer_singleconc.yaml --checkpoint outputs/singleconc_pretrain/fold0/seed42/checkpoints/best.pt --transfer-mode full --fold 0 --seed 42 --device cuda:0 --max-runtime-minutes 50
```

6. Freeze-then-unfreeze transfer:

```bash
python scripts/finetune_pic50.py --config configs/transfer_singleconc.yaml --checkpoint outputs/singleconc_pretrain/fold0/seed42/checkpoints/best.pt --transfer-mode freeze_then_unfreeze --fold 0 --seed 42 --device cuda:0 --max-runtime-minutes 50
```

7. Five-fold, multi-seed GPU queue:

```bash
python scripts/run_matrix.py --script scripts/train_multitask.py --config configs/chemprop_multitask.yaml --folds 0 1 2 3 4 --seeds 1 2 3 --devices 0 1 2 3 4 5 6 7 --output-dir outputs --max-runtime-minutes 50
```

The same launcher runs pretraining jobs by setting `--script` to
`scripts/pretrain_single_conc.py`. Fine-tuning must use the matching fold/seed
checkpoint, which the launcher resolves safely from a template:

```bash
python scripts/run_matrix.py --script scripts/finetune_pic50.py --config configs/transfer_singleconc.yaml --folds 0 1 2 3 4 --seeds 1 2 3 --devices 0 1 2 3 4 5 6 7 --checkpoint-template 'outputs/singleconc_pretrain/fold{fold}/seed{seed}/checkpoints/best.pt' --output-dir outputs --max-runtime-minutes 50 -- --transfer-mode full
```

Add `--grid` to expand the config's controlled `search_grid`. Grid jobs are stored
under deterministic `grid000`, `grid001`, ... run names. Matching grid-based
fine-tuning checkpoints can use `{run_name}` in `--checkpoint-template`.

8. OOF evaluation:

```bash
python scripts/evaluate.py --input outputs/chemprop_multitask_direct --aggregate-oof
```

9. Blinded prediction and final arithmetic ensemble:

```bash
python scripts/predict.py --run-dir outputs/chemprop_multitask_direct/fold0/seed42 --device cuda:0
python scripts/ensemble.py --method mean --prediction-inputs outputs/lightgbm_reference/fold0/seed42/test_predictions.csv outputs/chemprop_multitask_direct/fold0/seed42/test_predictions.csv outputs/singleconc_to_pic50_full/fold0/seed42/test_predictions.csv --output-dir outputs/final_ensemble
```

For a weighted ensemble, supply the matching OOF files through `--oof-inputs`.
Weights are computed from their official macro ST-RAE values. The explicitly listed
inputs select the model families/folds/seeds. When OOF inputs are supplied, the
script writes both `test_ensemble_predictions.csv` and
`oof_ensemble_predictions.csv`; test data are never used to derive weights.

10. Validated challenge submission:

```bash
python scripts/make_submission.py --predictions outputs/final_ensemble/test_ensemble_predictions.csv --output submissions/submission.csv
```

11. Experiment leaderboard:

```bash
python scripts/leaderboard.py --input outputs --output outputs/experiment_leaderboard.csv
```

This also writes `outputs/experiment_leaderboard_aggregated.csv` across folds and
seeds.

## Reproducibility and run layout

The scripts consume the canonical identities and saved fold assignments from
`analysis/cyp_pre_model/tables/`. The split scheme is selected in YAML and each
training invocation requires an explicit fold. Available schemes are
`ecfp_cluster`, `random`, and `scaffold`.

Each run is stored as:

```text
outputs/<experiment>/fold<fold>/seed<seed>/
  config.yaml
  cli_args.json
  metadata.json
  split_molecules.csv
  predictions.csv
  metrics.json
  metrics_per_cyp.csv
  training_history.csv
  checkpoints/best.pt
  checkpoints/best_encoder.pt
  checkpoints/latest.pt
  logs/events.jsonl
  COMPLETED
```

Use `--resume` to continue from `latest.pt`; the resolved configuration must match
the interrupted run. For new runs, YAML values can be changed without editing code
through repeatable `--set section.option=value` arguments. Example:

```bash
python scripts/train_multitask.py --config configs/chemprop_multitask.yaml --fold 0 --seed 42 --device cuda:0 --set model.message_hidden_dim=600 --set model.message_passing_depth=4 --set model.dropout=0.20 --set training.batch_size=128
```

Supported neural loss modes are `standard`, `clipped_uncertainty_weighted`, and
`interval_aware`. Pretrained encoders are loaded with compatible keys, and the
source path, SHA-256 hash, loaded keys, and compatibility diagnostics are written to
run metadata.

Neural targets support `data.target_normalization: none` and
`zscore_per_target`. Statistics are fit from the fold's training rows only, stored
in checkpoints, and inverted before metrics or prediction files are written.
Precision can be `fp32`, `fp16`, `bf16`, or `auto`; on CUDA, `auto` selects bf16
when supported and otherwise fp16 AMP. Checkpoints include optimizer, optional
scheduler, AMP scaler, early-stopping, data-loader, Python, NumPy, PyTorch, and CUDA
RNG state. Encoder loads are strict by default and currently support only this
pipeline's native OpenADMET D-MPNN checkpoint/key schema. Third-party Chemprop
checkpoints require a separately implemented and tested key/feature adapter;
`--partial-load` only relaxes completeness checks and does not translate key layouts.

Every run writes `leakage_check.json`, featurization audit files,
`prediction_integrity.json`, `environment_pip_freeze.txt`, per-epoch or per-target
runtime, peak CPU/GPU memory, and optional GPU-utilization statistics. SIGTERM,
SIGINT, and SIGUSR1 request a clean stop at the next epoch/target boundary after the
latest checkpoint is saved.

## Dry run and pre-flight checklist

Run the environment, data, fold, CUDA, checkpoint, and output-path checks before
requesting a cluster allocation:

```bash
python scripts/preflight.py --config configs/chemprop_multitask.yaml --device cuda:0 --output-dir outputs
```

Add a strict checkpoint compatibility check when transferring:

```bash
python scripts/preflight.py --config configs/transfer_singleconc.yaml --device cuda:0 --checkpoint outputs/singleconc_pretrain/fold0/seed42/checkpoints/best.pt
```

Run a tiny end-to-end training smoke test either directly or through pre-flight:

```bash
python scripts/train_multitask.py --config configs/chemprop_multitask.yaml --fold 0 --seed 42 --device cuda:0 --smoke-test
python scripts/preflight.py --config configs/chemprop_multitask.yaml --device cuda:0 --run-smoke
```

Before launching the matrix, confirm that pre-flight passes, CUDA is visible, the
saved fold files and data hashes match, transfer checkpoints pass strict loading,
the output directory is writable, and a smoke run reaches `COMPLETED` with leakage,
featurization, and prediction-integrity checks marked as passed.
