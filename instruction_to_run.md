# OpenADMET CYP: cluster run instructions

## Readiness

The CLI pipeline has passed CPU unit tests, CLI smoke training, resume, prediction,
evaluation, ensembling, and submission validation. The local host has no visible CUDA
device, so GPU execution must be qualified on the target cluster before launching the
full matrix.

This runbook assumes Slurm and one GPU per job. The code supports CUDA device selection,
walltime-safe checkpoints, deterministic settings, fold-safe transfer, and `--resume`.
Do not run two jobs with the same experiment/fold/seed/run-name identity concurrently.

## 1. One-time setup

From the repository root:

```bash
conda env create -f environment-modeling-cuda.yml
conda activate openadmet-cyp-models-cuda
```

If the node CUDA runtime is not compatible with the CUDA 12.1 environment, use the CPU
environment only for LightGBM or create a cluster-specific environment. Do not silently
fall back to CPU for neural jobs.

Confirm the data and saved split files are present:

```bash
test -f cyp-challenge-train-test/cyp-challenge-TRAIN_inhibition.csv
test -f cyp-challenge-train-test/cyp-challenge-single-concentration-TRAIN.csv
test -f cyp-challenge-train-test/cyp-challenge-TEST-BLINDED.csv
test -f analysis/cyp_pre_model/tables/cv_fold_assignments.csv
```

## 2. GPU pre-flight: required before the matrix

Request a short interactive GPU allocation, then run:

```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=24G --time=00:20:00 --pty bash
conda activate openadmet-cyp-models-cuda
nvidia-smi
python scripts/preflight.py --config configs/chemprop_multitask.yaml --device cuda:0 --output-dir outputs
python scripts/train_multitask.py --config configs/chemprop_multitask.yaml --fold 0 --seed 42 --device cuda:0 --output-dir outputs --experiment cluster_smoke --run-name gpu_smoke --smoke-test
```

Proceed only if pre-flight passes and the smoke run writes `COMPLETED`,
`leakage_check.json`, `featurization_summary.json`, and `prediction_integrity.json`.
Inspect `metadata.json` for the selected precision mode and GPU memory. Delete neither
the smoke run nor its logs; they are the environment audit trail.

## 3. Recommended first production matrix

Use five ECFP-cluster folds and seeds `1 2 3`. Start with these four model families:

1. LightGBM reference: CPU, per CYP.
2. Direct masked multitask D-MPNN: GPU.
3. Official Chemprop 2.2.1 masked multitask baseline: GPU.
4. Single-concentration-pretrained D-MPNN followed by full fine-tuning: GPU.

Run the freeze-then-unfreeze transfer variant only after the full-transfer runs have
completed and been evaluated. Do not use the YAML search grid for the first matrix.

The official Chemprop baseline is intentionally independent of the native D-MPNN:

```bash
python scripts/preflight.py --config configs/chemprop_official_multitask.yaml --device cuda:0 --output-dir outputs
python scripts/train_chemprop_official.py --config configs/chemprop_official_multitask.yaml --fold 0 --seed 42 --device cuda:0 --output-dir outputs
```

It uses the upstream Chemprop 2.2.1 CLI and the pinned PyTorch 2.3–2.5 range in
the supplied environments. Chemprop’s v2.2 CLI does not restore native optimizer
and scheduler state, so do not pass `--resume` to this script; choose a new run name
after an interruption. It also rejects `--max-runtime-minutes`; request enough walltime
for the configured epoch budget. The native D-MPNN scripts do support exact `--resume`
and walltime-safe stopping.

## 4. Single-node, eight-GPU launcher

Use this only inside one allocation that owns all listed GPUs. `run_matrix.py` queues
one independent fold/seed process per GPU and writes a launcher summary.

```bash
srun --partition=gpu --gres=gpu:8 --cpus-per-task=32 --mem=128G --time=01:00:00 --pty bash
conda activate openadmet-cyp-models-cuda

python scripts/run_matrix.py \
  --script scripts/train_multitask.py \
  --config configs/chemprop_multitask.yaml \
  --folds 0 1 2 3 4 --seeds 1 2 3 --devices 0 1 2 3 4 5 6 7 \
  --output-dir outputs --max-runtime-minutes 50
```

The launcher log and `summary.csv` are written below `outputs/launcher_logs/`.
Checkpointed jobs may be resumed with the identical command plus `--resume`.

## 5. Slurm job-array template: preferred for multi-node clusters

Save the following as a site-local Slurm script, replacing the partition/account/module
lines to match the cluster. It intentionally uses `cuda:0`: Slurm exposes the assigned
GPU as the first visible GPU within each one-GPU job.

```bash
#!/usr/bin/env bash
#SBATCH --job-name=openadmet-direct
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --array=0-14
#SBATCH --output=slurm_logs/direct_%A_%a.out
#SBATCH --error=slurm_logs/direct_%A_%a.err

set -euo pipefail
conda activate openadmet-cyp-models-cuda

folds=(0 1 2 3 4)
seeds=(1 2 3)
fold=${folds[$((SLURM_ARRAY_TASK_ID / 3))]}
seed=${seeds[$((SLURM_ARRAY_TASK_ID % 3))]}

python scripts/train_multitask.py \
  --config configs/chemprop_multitask.yaml \
  --fold "$fold" --seed "$seed" --device cuda:0 \
  --output-dir outputs --max-runtime-minutes 50
```

For an interrupted job, resubmit the same array element with `--resume`. Do not use
`--resume` after changing the resolved YAML configuration.

## 6. Transfer learning sequence

First run fold-safe auxiliary pretraining for the same fold/seed matrix:

```bash
python scripts/run_matrix.py \
  --script scripts/pretrain_single_conc.py \
  --config configs/transfer_singleconc.yaml \
  --folds 0 1 2 3 4 --seeds 1 2 3 --devices 0 1 2 3 4 5 6 7 \
  --output-dir outputs --max-runtime-minutes 50
```

Verify every expected pretraining checkpoint before fine-tuning:

```bash
python scripts/preflight.py \
  --config configs/transfer_singleconc.yaml --device cuda:0 \
  --checkpoint outputs/singleconc_pretrain/fold0/seed1/checkpoints/best.pt \
  --output-dir outputs
```

Then run full fine-tuning. Each fold/seed loads only its matching pretrained encoder:

```bash
python scripts/run_matrix.py \
  --script scripts/finetune_pic50.py \
  --config configs/transfer_singleconc.yaml \
  --folds 0 1 2 3 4 --seeds 1 2 3 --devices 0 1 2 3 4 5 6 7 \
  --checkpoint-template 'outputs/singleconc_pretrain/fold{fold}/seed{seed}/checkpoints/best.pt' \
  --output-dir outputs --max-runtime-minutes 50 -- --transfer-mode full
```

For a Slurm dependency, submit fine-tuning only after its pretraining job has succeeded:

```bash
pretrain_job=$(sbatch --parsable pretrain_array.sbatch)
sbatch --dependency=afterok:"$pretrain_job" finetune_array.sbatch
```

The fine-tuning array must use the same fold/seed mapping and checkpoint template as
above. `leakage_check.json` fails the run if outer-validation molecules appear in the
auxiliary pretraining partition.

## 7. CPU LightGBM reference

Run this on CPU nodes. Four threads per task is the default upper bound unless Slurm
sets `SLURM_CPUS_PER_TASK`; do not allocate many CPUs without increasing that setting.

```bash
python scripts/run_matrix.py \
  --script scripts/train_lightgbm.py \
  --config configs/lightgbm.yaml \
  --folds 0 1 2 3 4 --seeds 1 2 3 --devices cpu cpu cpu cpu \
  --output-dir outputs --max-runtime-minutes 50
```

For a CPU array, invoke `train_lightgbm.py` directly with the same fold/seed mapping
used in the GPU template and `--device cpu`.

## 8. Evaluate before test prediction

Evaluate completed runs only. Checkpointed or failed runs are explicitly excluded from
metrics and OOF aggregation.

```bash
python scripts/evaluate.py --input outputs/chemprop_multitask_direct --aggregate-oof
python scripts/evaluate.py --input outputs/singleconc_to_pic50_full --aggregate-oof
python scripts/evaluate.py --input outputs/lightgbm_reference --aggregate-oof
python scripts/leaderboard.py --input outputs --output outputs/experiment_leaderboard.csv
```

Before choosing a final family or ensemble, inspect `missing_folds.csv`, `run_status.csv`,
the aggregate leaderboard, and every run's `metadata.json`. Do not select on blinded
test predictions.

## 9. Blinded test predictions, ensemble, and submission

Only predict from runs with a `COMPLETED` marker. `--resume` validates existing test
predictions against run provenance and checkpoint hashes.

```bash
python scripts/predict.py \
  --run-dir outputs/chemprop_multitask_direct/fold0/seed1 --device cuda:0
```

For a weighted ensemble, provide matched run-level test and OOF files in the same order.
Weights are derived from OOF macro ST-RAE only.

```bash
python scripts/ensemble.py --method weighted \
  --prediction-inputs \
    outputs/lightgbm_reference/fold0/seed1/test_predictions.csv \
    outputs/chemprop_multitask_direct/fold0/seed1/test_predictions.csv \
    outputs/singleconc_to_pic50_full/fold0/seed1/test_predictions.csv \
  --oof-inputs \
    outputs/lightgbm_reference/fold0/seed1/predictions.csv \
    outputs/chemprop_multitask_direct/fold0/seed1/predictions.csv \
    outputs/singleconc_to_pic50_full/fold0/seed1/predictions.csv \
  --output-dir outputs/final_ensemble

python scripts/make_submission.py \
  --predictions outputs/final_ensemble/test_ensemble_predictions.csv \
  --output submissions/submission.csv
```

The ensemble script requires exact molecule/CYP coverage and canonical molecule identity
agreement. The submission writer validates challenge order, required columns, duplicates,
missing values, and finite numeric predictions.

## 10. Stop conditions

Do not continue to the next stage if any of these occur:

- GPU pre-flight or smoke test fails.
- A run is marked `failed`, has no `COMPLETED` marker, or lacks `best.pt`.
- `leakage_check.json`, featurization audit, or prediction integrity check fails.
- `evaluate.py` reports missing folds or incomplete runs.
- A transfer checkpoint fails strict compatibility checks.

Resolve the failing job first, then resume only that exact fold/seed/configuration.
