# OpenADMET CYP challenge

The repository contains the validated exploratory analysis under `analysis/` and a
CLI-first modelling pipeline under `src/`, `models/`, `scripts/`, and `configs/`.
For the staged Slurm workflow, GPU pre-flight gate, array template, and recovery
rules, see [instruction_to_run.md](instruction_to_run.md).

## Project scope

This project develops reproducible models for the OpenADMET CYP inhibition challenge.
The main modelling task is direct-inhibition regression: predict pIC50 for CYP1A2,
CYP2C9, CYP2D6, and CYP3A4 from molecular structure. The direct-inhibition dataset
has sparse labels across CYPs, so missing targets are masked rather than treated as
inactive. The challenge definition, auxiliary assays, and submission schema are
documented in [task.md](task.md).

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
[Next-round experiments](#next-round-experiments).

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

3. Official Chemprop v2 masked-multitask baseline:

```bash
python scripts/train_chemprop_official.py --config configs/chemprop_official_multitask.yaml --fold 0 --seed 42 --device cuda:0
```

This is the genuine upstream Chemprop 2.2.1 CLI implementation. It is separate
from the project-native D-MPNN above and uses the pinned PyTorch 2.3–2.5 range.
Its native CLI cannot safely resume an interrupted optimizer/scheduler state, so
rerun an interrupted official-Chemprop job with a new run identity; all project-native
D-MPNN jobs support `--resume`. It also rejects `--max-runtime-minutes`, because
native Chemprop 2.2 cannot stop cleanly at an epoch boundary and restore that state.

4. CYP2D6 single-task control:

```bash
python scripts/train_chemprop.py --config configs/chemprop_multitask.yaml --cyp CYP2D6 --fold 0 --seed 42 --device cuda:0
```

5. Fold-safe single-concentration pretraining:

```bash
python scripts/pretrain_single_conc.py --config configs/transfer_singleconc.yaml --fold 0 --seed 42 --device cuda:0 --max-runtime-minutes 50
```

6. Full fine-tuning to direct pIC50:

```bash
python scripts/finetune_pic50.py --config configs/transfer_singleconc.yaml --checkpoint outputs/singleconc_pretrain/fold0/seed42/checkpoints/best.pt --transfer-mode full --fold 0 --seed 42 --device cuda:0 --max-runtime-minutes 50
```

7. Freeze-then-unfreeze transfer:

```bash
python scripts/finetune_pic50.py --config configs/transfer_singleconc.yaml --checkpoint outputs/singleconc_pretrain/fold0/seed42/checkpoints/best.pt --transfer-mode freeze_then_unfreeze --fold 0 --seed 42 --device cuda:0 --max-runtime-minutes 50
```

8. Five-fold, multi-seed GPU queue:

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

Every completed training run updates this same CSV automatically. The registry keeps
one row per run/CYP, includes per-run and CV aggregate metrics, hyperparameters,
resource usage, status, paths, and UTC run/registration timestamps. Updates use a
file lock and atomic replacement, so independent cluster jobs can finish concurrently.
Rerunning the command refreshes existing rows and adds newly discovered models without
duplicating earlier runs.

## Next-round experiments

This round adds capacity, architecture, and multi-assay comparisons on top of the
existing CLI. No experiment below has its own training script: each one is selected
through a YAML configuration and the scripts already listed above. New configurations
are `configs/transfer_singleconc_large.yaml`,
`configs/transfer_singleconc_xlarge.yaml`, `configs/transfer_singleconc_search.yaml`,
`configs/gine_multitask.yaml`, `configs/gine_transfer_singleconc.yaml`,
`configs/target_conditioned_transfer.yaml`, and `configs/joint_multiassay.yaml`.

### Which loss each stage actually optimizes

`loss.loss_mode` is a shared fallback; `pretraining:` and `transfer:` may now set their
own `loss_mode` (or a nested `loss:` mapping). A configuration without stage-specific
losses behaves exactly as before, and every run records `pretraining_loss_mode`,
`finetuning_loss_mode`, `transfer_mode`, `architecture`, `encoder_type`, and
`head_type` in `metadata.json` and in the leaderboard.

Two facts worth stating explicitly before comparing this round against the previous one:

- `configs/transfer_singleconc.yaml` sets `loss.loss_mode: interval_aware` at the top
  level only, so **both** its stages resolve to `interval_aware`. It is left unchanged
  so existing results stay reproducible, and no redundant "transfer + interval-aware"
  experiment is added.
- During single-concentration pretraining, `interval_aware` is numerically identical to
  squared error, because that assay supplies no confidence bounds and the loss falls
  back to the point loss for every row. The new transfer configs therefore declare
  `pretraining.loss_mode: standard` while the scored stage inherits `interval_aware`
  from the shared `loss:` block. That changes nothing numerically but records the truth,
  and it makes the hypothesis testable: override `pretraining.loss_mode` alone and the
  fine-tuning stage is unaffected.

A stage key overrides the shared block, so each stage is set in exactly one place: use
`--set pretraining.loss_mode=…` for the pretraining stage and `--set loss.loss_mode=…`
for the scored stage. No config sets a stage loss that merely repeats the shared one,
so an override can never be silently ignored.

### A. Larger D-MPNN transfer (config only)

```bash
python scripts/run_matrix.py --script scripts/pretrain_single_conc.py --config configs/transfer_singleconc_large.yaml --folds 0 1 2 3 4 --seeds 1 2 3 --devices 0 1 2 3 4 5 6 7 --output-dir outputs --max-runtime-minutes 50
python scripts/run_matrix.py --script scripts/finetune_pic50.py --config configs/transfer_singleconc_large.yaml --folds 0 1 2 3 4 --seeds 1 2 3 --devices 0 1 2 3 4 5 6 7 --checkpoint-template 'outputs/singleconc_pretrain_large/fold{fold}/seed{seed}/checkpoints/best.pt' --output-dir outputs --max-runtime-minutes 50 -- --transfer-mode full
```

Swap `_large` for `_xlarge` to run the 1024-wide, depth-5 variant; its pretraining
checkpoints live under `outputs/singleconc_pretrain_xlarge/`.

### B. Target-conditioned transfer

`f(molecular_embedding, CYP_embedding) -> pIC50` with one shared predictor instead of
four separate heads. The encoder is unchanged, so its checkpoints remain interchangeable
with the `cyp_specific_heads` runs.

```bash
python scripts/run_matrix.py --script scripts/pretrain_single_conc.py --config configs/target_conditioned_transfer.yaml --folds 0 1 2 3 4 --seeds 1 2 3 --devices 0 1 2 3 4 5 6 7 --output-dir outputs --max-runtime-minutes 50
python scripts/run_matrix.py --script scripts/finetune_pic50.py --config configs/target_conditioned_transfer.yaml --folds 0 1 2 3 4 --seeds 1 2 3 --devices 0 1 2 3 4 5 6 7 --checkpoint-template 'outputs/target_conditioned_pretrain/fold{fold}/seed{seed}/checkpoints/best.pt' --output-dir outputs --max-runtime-minutes 50 -- --transfer-mode full
```

### C. GINE direct multitask

GINE (`torch_geometric.nn.GINEConv`, residual, LayerNorm, global mean pooling) is
selected by `model.type`; the same trainer, folds, checkpoints, resume, and evaluation
apply. It consumes the existing atom features and uses bond features as edge attributes.

```bash
python scripts/train_multitask.py --config configs/gine_multitask.yaml --fold 0 --seed 1 --device cuda:0 --max-runtime-minutes 50
python scripts/run_matrix.py --script scripts/train_multitask.py --config configs/gine_multitask.yaml --folds 0 1 2 3 4 --seeds 1 2 3 --devices 0 1 2 3 4 5 6 7 --output-dir outputs --max-runtime-minutes 50
```

Any D-MPNN config can also be switched per run: `--set model.type=gine --set model.hidden_dim=512 --set model.num_layers=5`.

### D. GINE transfer

Paired with C, this separates "transfer learning helps" from "the D-MPNN architecture helps".

```bash
python scripts/run_matrix.py --script scripts/pretrain_single_conc.py --config configs/gine_transfer_singleconc.yaml --folds 0 1 2 3 4 --seeds 1 2 3 --devices 0 1 2 3 4 5 6 7 --output-dir outputs --max-runtime-minutes 50
python scripts/run_matrix.py --script scripts/finetune_pic50.py --config configs/gine_transfer_singleconc.yaml --folds 0 1 2 3 4 --seeds 1 2 3 --devices 0 1 2 3 4 5 6 7 --checkpoint-template 'outputs/gine_singleconc_pretrain/fold{fold}/seed{seed}/checkpoints/best.pt' --output-dir outputs --max-runtime-minutes 50 -- --transfer-mode full
```

### E. Joint multi-assay model

Direct pIC50 and single-concentration inhibition are trained together through one shared
encoder with learned CYP and assay embeddings, as the alternative to sequential transfer.
`data.task_set: joint_multiassay` selects the joint table, so the existing multitask
trainer runs it. Only the four direct-pIC50 endpoints are scored and predicted, so its
out-of-fold metrics and submission files are directly comparable. Every outer-validation
molecule is withheld from training in both assays.

```bash
python scripts/run_matrix.py --script scripts/train_multitask.py --config configs/joint_multiassay.yaml --folds 0 1 2 3 4 --seeds 1 2 3 --devices 0 1 2 3 4 5 6 7 --output-dir outputs --max-runtime-minutes 50
```

### F. CYP2D6 endpoint refinement

A second stage after multitask transfer: load a completed run, keep the shared encoder
and that endpoint's head, and fine-tune one CYP. CYP2D6 is the intended first target
because its pairwise correlations with the other CYPs are near zero. The refinement must
use the same fold as its source run and becomes an ordinary run of its own.

```bash
python scripts/refine_endpoint.py --run-dir outputs/singleconc_to_pic50_full/fold0/seed1 --cyp CYP2D6 --fold 0 --seed 1 --device cuda:0 --encoder-mode frozen
python scripts/refine_endpoint.py --run-dir outputs/singleconc_to_pic50_full/fold0/seed1 --cyp CYP2D6 --fold 0 --seed 1 --device cuda:0 --encoder-mode full --learning-rate 5.0e-5
```

Defaults come from `transfer.endpoint_refinement` in the source run's configuration
(`encoder_mode`, `learning_rate`, `epochs`) and CLI flags override them.

A refinement run covers one endpoint, so its `predictions.csv` and `test_predictions.csv`
contain only that CYP. `scripts/ensemble.py` requires identical molecule/CYP coverage
across its inputs, so a refined endpoint cannot be averaged directly with a four-endpoint
model; compare it against the source run's CYP2D6 rows in
`evaluation/metrics_by_fold_seed_cyp.csv` first, and substitute the column deliberately
when assembling a submission.

### G. Experiment-matrix launch across 8 GPUs

`--experiment-matrix` runs exactly the named configurations in the selected YAML
list — no Cartesian product. `configs/transfer_singleconc_search.yaml`
declares fourteen chosen points around the current best model; its `search_space:` block
records the declared axes but is deliberately not a `search_grid`, so `--grid` cannot
expand it into 648 runs. The old `--grid` behaviour is unchanged.

```bash
python scripts/run_matrix.py --script scripts/pretrain_single_conc.py --config configs/transfer_singleconc_search.yaml --folds 0 1 2 3 4 --seeds 1 2 3 --devices 0 1 2 3 4 5 6 7 --experiment-matrix --experiment-matrix-section pretraining_experiments --output-dir outputs --max-runtime-minutes 50
python scripts/run_matrix.py --script scripts/finetune_pic50.py --config configs/transfer_singleconc_search.yaml --folds 0 1 2 3 4 --seeds 1 2 3 --devices 0 1 2 3 4 5 6 7 --experiment-matrix --checkpoint-template 'outputs/singleconc_pretrain_search/fold{fold}/seed{seed}/{checkpoint_run_name}/checkpoints/best.pt' --output-dir outputs --max-runtime-minutes 50 -- --transfer-mode full
```

Each entry's `name` becomes the run name, so results land in
`outputs/<experiment>/fold<N>/seed<N>/<name>/` and the fine-tuning stage resolves its
matching pretrained encoder through `{checkpoint_run_name}`. Several fine-tuning-only
variants can therefore reuse one compatible deterministic pretraining run.

### H. Paired model comparison

```bash
python scripts/evaluate.py --input outputs --aggregate-oof --reference-model singleconc_to_pic50_full
```

`evaluation/model_comparison.csv` reports, per model, the mean and SD of macro ST-RAE
overall and separately across folds and across seeds, the raw delta against the
reference, the fold/seed-**paired** delta with a bootstrap confidence interval, the
number of folds improved, MAE, Spearman, and the number of scored CYPs. A difference
such as 0.691 versus 0.696 is not a result unless the paired interval excludes zero and
the folds agree; the `scored_CYPs` column exists because a macro over one refined
endpoint is not comparable with a macro over four.

`evaluation/oof_residual_complementarity.csv` reports, for every model pair, the Pearson
and Spearman correlation of out-of-fold residuals, overall and per CYP, plus mean and
maximum absolute disagreement. Consult it before choosing ensemble members: two models
with residual correlation near 1 and small disagreement contribute the same information,
so a freeze/unfreeze variant should not be ensembled with full fine-tuning on the
strength of a small score difference alone. `scripts/ensemble.py` writes the same table
for the specific members passed to it whenever `--oof-inputs` is supplied; ensemble
weights continue to come only from out-of-fold scores.

### Capacity diagnostics

Every neural run writes `model_capacity.json` (total, encoder, head, and trainable
parameter counts) and records them in `metadata.json` and the leaderboard.
`training_history.csv` gains per-epoch gradient-norm mean/max, per-group learning rates,
per-CYP validation ST-RAE, and whether the encoder was trainable, so a larger model that
is merely overfitting can be told from one that is genuinely better.

### Multi-stage transfer

`transfer.stages` declares an encoder lineage of more than one stage. The final entry is
the stage being trained; earlier entries may carry checkpoints, and the last one that
does is the initialization source, which `--checkpoint` overrides. Each stage's path,
SHA-256, source task, architecture, and stage name are written to run metadata.

```yaml
transfer:
  stages:
    - name: molecular_pretraining
      checkpoint: null
    - name: single_concentration
      checkpoint: null
    - name: direct_pic50
```

Without this block, transfer behaves exactly as before. External encoders are added
through `models/encoder_registry.py`: `build_encoder` constructs `native_dmpnn` and
`gine`, and `load_pretrained_encoder` dispatches on the checkpoint's source format. A
third-party checkpoint requires its own registered adapter that uses that project's API
or a verified key mapping; `--partial-load` relaxes completeness checks for native
checkpoints only and must never be used to silence an incompatible foreign layout.

### Recommended execution order

Run each round to completion and review the paired comparison before starting the next.

- **Round A — capacity.** Reproducibility check of the current best transfer model, then
  large, then xlarge, then the fourteen controlled configurations via `--experiment-matrix`.
- **Round B — architecture.** Target-conditioned transfer, then CYP2D6 endpoint
  refinement from the best available multitask run.
- **Round C — encoder family.** GINE direct, then GINE single-concentration transfer.
- **Round D — multi-assay.** The joint model, compared against sequential transfer.
- **Only afterwards:** choose architectures for expensive pretrained/foundation-encoder
  work, and build the final ensemble from out-of-fold error complementarity.

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
