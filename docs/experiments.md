# Next-round experiments

This round adds capacity, architecture, and multi-assay comparisons on top of the
existing CLI. No experiment below has its own training script: each one is selected
through a YAML configuration and the scripts in the [quick start](quickstart.md). New configurations
are `configs/transfer_singleconc_large.yaml`,
`configs/transfer_singleconc_xlarge.yaml`, `configs/transfer_singleconc_search.yaml`,
`configs/gine_multitask.yaml`, `configs/gine_transfer_singleconc.yaml`,
`configs/target_conditioned_transfer.yaml`, and `configs/joint_multiassay.yaml`.

## Which loss each stage actually optimizes

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

## A. Larger D-MPNN transfer (config only)

```bash
python scripts/run_matrix.py --script scripts/pretrain_single_conc.py --config configs/transfer_singleconc_large.yaml --folds 0 1 2 3 4 --seeds 1 2 3 --devices 0 1 2 3 4 5 6 7 --output-dir outputs --max-runtime-minutes 50
python scripts/run_matrix.py --script scripts/finetune_pic50.py --config configs/transfer_singleconc_large.yaml --folds 0 1 2 3 4 --seeds 1 2 3 --devices 0 1 2 3 4 5 6 7 --checkpoint-template 'outputs/singleconc_pretrain_large/fold{fold}/seed{seed}/checkpoints/best.pt' --output-dir outputs --max-runtime-minutes 50 -- --transfer-mode full
```

Swap `_large` for `_xlarge` to run the 1024-wide, depth-5 variant; its pretraining
checkpoints live under `outputs/singleconc_pretrain_xlarge/`.

## B. Target-conditioned transfer

`f(molecular_embedding, CYP_embedding) -> pIC50` with one shared predictor instead of
four separate heads. The encoder is unchanged, so its checkpoints remain interchangeable
with the `cyp_specific_heads` runs.

```bash
python scripts/run_matrix.py --script scripts/pretrain_single_conc.py --config configs/target_conditioned_transfer.yaml --folds 0 1 2 3 4 --seeds 1 2 3 --devices 0 1 2 3 4 5 6 7 --output-dir outputs --max-runtime-minutes 50
python scripts/run_matrix.py --script scripts/finetune_pic50.py --config configs/target_conditioned_transfer.yaml --folds 0 1 2 3 4 --seeds 1 2 3 --devices 0 1 2 3 4 5 6 7 --checkpoint-template 'outputs/target_conditioned_pretrain/fold{fold}/seed{seed}/checkpoints/best.pt' --output-dir outputs --max-runtime-minutes 50 -- --transfer-mode full
```

## C. GINE direct multitask

GINE (`torch_geometric.nn.GINEConv`, residual, LayerNorm, global mean pooling) is
selected by `model.type`; the same trainer, folds, checkpoints, resume, and evaluation
apply. It consumes the existing atom features and uses bond features as edge attributes.

```bash
python scripts/train_multitask.py --config configs/gine_multitask.yaml --fold 0 --seed 1 --device cuda:0 --max-runtime-minutes 50
python scripts/run_matrix.py --script scripts/train_multitask.py --config configs/gine_multitask.yaml --folds 0 1 2 3 4 --seeds 1 2 3 --devices 0 1 2 3 4 5 6 7 --output-dir outputs --max-runtime-minutes 50
```

Any D-MPNN config can also be switched per run: `--set model.type=gine --set model.hidden_dim=512 --set model.num_layers=5`.

## D. GINE transfer

Paired with C, this separates "transfer learning helps" from "the D-MPNN architecture helps".

```bash
python scripts/run_matrix.py --script scripts/pretrain_single_conc.py --config configs/gine_transfer_singleconc.yaml --folds 0 1 2 3 4 --seeds 1 2 3 --devices 0 1 2 3 4 5 6 7 --output-dir outputs --max-runtime-minutes 50
python scripts/run_matrix.py --script scripts/finetune_pic50.py --config configs/gine_transfer_singleconc.yaml --folds 0 1 2 3 4 --seeds 1 2 3 --devices 0 1 2 3 4 5 6 7 --checkpoint-template 'outputs/gine_singleconc_pretrain/fold{fold}/seed{seed}/checkpoints/best.pt' --output-dir outputs --max-runtime-minutes 50 -- --transfer-mode full
```

## E. Joint multi-assay model

Direct pIC50 and single-concentration inhibition are trained together through one shared
encoder with learned CYP and assay embeddings, as the alternative to sequential transfer.
`data.task_set: joint_multiassay` selects the joint table, so the existing multitask
trainer runs it. Only the four direct-pIC50 endpoints are scored and predicted, so its
out-of-fold metrics and submission files are directly comparable. Every outer-validation
molecule is withheld from training in both assays.

```bash
python scripts/run_matrix.py --script scripts/train_multitask.py --config configs/joint_multiassay.yaml --folds 0 1 2 3 4 --seeds 1 2 3 --devices 0 1 2 3 4 5 6 7 --output-dir outputs --max-runtime-minutes 50
```

## F. CYP2D6 endpoint refinement

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

## G. Experiment-matrix launch across 8 GPUs

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

## H. Paired model comparison

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

## Capacity diagnostics

Every neural run writes `model_capacity.json` (total, encoder, head, and trainable
parameter counts) and records them in `metadata.json` and the leaderboard.
`training_history.csv` gains per-epoch gradient-norm mean/max, per-group learning rates,
per-CYP validation ST-RAE, and whether the encoder was trainable, so a larger model that
is merely overfitting can be told from one that is genuinely better.

## Multi-stage transfer

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

## Recommended execution order

Run each round to completion and review the paired comparison before starting the next.

- **Round A — capacity.** Reproducibility check of the current best transfer model, then
  large, then xlarge, then the fourteen controlled configurations via `--experiment-matrix`.
- **Round B — architecture.** Target-conditioned transfer, then CYP2D6 endpoint
  refinement from the best available multitask run.
- **Round C — encoder family.** GINE direct, then GINE single-concentration transfer.
- **Round D — multi-assay.** The joint model, compared against sequential transfer.
- **Only afterwards:** choose architectures for expensive pretrained/foundation-encoder
  work, and build the final ensemble from out-of-fold error complementarity.
