# OpenADMET CYP ML and transfer-learning experiment design

> Status: pre-run design draft, 2026-09-11. The live Space was checked on this date:
> it remains in Phase 1 and permits external data and pretrained models. No result in
> this document is a model result until its run is recorded in the experiment ledger.

## Decision to be made

Determine whether transfer learning improves leakage-resistant prediction of:

1. four direct-inhibition pIC50 endpoints; and
2. CYP2D6/CYP3A4 time-dependent inhibition (TDI) classes,

relative to matched models trained from scratch, and whether any improvement adds
out-of-fold value beyond a radius-3 Morgan plus LightGBM baseline.

The phrase *transfer learning* is intentionally split into three experiments:

- **internal supervised transfer:** pretrain on the challenge's auxiliary assays,
  then fine-tune on the scored targets;
- **generic molecular pretraining:** initialize a graph encoder from a public
  molecular foundation checkpoint, then fine-tune it; and
- **external CYP transfer:** pretrain on curated public human-CYP assays, then
  fine-tune on the challenge data.

These sources must not be pooled into one experiment because their leakage risk,
assay match, cost, and scientific interpretation differ.

## Assumptions

- Both current activity tracks are in scope. The dormant structure track is not.
- The fixed primary 2D fingerprint is Morgan radius 3 (ECFP6), as requested.
- Model selection uses only training data and frozen grouped folds. Test structures
  may be featurized for inference and domain-shift diagnostics, but not for fitting
  scalers, selecting models, tuning thresholds, or setting ensemble weights.
- Public data and pretrained models are allowed by the live challenge rules. Every
  external dataset and checkpoint will nevertheless be versioned and disclosed.
- The first implementation should use only supplied challenge data. External CYP
  curation begins only after the internal-transfer result is known.
- PyTorch and graph-model dependencies are proposed here, not installed by this
  design document. They will be pinned in a separate environment before training.
- The existing single-conformer USRCAT and pmapper results are diagnostic. They are
  not primary model inputs until a multi-conformer feature protocol is validated.

## Hypotheses

| ID | Predeclared hypothesis |
| --- | --- |
| H1 | Morgan radius-3 plus RDKit 2D features with LightGBM is a strong low-variance baseline. |
| H2 | Internal auxiliary-assay pretraining improves the PyTorch model over the identical encoder trained from scratch, especially for sparsely labeled endpoints. |
| H3 | A pretrained graph encoder improves novel-scaffold performance more than close-analog performance. |
| H4 | Curated external human-CYP data helps only after assay-aware pretraining and challenge-specific fine-tuning; naïve row pooling may hurt. |
| H5 | USRCAT or unhashed pmapper features are useful only if they improve grouped out-of-fold predictions beyond the 2D/graph ensemble. |

## Data table and target roles

Build one row per canonical training molecule by joining sources on
`Molecule_Name`. Preserve the original names and SMILES separately for submission.
Never concatenate the assay files as independent molecule rows.

### Scored targets

- Regression: four `CYP*_pIC50_direct_inhibition` targets.
- Classification: `CYP2D6_is_TDI` and `CYP3A4_is_TDI`.

Missing labels remain missing. They never become inactive values or negative
classes.

### Internal pretraining targets

Use these as a sparse, masked multitask vector:

- four TDI-condition pIC50 values;
- four direct-condition and four TDI-condition Emax values; and
- four single-concentration `log2fc_estimate` values, pivoted by enzyme.

For the TDI transfer model, the four direct-inhibition pIC50 values may also be
pretraining targets. CYP1A2/CYP2C9 TDI labels are an optional auxiliary
classification ablation, not part of the first pretraining run because their
positive rates are extremely low.

The scored target being fine-tuned must be absent from its own pretraining objective.
In particular, do not pretrain a TDI model on the identical CYP2D6/CYP3A4 `is_TDI`
labels copied between challenge files. Confidence intervals and standard deviations
are metadata for losses and diagnostics, not extra independent labels.

## Leakage-resistant evaluation

### Frozen outer folds

Create and save one five-fold assignment over the complete 6,147-molecule training
union:

1. Generate 2,048-bit, non-chiral Morgan radius-3 fingerprints.
2. Form connected components using Tanimoto similarity >= 0.50.
3. Assign whole components to folds with a deterministic greedy algorithm that
   balances fold size, per-endpoint measured counts, regression target quantile bins,
   and TDI positive/negative counts.
4. Use stable sorting and a single recorded split seed.

Before accepting the split, report each fold's nearest-training-similarity
distribution. If connected-component chaining creates a component too large for
balanced five-fold assignment, stop and document it; use deterministic Butina
clusters at the same threshold as the predeclared fallback. Do not silently change
the threshold after viewing model scores.

Run a secondary five-fold Bemis-Murcko scaffold split as a robustness check for
finalists only. The fingerprint-component split is primary because it directly
controls the close analogs identified in the similarity audit.

### Inner selection

Within each outer-training partition, reserve deterministic whole groups for early
stopping and hyperparameter/threshold selection. All preprocessing is fitted there:

- descriptor filtering, imputation, and scaling;
- class weights, computed from the current inner-fit partition only;
- feature selection;
- early stopping;
- probability calibration and class thresholds; and
- ensemble weights.

An outer validation molecule must be excluded from **all challenge-derived
pretraining targets**, not only from the final scored head. Otherwise an encoder can
memorize that molecule through Emax, screening, or TDI-condition measurements and
produce an optimistic transfer estimate.

### External-data audit

For each external source, record source version, retrieval query, license, assay
metadata, units, canonical structure, and raw-to-model target conversion. Report
exact structure overlaps and Morgan similarity to every challenge fold and to the
blinded test set.

The primary external-transfer result uses a conservative external set with exact
standardized-structure matches to all challenge molecules removed. An allowed-overlap
sensitivity analysis may be reported separately, but it must never be confused with
the primary generalization estimate.

## Representations

### Primary 2D input

- Morgan radius 3, 2,048 bits, `includeChirality=False`.
- A fixed RDKit 2D descriptor list.
- Replace non-finite descriptor values with missing values; remove constant features
  and fit median imputation/scaling inside each fold.

The 2,048-bit representation matches the existing similarity audit. Sparse Morgan
count fingerprints and `includeChirality=True` are secondary representation
ablations, each tested by changing only that factor.

### Deferred 3D input

- USRCAT: aggregate a deterministic multi-conformer ensemble per molecule.
- Pmapper: use unhashed count descriptors only, with training-fold frequency
  filtering or a similarity/kernel learner.
- Do not use the saturated 2,048-bit pmapper fingerprint or shape-only USR as primary
  features.

All generated plots and persistent model outputs must be written under this project,
for example `outputs/plots/` and `outputs/runs/`; `/tmp` is not an output location.

## Model families

### B0: non-neural controls

1. Training-target mean/prevalence control.
2. Radius-3 Morgan similarity-weighted nearest-neighbor control.
3. LightGBM on Morgan radius-3 plus RDKit 2D inputs, one model per endpoint.

LightGBM is the anchor for judging whether neural and transfer models justify their
additional complexity.

### N0: PyTorch MLP trained from scratch

Use a two-tower network so binary fingerprint inputs and continuous descriptors do
not enter through an arbitrarily shared scale:

```text
Morgan 2048 bits -> Linear(2048, 512) -> LayerNorm -> GELU -> Dropout
RDKit 2D         -> Linear(n_desc, 128) -> LayerNorm -> GELU -> Dropout
concatenate      -> Linear(640, 256) -> GELU -> Dropout
                 -> Linear(256, 128) -> shared representation
                 -> one linear output head per target
```

Initial training defaults:

- dropout 0.15;
- AdamW, learning rate 3e-4, weight decay 1e-4;
- batch size 128, maximum 300 epochs, early-stopping patience 30;
- gradient norm clipped at 1.0;
- seeds 17, 29, and 43 for finalist ensembles; seed 17 for screening runs.

These are starting values, not claims of optimality. Restrict inner-fold tuning to a
small declared grid over learning rate `{1e-3, 3e-4}`, dropout `{0.10, 0.25}`, and
shared width `{128, 256}`. Do not widen the search based on outer-fold results.

Use separate regression and classification networks. Within each network, use masked
multitask loss: calculate each task loss over its observed labels, normalize by its
observed count, and then average active task losses. This prevents the densest target
from dominating.

- Regression control loss: Smooth L1/Huber on fold-standardized targets. Transform
  each target's confidence bounds with the identical fit-partition statistics.
- Regression loss ablation: the soft-threshold absolute error defined by each
  observation's confidence interval, optionally mixed 1:1 with Huber. Select this
  once with inner folds and then freeze it for all transfer comparisons.
- Classification loss: `BCEWithLogitsLoss`, with `pos_weight` computed from the
  current fit partition only.

### N1/N2: PyTorch internal-transfer models

Use the N0 encoder and run two matched transfer variants:

- **N1 frozen transfer:** pretrain the encoder and auxiliary heads; discard the
  heads; freeze the encoder; train new scored-task heads.
- **N2 gradual fine-tuning:** initialize from the same pretrained encoder; train new
  heads with the encoder frozen for 10 epochs; then unfreeze it with encoder learning
  rate 1e-4 and head learning rate 1e-3.

Pretraining uses the internal targets listed above, fold-standardized continuous
targets, masked task-balanced loss, and early stopping on an inner group split.
Compare both variants with N0 using identical outer folds, features, target heads,
fine-tuning budget, batch construction, and seed. Report the additional pretraining
compute separately. This matched comparison is the primary test of transfer learning.

### G0-G2: graph-network transfer comparison

- **G0:** multitask Chemprop MPNN trained from scratch.
- **G1:** the identical MPNN pretrained on internal auxiliary targets, exported as an
  encoder, and reused with newly initialized downstream heads.
- **G2S:** the foundation-compatible Chemprop architecture initialized randomly.
- **G2T:** the same architecture initialized from a pinned public foundation encoder
  such as CheMeleon, first frozen and then fine-tuned in separate runs.

Keep the message-passing architecture compatible between G0 and G1. Reuse only the
pretrained message-passing encoder when the downstream output dimension differs;
do not force incompatible auxiliary heads through checkpoint loading. Missing targets
use masked losses. G2T versus G2S is the generic foundation-transfer comparison,
while G1 versus G0 is the cleaner assay-related transfer test; their results answer
different questions.

An optional feature-model candidate is frozen MoLFormer embeddings plus a small
PyTorch head. It enters only after G2T because its large SMILES encoder increases
runtime and changes both representation and model family. It may be evaluated for
predictive or ensemble value, but it is not evidence for a transfer-learning effect
unless compared with an architecture-matched random-initialization control.

### X1: external human-CYP supervised transfer

This phase is conditional on completion of B0, N0, N2, G0, and G1. Curate public
human CYP1A2/CYP2C9/CYP2D6/CYP3A4 inhibition measurements with explicit assay and
unit filters. Convert IC50 in nM as:

```text
pIC50 = 9 - log10(IC50_nM)
```

Do not average incompatible readouts blindly. Aggregate replicates only after
grouping by standardized structure, CYP, assay type, and compatible conditions;
retain dispersion and source counts. Pretrain an MPNN on this data and fine-tune on
challenge targets. Compare against the identical scratch MPNN and internal-transfer
MPNN. Direct row pooling is a negative-control ablation, not the proposed final
method, because public assays are heterogeneous relative to the challenge assay.

## Experiment matrix

| ID | Model | Initialization/source | Inputs | Purpose | Priority |
| --- | --- | --- | --- | --- | --- |
| B0 | Mean/prevalence | None | None | Floor | Required |
| B1 | Similarity kNN | None | Morgan r3 | Series baseline | Required |
| B2 | LightGBM | Scratch | Morgan r3 + RDKit 2D | Main classical baseline | Required |
| N0 | PyTorch MLP | Scratch | Morgan r3 + RDKit 2D | Matched neural control | Required |
| N1 | PyTorch MLP | Internal, frozen | Same as N0 | Representation-transfer test | Required |
| N2 | PyTorch MLP | Internal, gradual unfreeze | Same as N0 | Primary PyTorch transfer test | Required |
| G0 | Chemprop MPNN | Scratch | Molecular graph | Graph control | Required |
| G1 | Chemprop MPNN | Internal auxiliary assays | Molecular graph | Assay-related graph transfer | Required |
| G2S | Chemprop MPNN | Random foundation-compatible encoder | Molecular graph | Matched foundation control | Recommended |
| G2T | Chemprop MPNN | Pinned public foundation encoder | Molecular graph | Generic pretraining | Recommended |
| X1 | Chemprop MPNN | Curated public CYP assays | Molecular graph | External supervised transfer | Conditional |
| D1 | Best finalist + USRCAT | Winner checkpoint | 2D/graph + multi-conformer 3D | 3D incremental value | Conditional |
| D2 | Sparse pmapper learner | Scratch | Unhashed pmapper counts | Pharmacophore complement | Conditional |

Run the required rows before conditional descriptor or external-data work. This
ordering prevents a large search from consuming the pre-interim window without a
validated baseline.

## Metrics and comparisons

### Regression

- Primary: exact official MA-ST-RAE, lower is better.
- Report per-isoform ST-RAE, MAE, R2, Spearman rho, and Kendall tau.
- Report performance by validation-to-training Morgan similarity band and scaffold
  novelty.

### Classification

- Primary: macro MCC across CYP2D6/CYP3A4, higher is better.
- Report per-isoform MCC, precision, recall, F1, and class prevalence.
- For unbiased outer-fold reporting, select each probability threshold using only
  that fold's inner data. After model selection, derive deployment thresholds from
  cross-fitted out-of-fold probabilities over all training molecules.

### Transfer decision rule

Every transfer claim is a paired comparison against the same architecture trained
from scratch. Report fold-wise deltas and a molecule-group bootstrap confidence
interval. Retain a transfer model for ensembling only when:

- the mean primary metric improves;
- at least three of five outer folds improve;
- no scored endpoint shows a material unexplained regression; and
- the gain remains directionally consistent on the secondary scaffold split.

As a practical materiality flag, highlight regression improvements of at least 0.02
absolute MA-ST-RAE and classification improvements of at least 0.02 absolute
macro-MCC. Smaller gains may still enter a low-weight ensemble if cross-fitted
predictions are complementary, but should not be described as stand-alone wins.

## Ensembling

Start with the unweighted mean of B2 and the best neural finalist. Optimize
non-negative weights summing to one using only inner/cross-fitted predictions and the
official primary metric. Limit the final ensemble to three materially distinct
models; adding many seeds or near-duplicate models increases overfitting risk.

For regression, average continuous predictions. For classification, average
probabilities and apply the preselected per-isoform deployment thresholds only after
averaging. Do not majority-vote already thresholded labels during model selection.

## Run records and durable outputs

Each run must write under `outputs/runs/<run_id>/`:

- resolved configuration and command;
- dataset and external-data hashes;
- fold assignment hash;
- environment/package lock and model/checkpoint identifiers plus hashes;
- seeds and deterministic-mode status;
- per-fold and aggregate metrics;
- out-of-fold predictions with identifiers;
- test predictions only for frozen finalist runs;
- training curves, calibration/threshold curves, and similarity-stratified plots;
- warnings, skipped molecules, and non-finite feature counts.

Maintain one append-only `outputs/experiment_ledger.csv` with model ID, run ID,
code revision, data/fold hashes, status, primary metrics, and artifact path. Stable
sorting must be used before hashing or serializing tabular outputs.

## Execution stages

1. **Freeze data/folds:** create the molecule table, target masks, component folds,
   and leakage audit.
2. **Classical baseline:** run B0-B2 and verify the official scorers end to end.
3. **PyTorch:** run N0, N1, and N2; review the transfer delta before adding models.
4. **Graph models:** run G0 and G1, then matched G2S/G2T if resources permit.
5. **External transfer:** curate and run X1 only if internal transfer or G0 indicates
   graph capacity is useful.
6. **Deferred descriptors:** test D1/D2 only against frozen finalist predictions.
7. **Ensemble and freeze:** select from cross-fitted predictions, retrain on all
   eligible training molecules, validate the serialized submission, and record its
   hash.

## Stop/go gates

- Stop a neural family if it cannot beat B2 after the declared tuning budget and
  three seeds, unless it provides a reproducible ensemble gain.
- Stop internal transfer if both frozen and gradual variants fail to improve their
  matched scratch model on at least three outer folds.
- Do not start external CYP transfer until its provenance and overlap audit is
  reproducible.
- Do not add 3D features until a deterministic multi-conformer protocol succeeds on
  at least 99% of molecules or defines and validates a missing-feature fallback.
- Do not use the live leaderboard to choose among minor hyperparameters. The
  one-time interim result is an external check of already frozen candidates.

## Risks and validation checks

| Risk | Control |
| --- | --- |
| Auxiliary-label leakage across folds | Exclude every outer validation molecule from all challenge pretraining stages. |
| Comparing transfer with a weaker scratch control | Match architecture, features, folds, seed, and fine-tuning budget; report pretraining compute separately. |
| Assay mismatch in public CYP data | Pretrain then fine-tune; keep assay metadata; compare with a pooled negative control. |
| Sparse tasks dominated by dense tasks | Mask missing labels and average normalized per-task losses. |
| TDI class imbalance | Fold-local `pos_weight`; nested threshold selection by MCC. |
| Fingerprint collision or stereo loss | Keep 2,048-bit non-chiral input as the anchor; run count/chiral variants separately. |
| Foundation-model contamination | Record pretraining corpus/version and audit exact challenge structures where feasible. |
| Non-deterministic GPU training | Fixed seeds, deterministic operations where supported, three-seed finalist report. |
| Descriptor-driven complexity without gain | Require paired grouped-CV and ensemble improvement before retention. |

## Acceptance criteria for the experiment design

- [ ] Fold-generation logic and the fold file are frozen before model tuning.
- [ ] B2, N0, N1, N2, G0, and G1 have matched five-fold out-of-fold predictions.
- [ ] Every transfer result has an architecture-matched scratch control.
- [ ] No outer-fold molecule appears in internal pretraining for that fold.
- [ ] Thresholds, preprocessing, and ensemble weights are learned without outer-fold
      or test leakage.
- [ ] All plots and run artifacts are durable and stored inside this workspace.
- [ ] External sources and pretrained checkpoints have version, license, and hashes.
- [ ] Finalists are reviewed by endpoint, scaffold novelty, and Morgan-similarity
      band, not only by a macro average.
- [ ] The frozen submission passes the official validator after serialization and
      reload.

## References

- [Live challenge Space](https://huggingface.co/spaces/openadmet/cyp-challenge)
- [Current live Space configuration](https://huggingface.co/spaces/openadmet/cyp-challenge/blob/main/config.py)
- [Official challenge tutorial](https://github.com/OpenADMET/CYP-Challenge-Tutorial)
- [Chemprop training, missing targets, and transfer learning](https://chemprop.readthedocs.io/en/main/tutorial/cli/train.html)
- [MoLFormer model card](https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct)
- [PyTorch `BCEWithLogitsLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
- [PyTorch `HuberLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html)
- [RDKit Morgan fingerprint API](https://www.rdkit.org/docs/cppapi/namespaceRDKit_1_1MorganFingerprint.html)
