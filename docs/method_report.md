# OpenADMET CYP challenge: direct-inhibition method report

This entry predicts direct-inhibition pIC50 for CYP1A2, CYP2C9, CYP2D6, and CYP3A4
from molecular structure. It addresses the regression track only; no TDI predictions
are included. Code and configurations are available in the
[project repository](https://github.com/avnikonenko/openadmet_challenge_cyp_2026).
Challenge data are obtained separately from the
[organizers' dataset](https://huggingface.co/datasets/openadmet/cyp-challenge-train-test)
and are not redistributed here.

## Data, models, and validation

RDKit-canonicalized molecules and saved molecule identities are used consistently.
The labelled direct-inhibition set contains 4,905 molecules and 6,525 observed
molecule–CYP values; missing CYP labels are masked. The final predictor is an
unweighted mean of three model families, each run on five folds with seeds 1, 2,
and 3 (45 family/fold/seed prediction sets; each LightGBM run fits four per-CYP regressors):

1. A per-CYP LightGBM model using 2,048-bit Morgan radius-3 fingerprints and eight
   RDKit physicochemical descriptors ([configuration](../configs/lightgbm.yaml)).
2. A project-native, Chemprop-style directed message-passing network (D-MPNN) with
   a shared molecular encoder and CYP-specific heads. Its encoder has hidden width
   300, depth 3, and dropout 0.10. It is first trained on single-concentration CYP
   inhibition, then loaded with new pIC50 heads and fully fine-tuned using a lower
   encoder learning rate (1e-4) than head learning rate (1e-3). This is **not** an
   externally pretrained or official Chemprop model
   ([configuration](../configs/transfer_singleconc.yaml)).
3. A joint multi-assay D-MPNN with learned CYP and assay embeddings, trained on
   single-concentration inhibition and direct pIC50 together. Only direct pIC50
   outputs contribute to the reported challenge score. It uses encoder width 600,
   depth 4, and 32- and 16-dimensional CYP and assay embeddings
   ([configuration](../configs/joint_multiassay.yaml)).

Neural runs use training-fold-only per-CYP target standardization and inverse-transform
predictions before evaluation. Fine-tuning and joint training use an interval-aware
loss for pIC50; the original transfer configuration also names this loss for
single-concentration pretraining, where absent interval bounds make it equivalent to
ordinary squared error. Validation molecules are excluded from both direct and
auxiliary-assay training in each fold. The blinded test set is used only for inference.
No external molecular pretraining data or pretrained weights were used.

Model selection used fixed five-fold ECFP6-cluster CV, three seeds per fold, with a
separate five-fold Bemis–Murcko scaffold CV stress test. The primary metric is the
organizers' macro soft-threshold relative absolute error (MA-ST-RAE): per-CYP error
outside experimental credible intervals, normalized by the mean-prediction baseline,
then averaged over the four CYPs. Lower is better. Ensemble membership and equal
weights were selected from out-of-fold (OOF) predictions, never test labels.

## Cross-validation results

| CV split | Transfer D-MPNN | Joint multi-assay | LightGBM | Equal-weight ensemble |
| --- | ---: | ---: | ---: | ---: |
| ECFP6 cluster | 0.6792 | 0.6757 | 0.7449 | **0.6500** |
| Bemis–Murcko scaffold | 0.6790 | 0.6694 | 0.7478 | **0.6482** |

These are OOF macro ST-RAE values after averaging seeds within each family; they are
**not blinded-test scores**. The ensemble improved over transfer alone in all five
folds of both split schemes. On the primary ECFP6-cluster split, ensemble ST-RAE by
endpoint was 0.7590 (CYP1A2), 0.5538 (CYP2C9), 0.8418 (CYP2D6), and 0.4455
(CYP3A4). CYP2D6 remained the hardest endpoint. Its small improvement over transfer
alone was not conclusive in a molecule-level bootstrap. Ensemble composition was
chosen using these OOF data, so the reported score can still be optimistic from
model selection; the two CV schemes also reuse the same molecules. Additional
details are in the [ensemble validation report](../analysis/ensemble_validation/report.md).

## Final artifact and reproducibility

For each blinded molecule, predictions from all 45 run-level prediction sets were averaged with
equal weight (equivalently, one third per model family). The resulting regression CSV
has 750 unique organizer-provided molecule IDs and 3,000 finite pIC50 predictions.
It passed the project's serialized-file checks and the organizers' official
submission validator (tutorial revision `832ae6d`). The exact validated CSV has
SHA-256 `9c239bf48d34ef659d39974e3ac05ca67f94c5fb3ccd3df9f6a8a7e20644b3d6`.
**It has not been uploaded to the challenge platform, and no
blind-test performance is claimed here.** The CSV and model checkpoints are excluded
from this Git repository. Each run retains its exact config, seed, fold, input hashes,
code revision, checkpoint provenance, and prediction-integrity audit. The
[cluster runbook](cluster.md) and [quick start](quickstart.md) describe execution and
validation; the [experiment guide](experiments.md) records the broader comparisons.
