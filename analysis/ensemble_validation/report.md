# Three-family CYP ensemble validation

Analysis run: 2026-09-27. Input: saved out-of-fold predictions from the completed
five-fold, three-seed ECFP-cluster and scaffold experiments. There are 4,905 labelled
molecules and 6,525 molecule–CYP observations per split. The ensemble equally averages
seed-averaged predictions from single-concentration transfer D-MPNN, joint multi-assay
D-MPNN, and LightGBM. Lower ST-RAE is better. The ECFP result reproduces the saved
ensemble value to numerical precision (0.650020).

## Main result

| Split | Transfer, seed-averaged | Joint, seed-averaged | LightGBM, seed-averaged | Three-family ensemble | Ensemble minus transfer |
| --- | ---: | ---: | ---: | ---: | ---: |
| ECFP cluster | 0.6792 | 0.6757 | 0.7449 | **0.6500** | −0.0292 |
| Scaffold | 0.6790 | 0.6694 | 0.7478 | **0.6482** | −0.0309 |

The ensemble beats the seed-averaged transfer model on **all five folds in both split
schemes**. Fold-level gains range from 0.0190 to 0.0391 ST-RAE for ECFP and 0.0230 to
0.0378 for scaffold. In a paired bootstrap resampling molecule IDs (all CYP labels
of a molecule move together; 1,000 fixed-seed replicates), the 95% percentile intervals
for ensemble-minus-transfer macro ST-RAE are **[−0.0368, −0.0218]** for ECFP and
**[−0.0388, −0.0230]** for scaffold.

## Endpoints

| Split | CYP | Transfer | Ensemble | Difference | Bootstrap 95% interval for difference |
| --- | --- | ---: | ---: | ---: | ---: |
| ECFP | CYP1A2 | 0.7895 | 0.7590 | −0.0305 | [−0.0450, −0.0162] |
| ECFP | CYP2C9 | 0.5876 | 0.5538 | −0.0338 | [−0.0540, −0.0156] |
| ECFP | CYP2D6 | 0.8510 | 0.8418 | −0.0092 | [−0.0208, 0.0025] |
| ECFP | CYP3A4 | 0.4885 | 0.4455 | −0.0430 | [−0.0552, −0.0311] |
| Scaffold | CYP1A2 | 0.7825 | 0.7543 | −0.0282 | [−0.0422, −0.0136] |
| Scaffold | CYP2C9 | 0.5817 | 0.5388 | −0.0429 | [−0.0631, −0.0234] |
| Scaffold | CYP2D6 | 0.8504 | 0.8438 | −0.0066 | [−0.0190, 0.0059] |
| Scaffold | CYP3A4 | 0.5014 | 0.4557 | −0.0457 | [−0.0589, −0.0336] |

CYP2D6 remains the hardest endpoint. Its point estimate improves on both splits, but
its bootstrap interval includes no improvement. The ensemble's clearest gains are
for CYP2C9 and CYP3A4.

## Chemical novelty and difficult examples

Using ECFP6 similarity to the saved direct-pIC50 training partition for each fold,
**3,786/6,525** labelled ECFP observations and **3,835/6,525** scaffold observations
are below 0.4.
Ensemble MAE improves over transfer in both the low and moderate similarity groups:
ECFP 0.5754→0.5544 (<0.4) and 0.5619→0.5385 (0.4–<0.7); scaffold
0.5802→0.5593 and 0.5618→0.5368, respectively. There are **zero** ECFP observations
and only **21** scaffold observations with similarity ≥0.7, so these splits cannot
establish performance for close analogs. All scaffold validation molecules have unseen
Bemis–Murcko scaffolds by construction; ECFP has 472/6,525 observations on seen
scaffolds. Counts here are molecule–CYP observations, not unique molecules.
The transfer and joint models may additionally encounter auxiliary-assay molecules
during training; the similarity diagnostic consistently uses the direct-pIC50
partition shared with LightGBM.

The prior activity-cliff report contains selected pairs with ECFP6 similarity ≥0.6
and |ΔpIC50| ≥1. Among this selected set, 97 labelled OOF observations are flagged.
Their ensemble MAE is 0.6791 on ECFP and 0.6907 on scaffold, compared with 0.5458
and 0.5477 outside the selected examples. This is **illustrative, not a prevalence
estimate**: the list contains only highlighted pairs, and pair membership does not
establish that its analog was in that fold's training partition. The per-CYP
`ensemble_top_errors.csv` identifies individual large errors for follow-up chemistry
review.

## Interpretation and limits

The equal-weight three-family ensemble is supported by both CV schemes and needs no
additional model training before submission preparation. The 0.6500 ECFP score is an
OOF estimate; it is **not** a blind-test result. Ensemble membership was chosen using
these OOF results, so the reported score and fixed-model bootstrap intervals do not
account for model-selection bias. The two CV schemes reuse the same molecules and are
not independent test sets. The bootstrap also holds fitted models and seeds fixed; it
measures molecule-sampling variation, not training-run variability.

The next gate is to run the organizers' submission validator on the existing regression
submission and prepare the required method report. Any new modelling should target a
specific failure mode, especially CYP2D6, rather than launch another broad search.

Reproduce with the command in [README.md](README.md). The key machine-readable
outputs are `tables/ensemble_metrics.csv`, `tables/ensemble_fold_metrics.csv`,
`tables/ensemble_bootstrap_ci.csv`, `tables/ensemble_error_by_domain.csv`, and
`tables/ensemble_top_errors.csv`. Exact fold-specific similarities are in
`tables/cv_direct_training_applicability.csv`. The figures are `figures/ensemble_by_split.png`
and `figures/error_by_similarity.png`.
