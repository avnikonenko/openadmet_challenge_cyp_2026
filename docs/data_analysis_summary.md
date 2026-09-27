# CYP challenge data and chemical-space analysis

This is a compact summary of the pre-model analysis performed on the September 2026
OpenADMET CYP data snapshot. It uses the supplied training labels and blinded-test
**structures only**; no test activities were available or inferred. The detailed
[analysis report](../analysis/cyp_pre_model/report.md) contains the methods and full
results; [run_all.py](../analysis/cyp_pre_model/scripts/run_all.py) regenerates the
outputs. Source challenge CSVs and bulk generated tables/figures are intentionally
not stored in this Git repository.

## Main findings

| Question | Result | Modelling consequence |
| --- | --- | --- |
| Are molecular identities clean? | All 6,147 unique-name training molecules and 750 blinded-test molecules parse with RDKit. No exact canonical train/test overlap or duplicate molecule/CYP/assay measurements was found. One training molecule has multiple fragments; obvious stereoisomer/tautomer groups were reported, not merged. | Join assays by canonical molecule identity, preserve organizer IDs, and audit near-identity variants across folds. |
| How sparse are the direct labels? | Of 4,905 molecules with direct pIC50, 3,596 have one CYP label and only 41 have all four. CYP2C9/CYP3A4 paired pIC50 has Spearman ρ = 0.688, while CYP2D6 correlations with the other CYPs are near zero. | Use masked multitask losses and CYP-specific outputs; keep CYP2D6-specific controls. |
| Is the single-concentration assay useful? | Its Spearman correlation with direct pIC50 ranges from −0.828 to −0.936 across CYPs (the sign follows the assay readout). | It is a justified auxiliary pretraining target, provided outer-validation molecules are excluded from *all* training assays. |
| Does 2D similarity imply smooth activity? | For CYP3A4, 54 of 260 pairs with Morgan radius-3 (ECFP6) Tanimoto ≥0.6 differ by at least 1 pIC50 unit; 15 differ by at least 2. | Retain activity-cliff error analysis; nearest-neighbor similarity is an applicability diagnostic, not a predictor. |
| How novel is the blind set? | Only 187/750 test molecules share a Bemis–Murcko scaffold with the training union. Test-to-CYP-training median ECFP6 nearest-neighbor similarity is lowest for CYP2D6 (0.394) and highest for CYP3A4 (0.466). Test molecules also have lower Fsp3 (standardized mean difference −0.435) and lower mean MW by about 22 Da. | Report domain-shift and per-CYP coverage, especially for CYP2D6. |
| Do descriptors find the same neighbors? | Among 750 test queries, ECFP6's nearest-neighbor identity matches USR for 1.6%, USRCAT for 10.0%, and pmapper count for 22.5%. | Representations capture different geometry/pharmacophore relationships, but low agreement alone does not establish predictive value. |
| Are 3D descriptors ready for modelling? | On a declared 40-query, five-conformer subset, nearest-neighbor identity changed across conformers for 95.0% of USR, 72.5% of USRCAT, and 70.0% of pmapper-count queries. Three training structures saturated all 2,048 hashed pmapper bits and were chosen as nearest neighbors for 53/750 test queries. | Do not use single-conformer USR/USRCAT or hashed pmapper as primary inputs; a 3D model needs a validated multi-conformer protocol. |

## Validation choice and limits

The primary split groups molecules by connected components of an ECFP6 similarity
graph at Tanimoto 0.6; five saved folds prevent pairs above that threshold from
crossing a fold boundary. Bemis–Murcko scaffold folds provide a separate novelty
stress test. Random folds are an interpolation control, not the main model-selection
criterion. The splits are chemically challenging: the median nearest ECFP6 match
from ECFP6-cluster validation to its training partition is 0.378, whereas the blind
test median against that same reference is 0.506. Thus CV may be more novel than the
test set; neither split perfectly simulates the blind distribution.

The descriptor comparison uses **nearest-neighbor identities and similarity ranks**,
not raw similarity magnitudes across unlike representations. Its multi-conformer
experiment is a manageable subset, not an exhaustive test of every conformer or
train/test pair. The analysis led to the radius-3 Morgan LightGBM baseline, masked
multitask and fold-safe transfer experiments described in the
[method report](method_report.md); it did not use blinded test labels for model
selection.
