# OpenADMET CYP Inhibition Blind Challenge — task brief

> Status: active. Submissions opened 2026-08-17 and close 2026-11-03 at 23:59 UTC.
>
> Last verified: 2026-08-24 against the official challenge Space, launch posts,
> dataset, and tutorial/evaluation code.

## Assumptions and scope

- “Current OpenADMET” means the **OpenADMET CYP Inhibition Blind Challenge**, as
  identified by the dataset checked out in this workspace.
- This brief covers both currently active, independent tracks. We can enter either
  track, but the working goal is to build reproducible entries for both because they
  use the same 750 test molecules and can share representations.
- The live [challenge Space](https://huggingface.co/spaces/openadmet/cyp-challenge)
  and its announcements take precedence over this snapshot if the organizers change
  a rule. Recheck the Space and `#cyp-challenge` Discord channel before every milestone
  submission.
- The source contains a disabled, unfinished structure-track placeholder
  (`STRUCTURE_TRACK_LIVE = False`, with a provisional dataset size). It is **not a
  current track** and is outside scope unless the organizers formally activate it and
  release final instructions.
- This document defines the task and delivery gates; it does not choose a final model
  before baselines and leakage-resistant validation have been run.

## Desired outcome

Produce one or both of the following, together with a reproducible method report:

1. A valid direct-inhibition regression submission with four pIC50 predictions for
   every test compound.
2. A valid time-dependent inhibition (TDI) classification submission with hard
   boolean predictions for CYP2D6 and CYP3A4 for every test compound.

Model selection must use deterministic, chemistry-aware validation and the official
primary metrics. The final artifacts must preserve the test identifiers and pass the
organizers' local validation code before upload.

## Challenge summary

| Item | Current requirement |
| --- | --- |
| Organizers | OpenADMET; experimental data generated at Octant/UCSF |
| Test set | 750 blinded compounds; `Molecule_Name` and `SMILES` only |
| Direct-inhibition targets | CYP1A2, CYP2C9, CYP2D6, CYP3A4 pIC50 |
| TDI targets | CYP2D6 and CYP3A4 boolean labels |
| Regression primary score | Macro-averaged Soft-Threshold Relative Absolute Error (MA-ST-RAE), lower is better |
| TDI primary score | Macro-averaged MCC across CYP2D6/CYP3A4 on the overall leaderboard, higher is better |
| Live leaderboard | Half of the test set, partitioned by chemisimilar series |
| Interim reveal | Full-test-set performance released once on 2026-09-25 |
| Final leaderboard | Full test set after submissions close |
| Submission rate | At most one upload per track every 12 hours; latest valid upload counts |
| Data license | Apache-2.0 on the dataset card |

## Scientific task

Cytochrome P450 enzymes metabolize many small-molecule drugs. Inhibiting them can
slow clearance of co-administered drugs and create drug-drug interaction risk.

The challenge uses two biochemical preincubation arms:

- **Direct inhibition (`-NADPH`)**: metabolism-dependent inactivation cannot occur;
  observed inhibition primarily reflects direct binding by the parent compound.
- **TDI condition (`+NADPH`)**: catalytic turnover can produce metabolites or
  intermediates that increase inhibition, including mechanism-based inhibition.

The target potency is

```text
pIC50 = -log10(IC50 in mol/L)
```

Higher pIC50 therefore means stronger inhibition. Values below pIC50 4 correspond to
IC50 above 100 µM and are outside the assay's reliable lower activity range.

The fluorescence assay is used for CYP1A2, CYP2C9, and CYP3A4. CYP2D6 uses a
label-free Echo-MS parent-depletion assay with dextromethorphan because the tested
fluorogenic probes did not provide adequate signal quality.

## Track 1: direct inhibition regression

Predict these four continuous targets for all 750 test molecules:

- `CYP1A2_pIC50_direct_inhibition`
- `CYP2C9_pIC50_direct_inhibition`
- `CYP2D6_pIC50_direct_inhibition`
- `CYP3A4_pIC50_direct_inhibition`

### Primary metric

For endpoint ground truth `y_i`, its fitted 95% credible interval `[L_i, U_i]`, and
prediction `p_i`, the soft-threshold absolute error is:

```text
e_i(p_i) = max(p_i - U_i, L_i - p_i, 0)
```

A prediction inside the interval has zero error. For each endpoint, the official
ST-RAE implementation is:

```text
ST-RAE = sum_i e_i(p_i) / sum_i e_i(mean(y))
```

The constant mean baseline is passed through the same interval rule. Thus 0 is ideal,
1 is equivalent to the mean baseline under this loss, and values below 1 beat that
baseline. The overall MA-ST-RAE is the plain arithmetic mean of the four endpoint
scores for each bootstrap sample. This interval-aware loss—not an arbitrary clipping
of submitted predictions—is the current primary scorer.

Secondary metrics are MAE, R², Spearman's rho, and Kendall's tau. The evaluation code
uses 1,000 bootstrap resamples and exposes both overall and per-isoform results.

## Track 2: TDI classification

Submit a hard `True`/`False` call for each of:

- `CYP2D6_is_TDI`
- `CYP3A4_is_TDI`

The primary endpoint metric is Matthews correlation coefficient (MCC), which is
appropriate for the roughly 21% positive training labels. The current scoring code
also calculates Accuracy, Precision, Recall, and F1. The overall classification
leaderboard macro-averages each metric across CYP2D6 and CYP3A4.

Although predictions are required for all 750 molecules, the organizers state that
only compounds with confidently assigned ground-truth labels contribute to scoring.
This prevents the scored subset itself from leaking information about regression
activity.

### Official TDI label semantics

Define the shift on the pIC50 scale as:

```text
shift = pIC50_TDI_condition - pIC50_direct_inhibition
log10(2) = 0.301
```

The official categories are:

| Direct arm | TDI arm / shift | Category | Scored class |
| --- | --- | --- | --- |
| pIC50 > 4 | shift > 0.301 | Positive | Positive |
| pIC50 > 4 | shift <= 0.301 | Negative | Negative |
| pIC50 < 4 | TDI pIC50 > 4.301 | Inferred positive | Positive |
| pIC50 < 4 | TDI pIC50 < 4 | Assigned negative | Negative |
| pIC50 < 4 | TDI pIC50 from 4 through 4.301, or insufficient data | Not confidently assignable | Excluded from scoring |

Use the supplied `is_TDI` columns as labels. Do not silently rederive them with
slightly different boundary or missing-value rules.

## Local data inventory

All paths below are relative to this file. The counts are from the checked-out CSVs,
not estimates from the dataset card.

| File | Shape | Role |
| --- | ---: | --- |
| `cyp-challenge-train-test/cyp-challenge-TRAIN_inhibition.csv` | 4,905 × 18 | Sparse direct-inhibition pIC50, 95% interval bounds, and standard deviations |
| `cyp-challenge-train-test/cyp-challenge-TRAIN_TDI.csv` | 6,145 × 36 | Sparse paired direct/TDI-condition pIC50 data; CYP2D6/CYP3A4 TDI labels |
| `cyp-challenge-train-test/cyp-challenge-TRAIN_Emax.csv` | 6,146 × 30 | Direct/TDI-condition Emax values and interval bounds; auxiliary TDI labels for all four CYPs |
| `cyp-challenge-train-test/cyp-challenge-single-concentration-TRAIN.csv` | 17,504 × 12 | 4,376 compounds × 4 CYPs at 49.5049505 µM; primary-screen signals |
| `cyp-challenge-train-test/cyp-challenge-TEST-BLINDED.csv` | 750 × 2 | Unique test `Molecule_Name` and `SMILES`; no labels |

### Direct-inhibition label coverage

The training matrix is deliberately sparse because compounds were promoted from the
primary screen to dose-response follow-up only for selected isoforms.

| Target | Labeled | Median | Range | pIC50 < 4 |
| --- | ---: | ---: | ---: | ---: |
| CYP1A2 | 1,412 | 5.133 | 1.906–7.949 | 232 (16.4%) |
| CYP2C9 | 1,285 | 4.621 | 2.098–7.473 | 259 (20.2%) |
| CYP2D6 | 1,493 | 4.726 | 1.947–7.535 | 129 (8.6%) |
| CYP3A4 | 2,335 | 4.266 | 1.909–7.187 | 944 (40.4%) |

Of the 4,905 rows, 3,596 have one measured endpoint, 1,039 have two, 229 have three,
and only 41 have all four. Missing endpoint values are unmeasured, not negative labels.

### TDI label coverage and imbalance

| Target | Labeled | Positive | Negative | Positive rate |
| --- | ---: | ---: | ---: | ---: |
| CYP2D6 | 1,497 | 324 | 1,173 | 21.6% |
| CYP3A4 | 3,584 | 764 | 2,820 | 21.3% |

The Emax file also supplies auxiliary `is_TDI` labels for CYP1A2 (23/1,414 positive,
1.6%) and CYP2C9 (38/1,285 positive, 3.0%), but those two endpoints are not scored in
the TDI challenge track.

### Primary-screen interpretation

The single-concentration screen was run in the TDI condition. It provides
`log2fc_estimate`, `log2fc_fdr`, uncertainty/statistical columns, assay plate/batch,
and enzyme identity. Inhibition reduces probe turnover, so a more negative fold
change indicates stronger inhibition. The tutorial defines a screening hit as:

```text
log2fc_estimate < -1 and log2fc_fdr < 0.05
```

This signal is useful as an auxiliary feature/target; it is not a requested test
output and should not be confused with direct-inhibition pIC50.

### Join and integrity constraints

- Join all training sources by `Molecule_Name`; never assume equal row counts or row
  order.
- The regression rows are all present in the TDI file, and their direct-inhibition
  pIC50 values/missingness match exactly in this snapshot.
- The Emax and TDI files share 6,145 names and matching CYP2D6/CYP3A4 TDI labels;
  `OCNT-2329004` occurs only in Emax.
- `OCNT-2328658` occurs in the single-concentration data but not the TDI data.
- The 750 test names and SMILES are unique. They have no exact name or exact SMILES
  overlap with the supplied training files.
- Canonicalize molecules only for modeling and duplicate/leakage analysis. Preserve
  the organizer-supplied `Molecule_Name` and `SMILES` verbatim in submissions.

## Train/test chemical-similarity audit

This audit treats “training” as the union by `Molecule_Name` of the regression, TDI,
Emax, and single-concentration files: 6,147 unique training molecules versus 750 test
molecules. “Within train” and “within test” exclude self-comparisons. The analysis
environment is `openadmet-cyp-sim` with Python 3.11.16, RDKit 2026.03.5, NumPy 2.4.6,
pandas 3.0.5, pmapper 1.1.3, and NetworkX 3.6.1.

### 2D method and results

Following the requested fingerprint definition, 2D similarity is Tanimoto similarity
between 2,048-bit Morgan fingerprints with radius 3 (ECFP6), no chirality, and the
as-supplied molecular graph. All 4,610,250 test–train pairs, all 18,889,731 unique
train–train pairs, and all 280,875 unique test–test pairs were evaluated exactly.

| Comparison | Mean | Median | P05 | P95 | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| All test–train pairs | 0.104 | 0.102 | 0.059 | 0.157 | 0.919 |
| All within-train pairs | 0.103 | 0.101 | 0.058 | 0.156 | 1.000 |
| All within-test pairs | 0.115 | 0.107 | 0.062 | 0.178 | 1.000 |
| Test-to-train nearest neighbor | 0.531 | 0.523 | 0.420 | 0.677 | 0.919 |
| Within-train nearest neighbor | 0.443 | 0.409 | 0.244 | 0.775 | 1.000 |
| Within-test nearest neighbor | 0.601 | 0.605 | 0.409 | 0.780 | 1.000 |

Test-to-train nearest-neighbor coverage is:

| ECFP6 threshold | Test molecules | Fraction |
| --- | ---: | ---: |
| >= 0.30 | 749 | 99.9% |
| >= 0.40 | 730 | 97.3% |
| >= 0.50 | 468 | 62.4% |
| >= 0.60 | 145 | 19.3% |
| >= 0.70 | 15 | 2.0% |
| >= 0.80 | 3 | 0.4% |
| >= 0.90 | 1 | 0.1% |

The most similar cross-set pair is test `OCNT-2534939` and train `OCNT-2315408`
(0.919). The least-covered test molecule is `OCNT-2535242` (nearest-train similarity
0.234). There is no exact canonical-SMILES or radius-3 fingerprint match between
train and test.

The test set is substantially more series-clustered than the training pool: 75.9% of
test compounds are closer to another test compound than to any training compound.
At similarity >= 0.50, the train graph has 4,513 singletons out of 6,147 molecules,
whereas the test graph has only 135 singletons out of 750; the largest test component
contains 29 molecules. At >= 0.70, the corresponding singleton counts are 5,582 and
604, and the largest test component contains six molecules. This is consistent with
the organizers' analog-expansion design and makes random molecule splits a poor proxy
for the blind test.

Exact Bemis–Murcko scaffold overlap is much lower than the moderate ECFP6 coverage:
only 187/750 test molecules (24.9%) have a scaffold seen in training. The test set has
546 distinct nonempty scaffolds, of which only 66 (12.1%) occur in training. All 750
test molecules have a nonempty cyclic scaffold under this definition.

The non-chiral fingerprint setting intentionally maps enantiomers to the same bits.
The input structures themselves are unique, but this produces five exact fingerprint
pairs in training and two in test; both test pairs are specified enantiomer pairs.
If stereochemical sensitivity is important, `includeChirality=True` should be run as
a separate robustness analysis rather than silently changing this audit.

### Training-source overlap

The training files are mostly different assay views of the same molecules, not
independent chemical libraries:

| Pair | Shared molecule names |
| --- | ---: |
| Regression / TDI | 4,905 |
| Regression / Emax | 4,905 |
| Regression / single concentration | 4,375 |
| TDI / Emax | 6,145 |
| TDI / single concentration | 4,375 |
| Emax / single concentration | 4,375 |

All regression molecules occur in both TDI and Emax; Emax contains all TDI molecules
plus one; and the single-concentration set contributes one additional molecule to the
overall union. The TDI and Emax pools provide the same nearest ECFP6 training molecule
for every test structure. Regression alone provides that same neighbor for 748/750
test structures, and the single-concentration pool does so for 648/750. Consequently,
concatenating these files as independent rows would duplicate compounds and bias a
split; merge their assay endpoints by molecule instead.

### 3D method and results

The 3D audit is diagnostic, not a unique molecular truth. One deterministic ETKDGv3
conformer was generated per molecule from the largest fragment, with explicit
hydrogens during embedding, enforced input chirality, MMFF94 minimization, and heavy
atoms used for descriptors. Standard embedding and a random-coordinate fallback each
had a 10-second cap. Similarity uses RDKit's default USR score (shape only) and USRCAT
score (shape plus atom-type/pharmacophore information).

All test–train and test–test scores and all nearest neighbors were computed exactly.
Only the within-train all-pair distributions use a fixed-seed sample of 2,000,000
ordered, non-self pairs with replacement.

| Descriptor and comparison | Mean | Median | P05 | P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| USR, all test–train pairs | 0.609 | 0.615 | 0.393 | 0.807 |
| USR, sampled within-train pairs | 0.612 | 0.620 | 0.395 | 0.808 |
| USR, all within-test pairs | 0.608 | 0.613 | 0.392 | 0.809 |
| USR, test-to-train nearest neighbor | 0.924 | 0.925 | 0.888 | 0.952 |
| USR, within-train nearest neighbor | 0.922 | 0.925 | 0.882 | 0.953 |
| USR, within-test nearest neighbor | 0.908 | 0.908 | 0.852 | 0.960 |
| USRCAT, all test–train pairs | 0.166 | 0.161 | 0.105 | 0.240 |
| USRCAT, sampled within-train pairs | 0.165 | 0.161 | 0.105 | 0.240 |
| USRCAT, all within-test pairs | 0.169 | 0.164 | 0.105 | 0.249 |
| USRCAT, test-to-train nearest neighbor | 0.383 | 0.372 | 0.293 | 0.512 |
| USRCAT, within-train nearest neighbor | 0.387 | 0.378 | 0.294 | 0.509 |
| USRCAT, within-test nearest neighbor | 0.392 | 0.364 | 0.270 | 0.603 |

USR is highly saturated because global shape alone is permissive; USRCAT is the more
discriminating 3D diagnostic. The ECFP6 and USRCAT nearest-training molecules agree
for only 75/750 tests (10.0%), and their nearest-neighbor similarity ranks are
essentially uncorrelated (Spearman rho = -0.014). This apparent complementarity must
be interpreted cautiously because a flexible molecule is represented by only one
low-energy conformer.

Conformer generation succeeded for 6,144/6,147 training and all 750 test molecules.
`OCNT-0453746` and `OCNT-0453782` exceeded both embedding attempts; the largest
fragment of three-fragment `OCNT-2329004` had fewer than the four heavy atoms required
by USR. MMFF94 reached its iteration limit for 75 retained training conformers and
five retained test conformers. Any model that uses 3D similarity should therefore
test a conformer ensemble and a defined aggregation rule rather than treating these
single-conformer scores as definitive.

### Pmapper pharmacophore-descriptor audit

The requested [DrrDom/pmapper](https://github.com/DrrDom/pmapper) package was installed
as version 1.1.3 in `openadmet-cyp-sim`. The primary representation is the unhashed,
count-based four-feature descriptor returned by `get_descriptors(ncomb=4, tol=0)` at
the default 1 Å bin step. Similarity is generalized Tanimoto:

```text
dot(a, b) / (dot(a, a) + dot(b, b) - dot(a, b))
```

For comparison, a 2,048-bit fingerprint was generated from the same four-feature
pharmacophores with one activated bit per signature and compared by binary Tanimoto.
Both representations use the same single conformers and feature definitions. Empty
versus empty is explicitly assigned similarity zero. All successful test–train,
within-train, and within-test pairs were calculated exactly.

| Count-descriptor comparison | Mean | Median | P05 | P95 | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| All test–train pairs | 0.0044 | 0.0023 | 0.0000 | 0.0161 | 0.646 |
| All within-train pairs | 0.0041 | 0.0021 | 0.0000 | 0.0153 | 0.735 |
| All within-test pairs | 0.0055 | 0.0028 | 0.0000 | 0.0190 | 0.814 |
| Test-to-train nearest neighbor | 0.103 | 0.087 | 0.026 | 0.228 | 0.646 |
| Within-train nearest neighbor | 0.092 | 0.073 | 0.027 | 0.215 | 0.735 |
| Within-test nearest neighbor | 0.144 | 0.105 | 0.024 | 0.411 | 0.814 |

The count descriptor generated 1,767,732 distinct keys across the two sets. Of the 750
tests, 292 (38.9%) have a nearest training molecule at similarity >= 0.10, 58 (7.7%)
at >= 0.20, 17 (2.3%) at >= 0.30, and only three (0.4%) at >= 0.50. The highest
test–train pair is `OCNT-2535748` / `OCNT-2395545` at 0.646. Pmapper's count scale is
much stricter than ECFP6, so numerical thresholds must not be transferred between the
representations. A majority of tests (57.1%) are closer to another test pharmacophore
than to any training pharmacophore, again showing internal test-series structure.

| Hashed-fingerprint comparison | Mean | Median | P05 | P95 | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| All test–train pairs | 0.099 | 0.084 | 0.016 | 0.229 | 0.785 |
| All within-train pairs | 0.103 | 0.086 | 0.016 | 0.245 | 1.000 |
| All within-test pairs | 0.096 | 0.083 | 0.016 | 0.215 | 0.817 |
| Test-to-train nearest neighbor | 0.243 | 0.214 | 0.082 | 0.491 | 0.785 |
| Within-train nearest neighbor | 0.253 | 0.213 | 0.081 | 0.582 | 1.000 |
| Within-test nearest neighbor | 0.263 | 0.232 | 0.074 | 0.553 | 0.817 |

The 2,048-bit fingerprint is not reliable as the primary similarity metric for this
chemically size-diverse set. Three feature-rich training molecules activate all 2,048
bits, and 15 activate at least 90%. Nevertheless, fully saturated compounds are
selected as the nearest neighbor for 53/750 tests, and >=90%-saturated compounds for
201/750. This explains implausible high-scoring pairs between ordinary test compounds
and very large training molecules. The count and bit versions select the same nearest
training molecule for only 28.4% of tests (nearest-score Spearman rho 0.309). Use the
unhashed count descriptor for analysis; retain the bit result only as a collision and
saturation warning.

Pmapper conformers/descriptors succeeded for 6,145/6,147 training molecules and all
750 tests. `OCNT-0453746` and `OCNT-0453782` failed both bounded embedding attempts;
75 retained training conformers and five test conformers reached the MMFF94 iteration
limit. `OCNT-2329004` has one perceived pharmacophore feature and therefore an empty
four-feature bit fingerprint, although pmapper emits one lower-order count descriptor.

Durable outputs are `pmapper_similarity_summary.json`,
`pmapper_per_test_nearest_neighbors.csv`, `pmapper_top_test_train_pairs.csv`,
`pmapper_descriptor_metadata.csv`, `pmapper_similarity_distributions.png`,
`pmapper_descriptor_coverage.png`, and `pmapper_representation_agreement.png`. The
reproducible entry point is `pmapper_similarity_analysis.py`; descriptor caches are
JSON/Gzip rather than pickle.

### Property-domain shift and validation implications

The test molecules are modestly smaller and less saturated than training: mean
molecular weight is 337.5 versus 359.4, mean heavy-atom count is 23.87 versus 25.57,
and mean fraction Csp3 is 0.291 versus 0.365. Mean cLogP is slightly lower (2.83 versus
2.97), while TPSA is similar (71.27 versus 70.23). The fraction-Csp3 shift is the
largest univariate distribution difference (KS statistic 0.247).

For model selection, report both scaffold-group and radius-3 fingerprint-cluster
validation. In addition to fold-level primary metrics, stratify errors by test-like
nearest-training similarity bands (`<0.4`, `0.4–0.5`, `0.5–0.6`, and `>=0.6`) and by
scaffold novelty. A useful validation split should reproduce the blind set's median
nearest-training similarity near 0.52; an extremely remote holdout tests extrapolation
but is not by itself representative of this challenge.

### Reproducibility snapshot

The local dataset checkout is commit
`85f8b358d0a2056a98b990dd75d3b3ec9247862b` (2026-08-06). SHA-256 hashes:

| File | SHA-256 |
| --- | --- |
| `cyp-challenge-TEST-BLINDED.csv` | `a342f8444a8dcb531ca12f3685293f0bd6c36ae9073f491e44a9bc1cc4b741f9` |
| `cyp-challenge-TRAIN_Emax.csv` | `70e9eb41f2401d8c79b684c42da3721a7aa4fd5c7f561636025c5b91cc29a3b9` |
| `cyp-challenge-TRAIN_TDI.csv` | `b458f599a792412292664386e8f18adc5d4a4129d6bd212ae80a60fb9b96bb60` |
| `cyp-challenge-TRAIN_inhibition.csv` | `b8f79addd266fb6f9f4c222c5e4e73d926362328b6a8d2841871a54e46bd2278` |
| `cyp-challenge-single-concentration-TRAIN.csv` | `cf275440aa33d10b16e7a3a19f1b196d59dc379052175ab79b47ca43db663c7a` |

## Submission contract

Regression and classification are uploaded as separate `.parquet` (preferred by the
portal) or `.csv` files. For maximum compatibility, emit exactly the columns below,
with one row per test compound.

### Regression file: 750 rows × 6 columns

```text
SMILES
Molecule_Name
CYP1A2_pIC50_direct_inhibition
CYP2C9_pIC50_direct_inhibition
CYP2D6_pIC50_direct_inhibition
CYP3A4_pIC50_direct_inhibition
```

All four predictions must be finite numeric values: no missing values, `NaN`, `inf`,
or `-inf`.

### Classification file: 750 rows × 4 columns

```text
SMILES
Molecule_Name
CYP2D6_is_TDI
CYP3A4_is_TDI
```

Both predictions must be fully populated boolean/binary values. The validators accept
`True`/`False` or `1`/`0`. Submit hard classes, not probabilities.

The official tutorial notebook happens to emit the two classification endpoints in
the opposite order. Current validation is name-based, so order is not material; this
project will standardize on the live Space schema shown above.

### Mandatory pre-upload checks

- Exactly 750 rows and exactly one row for every blinded-test `Molecule_Name`.
- No duplicate, missing, or unexpected identifiers.
- `SMILES` and `Molecule_Name` copied from the test file without modification.
- Exact, case-sensitive target names.
- All required target columns present and fully populated with valid types.
- Predictions generated from the frozen model/config/seed recorded in the method
  report.
- Run the pinned official helpers
  `validation.activity_validation.validate_activity_submission` and/or
  `validation.tdi_validation.validate_tdi_submission`, passing the set of expected
  test identifiers.
- Inspect the final artifact after serialization and reloading; do not validate only
  the in-memory table.

## Participation and operational rules

- Create/use one Hugging Face account representing the team or lab. Cooperating team
  members may not create separate leaderboard entries.
- The two tracks are independent. Entering one does not require entering the other.
- Uploads are rate-limited to one per track per 12 hours. Only the latest valid
  submission appears in that track's live standings.
- External public data and pretrained models are allowed. Proprietary training data
  is also allowed but must be disclosed; that disclosure is shown publicly.
- Open code is encouraged but is not required for standard leaderboard ranking. The
  current FAQ ties open code/public method details to recognition for innovative ML.
- The portal requires a valid Hugging Face username. Anonymous display is available,
  but an alias is then required.
- A reachable `https://` method-report link is required when claiming open-source
  code, and the portal states that a method report is required before the deadline to
  appear on the final leaderboard.
- The portal says successful submissions should be processed within two hours. Check
  `#cyp-challenge-submissions` for receipts/errors; ask in `#cyp-challenge` if no
  receipt appears after two hours.

## Timeline

All submission deadlines are 23:59 UTC.

| Date | Milestone |
| --- | --- |
| 2026-07-29 | Challenge announced |
| 2026-08-17 | Data released; submissions opened |
| 2026-09-24 | Intermediate submission deadline |
| 2026-09-25 | One-time full-test interim leaderboard released |
| 2026-11-03 | Final submission deadline; challenge closes |
| From 2026-11-04 | Final leaderboard, webinars, and organizer write-up |

Internal policy: target a validated candidate at least 48 hours before each official
deadline. Do not depend on a last-minute upload or consume the 12-hour cooldown with
an unvalidated file.

## Modeling and validation work plan

The pre-registered model matrix, PyTorch architecture, transfer-learning controls,
and stop/go gates are specified in [`experiment_design.md`](experiment_design.md).

### 1. Freeze inputs and build one molecule table

- Record dataset commit/hash metadata above in every run manifest.
- Parse and validate SMILES with RDKit; log invalid structures rather than silently
  dropping them.
- Normalize structures only in model features. Retain raw identifiers for outputs.
- Join sparse labels, confidence bounds, Emax, and primary-screen measurements by
  `Molecule_Name` with explicit one-to-one/one-to-many checks.
- Keep assay/batch fields available for diagnostics; do not let plate identifiers act
  as accidental shortcuts in a molecule-only test setting.

### 2. Define leakage-resistant validation before tuning

- Use a deterministic Bemis–Murcko scaffold or fingerprint-cluster group split.
- Keep close analogs in the same fold. The test set was created by analog expansion,
  and the live subset is also split by chemisimilar series, so random molecule splits
  are expected to be optimistic.
- Use identical outer folds for models being compared. Fix and record every seed.
- Report the mean and dispersion across folds for the official primary metric and all
  secondary metrics, both overall and per isoform.
- Fit preprocessing, feature selection, calibration, and thresholds inside each fold.

### 3. Establish reproducible baselines

- Reproduce the organizers' baseline: RDKit descriptors plus one LightGBM model per
  endpoint, with missing labels excluded only for that endpoint.
- Add the radius-3 Morgan/ECFP6 nearest-neighbor baseline above as a chemical-series
  sanity check.
- Score regression predictions with the exact official ST-RAE implementation and the
  provided confidence bounds.
- Score TDI with hard labels and MCC. Compare the default 0.5 cutoff with thresholds
  selected only from out-of-fold predictions.

Random 75/25 splits may be reproduced to compare with the tutorial, but they are not
the primary model-selection evidence.

### 4. Evaluate higher-value improvements

Prioritize improvements that exploit the structure of this dataset:

- Multi-task learning across the four direct-inhibition endpoints; the organizers
  report that their internal multi-task experiments beat separate single-task models.
- Auxiliary supervision from the single-concentration screen and Emax values.
- Joint or shared-representation modeling of direct and TDI-condition pIC50, Emax,
  and TDI labels.
- Class weighting, balanced sampling, calibrated probabilities, and out-of-fold
  threshold selection for TDI.
- Ensembling only when gains persist across grouped folds and materially improve the
  primary score.
- External public CYP data/pretrained chemical encoders, with explicit provenance,
  assay-unit harmonization, and duplicate/similarity audits against train and test.

Do not select models on live leaderboard noise. Use it only as a coarse external
check; preserve the single full-test interim reveal as independent evidence.

### 5. Train, freeze, and submit

- Select the final configuration using a predeclared aggregation of grouped-fold
  metrics, with regression MA-ST-RAE and classification MA-MCC as the primary gates.
- Retrain on all eligible labeled data with fixed seeds. Average only independently
  trained models whose ensemble rule was fixed during validation.
- Produce regression and classification artifacts independently.
- Run the mandatory pre-upload checks, save hashes/config/environment information,
  and complete disclosure/method-report metadata.
- Upload early enough to observe the two-hour processing window and 12-hour cooldown.

## Acceptance criteria

The participation task is ready for a milestone submission when:

- [ ] A versioned, deterministic training/inference command recreates the artifact.
- [ ] Dataset hashes, code revision, environment, folds, seeds, and model parameters
      are recorded.
- [ ] Grouped out-of-fold evaluation uses the official primary metric implementation.
- [ ] Per-endpoint scores and failure slices are reviewed, not only the macro score.
- [ ] The final file contains exactly the expected 750 unique test identifiers and
      exact required columns.
- [ ] Serialized output passes the pinned official local validator with expected IDs.
- [ ] No submitted target contains a missing, infinite, nonnumeric, or invalid boolean
      value.
- [ ] Proprietary/external data and open-code claims are accurately disclosed.
- [ ] A reachable method report is ready before the final deadline.
- [ ] The upload is acknowledged and appears on the intended track leaderboard.

## Risks and open checks

1. **Rules may evolve.** The Space is operational software and Discord carries live
   announcements. Recheck both before the interim and final submissions.
2. **Series leakage can dominate apparent performance.** Random folds are likely to
   overstate generalization to the analog-expansion test set.
3. **Sparse labels are not negatives.** Treating missing pIC50/TDI entries as inactive
   would create incorrect training targets.
4. **Assays differ by isoform.** CYP2D6's mass-spec readout may introduce a systematic
   domain difference from the three fluorescence endpoints.
5. **Low-activity regression labels are uncertain.** Optimize the actual interval-aware
   score and inspect behavior around pIC50 4; do not overfit noisy point estimates.
6. **TDI decisions are threshold-sensitive.** Tune hard-class thresholds strictly on
   grouped out-of-fold predictions and retain a no-tuning 0.5 baseline.
7. **Submission validation is necessary but not sufficient.** The portal's front-end
   checks are less strict than a full expected-ID comparison; always use the tutorial
   validator with the blinded test ID set.
8. **Dormant structure-track code is not a specification.** Wait for a formal launch,
   final dataset, metric, and schema before allocating work to it.

## Official references

- [Live challenge and submission Space](https://huggingface.co/spaces/openadmet/cyp-challenge)
- [Launch-day post](https://openadmet.ghost.io/openadmets-cyp-challenge-is-underway/)
- [Detailed announcement, assay, labels, rules, and timeline](https://openadmet.ghost.io/announcing-openadmets-cyp-inhibition-blind-challenge/)
- [Dataset card and downloads](https://huggingface.co/datasets/openadmet/cyp-challenge-train-test)
- [Official tutorial repository](https://github.com/OpenADMET/CYP-Challenge-Tutorial)
- [Pinned tutorial/evaluation snapshot](https://github.com/OpenADMET/CYP-Challenge-Tutorial/tree/858ae63ce79934113bccdb7fc65467de5f7b1935)
- [Pinned live-Space source snapshot](https://huggingface.co/spaces/openadmet/cyp-challenge/tree/13c5057b37d1e72b3f036dd0d59718b1823f8fdd)
- [Announcement archival DOI](https://doi.org/10.5281/zenodo.21789716)
- [Launch-post archival DOI](https://doi.org/10.5281/zenodo.21987121)
- [DrrDom/pmapper repository](https://github.com/DrrDom/pmapper)
- [Pmapper pharmacophore API](https://pmapper.readthedocs.io/en/latest/pharmacophore_class.html)
