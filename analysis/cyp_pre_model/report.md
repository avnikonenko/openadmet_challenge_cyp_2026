# OpenADMET CYP exploratory and validation analysis

> Generated 2026-09-12 from the checked-out challenge CSVs. This is a pre-model
> analysis: no final predictive model and no test-label proxy was trained. Every
> summary table is CSV under [`tables/`](tables/) and every new figure is PNG under
> [`figures/`](figures/).

## Executive findings

1. **The labels are sparse and missing non-randomly.** Of 4,905 direct-inhibition
   molecules, 3,596 have one CYP label,
   1,039 have two, 229
   have three, and only 41 have all four. A masked
   multitask setup can use this matrix; a complete-case model would discard 99.2%.
2. **Multitask sharing should be selective.** CYP2C9/CYP3A4 has paired Spearman
   rho=0.688,
   while all CYP2D6 pairwise correlations are near zero. A shared encoder is
   scientifically plausible, but CYP-specific heads and single-task controls are
   required.
3. **The primary screen is unusually informative.** Its per-CYP Spearman correlation
   with direct pIC50 ranges from -0.828 to -0.936. It is the first auxiliary target to
   test for internal transfer learning.
4. **Close 2D analogs can still be activity cliffs.** Among 260 CYP3A4 pairs at ECFP6
   similarity >=0.6, 54 (20.8%) differ by at least 1 pIC50 and 15 differ by
   at least 2. Similarity generally improves SAR smoothness, but does not guarantee it.
5. **CYP2D6 has the weakest test coverage.** Its test-to-CYP-training nearest ECFP6
   median is 0.394, versus
   0.466 for CYP3A4.
6. **Single-conformer 3D neighbor identities are unstable.** On the declared 40-query,
   237-reference, five-conformer subset, the NN changed for
   95.0% of USR,
   72.5% of USRCAT, and
   70.0% of pmapper-count
   queries.

The ranked modelling consequences appear at the end of this report. The compact
machine-readable version is [`analysis_summary.csv`](tables/analysis_summary.csv).

## 1. Integrity and molecular identity

All 6,147 unique-name training molecules and
all 750 test molecules parse with RDKit. There are no
duplicate full isomeric canonical molecules within either set and no duplicated raw
SMILES. Exact canonical train/test overlap is
0, and there are
0 molecule-name/canonical-identity conflicts across input
files. The single-concentration file's repeated identities are the expected four CYP
rows per molecule, not replicate measurements. No duplicate molecule/CYP/assay keys
or conflicting replicates were found.

The train union has 5 nonisomeric identity groups containing more
than one stereoisomer, versus 2 in test. There are
1 obvious train tautomer-parent groups with multiple canonical
forms, and 1 multi-fragment train
molecule. These groups are reported but not merged. RDKit parsing—not optional
fragment/charge/tautomer parenting—defines whether a molecule is valid.

The direct-pIC50 columns duplicated in the direct and TDI files agree exactly:
0 numeric conflicts and
0 missingness conflicts. See
[`dataset_inventory.csv`](tables/dataset_inventory.csv),
[`input_file_hashes.csv`](tables/input_file_hashes.csv),
[`identity_resolution_counts.csv`](tables/identity_resolution_counts.csv),
[`duplicate_identity_groups.csv`](tables/duplicate_identity_groups.csv), and
[`cross_file_measurement_agreement.csv`](tables/cross_file_measurement_agreement.csv).

## 2. Direct-pIC50 targets

| CYP | N | mean | median | SD | IQR | range | skew | active >=4 | Tukey outliers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CYP1A2 | 1412 | 4.955 | 5.133 | 1.030 | 1.004 | 1.91-7.95 | -0.831 | 83.6% | 110 |
| CYP2C9 | 1285 | 4.581 | 4.621 | 0.782 | 0.922 | 2.10-7.47 | -0.342 | 79.8% | 46 |
| CYP2D6 | 1493 | 4.784 | 4.726 | 0.916 | 0.832 | 1.95-7.53 | -0.673 | 91.4% | 141 |
| CYP3A4 | 2335 | 4.096 | 4.266 | 1.093 | 1.498 | 1.91-7.19 | -0.348 | 59.6% | 1 |

The endpoints differ materially: CYP3A4 has the lowest mean and only
59.6% at pIC50 >=4, whereas CYP2D6
has 91.4%. All distributions are
left-skewed. Outliers are reported rather than removed.

![Target distributions](figures/target_pic50_distributions.png)

## 3. Missingness and suitability for multitask learning

Availability counts are 1/2/3/4 CYPs =
3,596/1,039/
229/41. Missingness
is chemically structured. The largest labeled-versus-missing property effects are:

| CYP | property | labeled N | missing N | SMD missing-labeled | p |
| --- | --- | --- | --- | --- | --- |
| CYP1A2 | Fsp3 | 1412 | 3493 | 0.485 | 0.000 |
| CYP2D6 | TPSA | 1493 | 3412 | 0.463 | 0.000 |
| CYP1A2 | AromaticRings | 1412 | 3493 | -0.446 | 0.000 |
| CYP2D6 | cLogP | 1493 | 3412 | -0.422 | 0.000 |

Scaffold/missingness Cramer's V ranges from
0.089 to 0.145.
This supports masked multitask learning but makes random missing-label assumptions
unsafe: preprocessing and task weighting must be fold-local.

![Missingness](figures/missingness_and_assay_coverage.png)

## 4. Cross-CYP relationships

| CYP 1 | CYP 2 | paired N | Pearson | Spearman |
| --- | --- | --- | --- | --- |
| CYP1A2 | CYP2C9 | 295 | 0.451 | 0.423 |
| CYP1A2 | CYP2D6 | 427 | 0.077 | 0.059 |
| CYP1A2 | CYP3A4 | 297 | 0.415 | 0.383 |
| CYP2C9 | CYP2D6 | 230 | -0.100 | -0.120 |
| CYP2C9 | CYP3A4 | 473 | 0.705 | 0.688 |
| CYP2D6 | CYP3A4 | 250 | 0.039 | 0.056 |

Using explicit exploratory definitions (at least three observed CYPs all >=5 for
*broad strong*; max >=5, min <=4, range >=1.5 for *selective*; range >=2 for
*discordant*), there are
32 broad-strong,
103 selective, and
127 discordant compounds. Categories
overlap and are diagnostic, not new labels.

![Cross-CYP relationships](figures/cross_cyp_pairwise_pic50.png)

## 5. Experimental uncertainty

Median 95% interval widths are
CYP1A2=0.328, CYP2C9=0.526, CYP2D6=0.272, CYP3A4=0.379.
Interval width is strongly anticorrelated with activity: Spearman rho is
CYP1A2=-0.885, CYP2C9=-0.899, CYP2D6=-0.558, CYP3A4=-0.928.
Uncertainty is therefore not independent noise; it partly encodes assay dynamic
range.

Raw inverse-variance weighting reduces effective sample size to
CYP1A2=53.0%, CYP2C9=37.0%, CYP2D6=55.6%, CYP3A4=32.1%.
The official interval-aware loss is the primary future experiment. If weights are
tested, clip them within each training fold; do not use raw inverse variance
uncritically. A heteroscedastic head is reasonable only as a matched ablation because
the supplied uncertainty is highly activity-dependent. Replicate counts are not
available, so no uncertainty-versus-replicate analysis is claimed.

![Measurement uncertainty](figures/measurement_uncertainty.png)

## 6. ECFP6 activity cliffs

Every unique measured pair was evaluated exactly: 996,166 CYP1A2, 824,970 CYP2C9,
1,113,778 CYP2D6, and 2,724,945 CYP3A4 pairs.

| CYP | similarity | pairs | median delta | delta >=1 N | delta >=1 % | delta >=2 N |
| --- | --- | --- | --- | --- | --- | --- |
| CYP1A2 | >=0.6 | 19 | 0.371 | 3 | 15.8% | 2 |
| CYP1A2 | >=0.7 | 5 | 0.454 | 1 | 20.0% | 0 |
| CYP1A2 | >=0.8 | 1 | 0.016 | 0 | 0.0% | 0 |
| CYP2C9 | >=0.6 | 6 | 1.285 | 3 | 50.0% | 1 |
| CYP2C9 | >=0.7 | 0 | NA | 0 | NA | 0 |
| CYP2C9 | >=0.8 | 0 | NA | 0 | NA | 0 |
| CYP2D6 | >=0.6 | 21 | 0.334 | 4 | 19.0% | 1 |
| CYP2D6 | >=0.7 | 8 | 0.513 | 2 | 25.0% | 1 |
| CYP2D6 | >=0.8 | 6 | 0.717 | 2 | 33.3% | 1 |
| CYP3A4 | >=0.6 | 260 | 0.425 | 54 | 20.8% | 15 |
| CYP3A4 | >=0.7 | 45 | 0.435 | 6 | 13.3% | 1 |
| CYP3A4 | >=0.8 | 8 | 0.182 | 1 | 12.5% | 0 |

At high similarity, counts for CYP1A2/CYP2C9 are small, so their percentages are
unstable. CYP3A4 provides the strongest evidence: median delta falls with similarity,
yet meaningful same- and different-scaffold cliffs persist. Increasing structural
similarity implies smoother average SAR, not a smooth-label guarantee. See the full
same-scaffold stratification in
[`activity_cliff_threshold_statistics.csv`](tables/activity_cliff_threshold_statistics.csv).

![Activity cliffs](figures/activity_cliffs_ecfp6_hexbin.png)

![Activity cliff examples](figures/activity_cliff_examples.png)

## 7. Scaffold structure

The 6,147-molecule training union contains 5,367
scaffolds; 93.4% of
scaffold identities are singletons. Test has 546
scaffolds and 83.0% singleton
scaffolds. Only 187/750
test molecules and 66
test scaffold identities overlap training.

Per-CYP test molecule scaffold coverage is
CYP1A2=12.5%, CYP2C9=11.1%, CYP2D6=7.3%, CYP3A4=15.7%.
Within-scaffold variances and high-range/selective scaffold examples are retained in
CSV rather than reduced to cherry-picked pictures.

![Scaffold summary](figures/scaffold_coverage_and_activity_ranges.png)

## 8. CYP-specific chemical-space coverage

| CYP | training N | within-train NN median | test NN median | test >=0.5 | test >=0.6 |
| --- | --- | --- | --- | --- | --- |
| CYP1A2 | 1412 | 0.343 | 0.454 | 35.9% | 10.0% |
| CYP2C9 | 1285 | 0.384 | 0.453 | 32.5% | 8.4% |
| CYP2D6 | 1493 | 0.280 | 0.394 | 30.3% | 7.6% |
| CYP3A4 | 2335 | 0.402 | 0.466 | 37.1% | 11.7% |

CYP2D6 is the weakest domain by test median and exact scaffold coverage; CYP3A4 is the
best covered, consistent with its larger labeled set. This must be evaluated per CYP,
not inferred from the 6,147-molecule union.

![Per-CYP test coverage](figures/per_cyp_test_chemical_space_coverage.png)

## 9. Auxiliary assays

| assay relationship | CYP | paired N | Spearman |
| --- | --- | --- | --- |
| single_concentration | CYP1A2 | 1412 | -0.862 |
| single_concentration | CYP2C9 | 1285 | -0.896 |
| single_concentration | CYP2D6 | 1493 | -0.828 |
| single_concentration | CYP3A4 | 1805 | -0.936 |
| Emax_direct | CYP1A2 | 1412 | -0.394 |
| Emax_direct | CYP2C9 | 1285 | -0.407 |
| Emax_direct | CYP2D6 | 1493 | 0.770 |
| Emax_direct | CYP3A4 | 2335 | -0.462 |
| TDI_vs_direct_pIC50 | CYP1A2 | 1412 | 0.990 |
| TDI_vs_direct_pIC50 | CYP2C9 | 1285 | 0.975 |
| TDI_vs_direct_pIC50 | CYP2D6 | 1493 | 0.906 |
| TDI_vs_direct_pIC50 | CYP3A4 | 2334 | 0.954 |

Single-concentration data are at one concentration per CYP in this snapshot and are
strongly monotonic with direct pIC50. Emax is weaker and strongly CYP-dependent;
CYP2D6 has the opposite correlation sign from the fluorescence CYPs, consistent with
its different assay readout and requiring separate scaling/heads. Direct and
TDI-condition pIC50 are very strongly correlated, but their shift is the TDI-defining
signal.

After rank-residual control for direct pIC50, TDI-condition Emax retains association
with scored TDI labels for CYP2D6 (partial rho=
0.404)
and CYP3A4 (0.128).
This justifies a leakage-controlled multi-assay/transfer experiment, not blind
concatenation of assays as rows.

![Single concentration and pIC50](figures/single_concentration_vs_pic50.png)

![Emax and pIC50](figures/emax_vs_pic50.png)

## 10. Representation complementarity

Similarity values were never compared directly across descriptor types. On the same
750 test queries, ECFP6 shares nearest-neighbor identity with USR for
1.6%, USRCAT for
10.0%, pmapper count for
22.5%, and hashed pmapper for
11.2%. Across 20,000 fixed-seed
test/train pairs, ECFP6 similarity-rank correlations are respectively
0.053, 0.149,
0.193, and
0.099.

USR remains over-permissive (nearest median 0.925).
USRCAT is more discriminating (median 0.375)
but conformer-sensitive. Hashed pmapper is invalid as a primary representation:
3 train molecules set all
2,048 bits, and these fully saturated structures are selected for
53/750 test queries.
Single-conformer embedding succeeded for 6,895/
6,897 molecules; both failures are training molecules and are
retained with their error messages in
[`single_conformer_3d_descriptors.csv`](tables/single_conformer_3d_descriptors.csv).

![Representation complementarity](figures/representation_complementarity.png)

## 11. Multi-conformer sensitivity

The deterministic subset contains 40 test queries stratified over four ECFP6
coverage bands and 237 candidate train
references: single-representation NNs plus fixed-seed random references. All received
five conformers. It is deliberately manageable and does not claim exhaustive
full-training-space 3D neighbors.

| representation | queries | NN changed | median unique NNs | median modal fraction | median best-score SD | p95 best-score SD |
| --- | --- | --- | --- | --- | --- | --- |
| USR | 40 | 95.0% | 4.000 | 0.400 | 0.020 | 0.038 |
| USRCAT | 40 | 72.5% | 2.000 | 0.700 | 0.030 | 0.111 |
| pmapper_count | 40 | 70.0% | 2.000 | 0.800 | 0.022 | 0.151 |

Single-conformer 3D descriptors are not sufficiently stable for primary modeling.
If retained later, use deterministic conformer ensembles with an aggregation rule
chosen inside CV.

![Conformer sensitivity](figures/conformer_sensitivity.png)

## 12. Candidate CV splits and applicability domain

The ECFP schemes use exact threshold-graph connected components, an order-independent
alternative to Butina series clustering. At thresholds 0.5/0.6 they produce
4,940/
5,451 groups; their largest components
contain only 22/
21 molecules, so component
chaining is controlled.

| scheme | fold min | fold max | validation NN | test NN same ref | val-test gap | scaffold seen | val NN <0.4 | val NN >=0.6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ecfp_component_0.5 | 1186 | 1290 | 0.359 | 0.501 | -0.142 | 7.0% | 68.6% | 0.0% |
| ecfp_component_0.6 | 1193 | 1284 | 0.378 | 0.506 | -0.128 | 9.6% | 58.2% | 0.0% |
| random | 1229 | 1230 | 0.393 | 0.507 | -0.114 | 16.7% | 52.1% | 13.9% |
| scaffold_group | 1191 | 1278 | 0.383 | 0.506 | -0.122 | 0.0% | 55.2% | 8.4% |

All validation schemes are chemically harder than test against the same fold-training
reference. Random CV is the closest but permits analog leakage. Scaffold CV forces
0% scaffold overlap although the actual test has 24.9%.
The recommended primary scheme is therefore ECFP component 0.6: it blocks pairs at
similarity >=0.6 across folds, is less extreme than 0.5, and preserves balanced fold
sizes. Report random CV as an interpolation ceiling and scaffold CV as a novelty
stress test; do not tune against whichever split gives the best score. Fold-training
mean predictions score approximately 1 under the exact official ST-RAE, providing an
implementation sanity check.

![CV novelty](figures/cv_split_novelty_and_fold_sizes.png)

## 13. Train/test physicochemical shifts

| property | train mean | test mean | SMD test-train | KS | Wasserstein |
| --- | --- | --- | --- | --- | --- |
| AromaticRings | 2.411 | 2.411 | 0.000 | 0.024 | 0.090 |
| Fsp3 | 0.365 | 0.291 | -0.435 | 0.247 | 0.078 |
| HBA | 4.198 | 3.805 | -0.300 | 0.143 | 0.399 |
| HBD | 1.208 | 1.532 | 0.373 | 0.171 | 0.335 |
| MW | 359.450 | 337.500 | -0.356 | 0.141 | 22.839 |
| RotatableBonds | 4.648 | 4.503 | -0.072 | 0.067 | 0.348 |
| TPSA | 70.227 | 71.266 | 0.049 | 0.038 | 1.722 |
| cLogP | 2.965 | 2.835 | -0.115 | 0.063 | 0.152 |

Fsp3 is the largest univariate shift (SMD
-0.435, KS
0.247); test is also smaller by about
22.0 Da. TPSA and aromatic
ring count are well aligned. Applicability files provide molecule-level max ECFP6,
seen/unseen scaffold, and low/moderate/high regimes for every test molecule and every
validation assignment.

![Train/test properties](figures/train_test_property_distributions.png)

## Implications for planned model families

- **LightGBM baseline:** use radius-3 Morgan plus fold-fitted RDKit descriptors. It is
  the low-variance anchor, but report CYP-specific applicability and cliff slices.
- **Multitask Chemprop:** justified by sparse complementary labels and some strong
  cross-CYP relationships, but use masked losses and CYP-specific heads; CYP2D6 needs
  a single-task control because it correlates poorly with the other endpoints.
- **Transfer learning:** internal assay pretraining is justified, especially from
  single-concentration data. Exclude each outer-validation molecule from every
  auxiliary pretraining stage to prevent molecule memorization.
- **Multi-assay learning:** use assay-specific heads/normalization. CYP2D6 Emax has a
  different direction from fluorescence assays, and direct/TDI pIC50 values are too
  redundant to count as independent evidence without modeling their shift.
- **Uncertainty-aware modeling:** compare the official interval loss first, clipped
  inverse-variance weights second, and a heteroscedastic head third. Raw inverse
  variance is too concentrated and activity-dependent.
- **3D/pmapper:** defer until multi-conformer aggregation. Exclude hashed pmapper and
  shape-only USR from primary models.

## Explicit warnings

- Missing targets are unmeasured, not negative or inactive.
- The four training files are mostly assay views of the same molecules; concatenating
  them as independent rows creates leakage and sample weighting errors.
- Auxiliary labels for an outer-validation molecule cannot be used during pretraining
  in that fold.
- Random folds can split close analogs; scaffold folds are harsher than actual test
  scaffold overlap. Report all three perspectives.
- Activity cliffs remain at high ECFP6 similarity. Nearest-neighbor similarity is an
  applicability diagnostic, not a predictor.
- Raw similarity magnitudes are descriptor-specific. USR saturation and pmapper-bit
  collisions make cross-descriptor threshold comparisons invalid.
- The five-conformer result is a representative subset study, not proof of exhaustive
  conformational coverage.
- Tautomer/protonation parent groups are diagnostics only; none were automatically
  merged.

## Ranked modelling consequences

1. Use **ECFP6-component 0.6 five-fold CV** for primary selection, random CV as an
   interpolation control, and scaffold CV as a stress test.
2. Establish **Morgan radius-3 + RDKit 2D LightGBM** before neural tuning.
3. Test **masked multitask learning**, but retain CYP-specific heads and a CYP2D6
   single-task control.
4. Make **single-concentration auxiliary pretraining** the first transfer experiment;
   enforce molecule-level fold exclusion across every assay.
5. Optimize/report the **official interval-aware ST-RAE** and test clipped uncertainty
   weighting; avoid raw inverse-variance weighting.
6. Evaluate errors by **ECFP6 applicability, scaffold novelty, and activity-cliff
   membership**, especially for CYP2D6.
7. Use 3D only through a **multi-conformer aggregation ablation**; do not use USR or
   hashed pmapper as primary inputs.
8. Ensemble only representations that add **paired out-of-fold value**, because low
   neighbor overlap alone does not establish predictive complementarity.

## Reproduction

Run from the project root with the existing `openadmet-cyp-sim` environment:

```bash
conda run -n openadmet-cyp-sim python analysis/cyp_pre_model/scripts/run_all.py
```

The scripts use seed 20260912, stable sorting for output tables, the official
ST-RAE logic from tutorial revision
`858ae63ce79934113bccdb7fc65467de5f7b1935`, and project-local matplotlib/output
directories. No analytical figure is written under `/tmp`.
