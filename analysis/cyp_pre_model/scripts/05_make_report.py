#!/usr/bin/env python3
"""Build the concise markdown interpretation and modelling-priority table."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ANALYSIS_DIR, CYPS, TABLES_DIR, write_csv  # noqa: E402


def read(name: str) -> pd.DataFrame:
    path = TABLES_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Required analysis output is missing: {path}")
    return pd.read_csv(path)


def percent(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%"


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    def clean(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            return "NA" if not np.isfinite(value) else f"{value:.3f}"
        return str(value).replace("|", "\\|")

    output = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    output.extend("| " + " | ".join(clean(value) for value in row) + " |" for row in rows)
    return "\n".join(output)


def main() -> int:
    inventory = read("dataset_inventory.csv").set_index("dataset")
    identity_counts = read("identity_resolution_counts.csv")
    identity_groups = read("duplicate_identity_groups.csv")
    cross_split_identity = read("cross_split_identity_overlap.csv").set_index("identity_level")
    cross_source_conflicts = read("cross_source_name_identity_conflicts.csv")
    duplicate_measurements = read("duplicate_measurements.csv")
    agreement = read("cross_file_measurement_agreement.csv")
    targets = read("target_distribution_summary.csv").set_index("CYP")
    label_counts = read("label_count_per_molecule.csv").set_index("number_of_CYP_labels")
    missing_property = read("missingness_property_associations.csv")
    missing_scaffold = read("missingness_scaffold_association.csv").set_index("CYP")
    cross = read("cross_cyp_correlations.csv")
    categories = read("cross_cyp_compound_categories.csv")
    uncertainty = read("uncertainty_summary.csv")
    uncertainty_rel = read("uncertainty_relationships.csv")
    weights = read("uncertainty_weighting_diagnostics.csv")
    cliffs = read("activity_cliff_threshold_statistics.csv")
    scaffold = read("scaffold_summary.csv").set_index("split")
    cyp_scaffold = read("per_cyp_scaffold_coverage.csv").set_index("CYP")
    cyp_coverage = read("per_cyp_chemical_space_coverage.csv").set_index("CYP")
    auxiliary = read("auxiliary_assay_correlations.csv")
    partial = read("emax_partial_associations.csv")
    representation = read("representation_nearest_score_summary.csv").set_index("representation")
    representation_overlap = read("representation_nearest_neighbor_overlap.csv")
    representation_rank = read("representation_pairwise_rank_correlations.csv")
    saturation = read("pmapper_hashed_saturation_diagnostics.csv").set_index("diagnostic")
    conformer = read("conformer_sensitivity_summary.csv").set_index("representation")
    conformer_metadata = read("conformer_sensitivity_embedding_metadata.csv")
    single_conformer = read("single_conformer_3d_descriptors.csv")
    shifts = read("train_test_property_shift_statistics.csv").set_index("property")
    cv = read("cv_split_summary.csv")
    cluster = read("ecfp_cluster_summary.csv").set_index("scheme")

    stereo_train = identity_groups.loc[(identity_groups["source"] == "train_union") & (identity_groups["identity_level"] == "nonisomeric_smiles")]
    stereo_test = identity_groups.loc[(identity_groups["source"] == "test") & (identity_groups["identity_level"] == "nonisomeric_smiles")]
    tautomer_train = identity_groups.loc[(identity_groups["source"] == "train_union") & (identity_groups["identity_level"] == "tautomer_parent_smiles") & identity_groups["unique_canonical_smiles"].gt(1)]
    strongest_missing = missing_property.loc[missing_property["analysis"] == "endpoint_missingness"].assign(abs_smd=lambda frame: frame["SMD_missing_minus_labeled"].abs()).nlargest(4, "abs_smd")

    cross_rows = [[row.CYP_1, row.CYP_2, int(row.overlap_N), row.pearson, row.spearman] for row in cross.itertuples(index=False)]
    target_rows = [[cyp, int(targets.loc[cyp, "N"]), targets.loc[cyp, "mean"], targets.loc[cyp, "median"], targets.loc[cyp, "SD"], targets.loc[cyp, "IQR"], f"{targets.loc[cyp, 'min']:.2f}-{targets.loc[cyp, 'max']:.2f}", targets.loc[cyp, "skewness"], percent(targets.loc[cyp, "active_fraction_ge_4"]), int(targets.loc[cyp, "tukey_outliers"])] for cyp in CYPS]
    cliff_rows = []
    for cyp in CYPS:
        for threshold in (0.6, 0.7, 0.8):
            row = cliffs.loc[(cliffs["CYP"] == cyp) & (cliffs["pair_scope"] == "all") & np.isclose(cliffs["similarity_threshold"], threshold)].iloc[0]
            cliff_rows.append([cyp, f">={threshold:.1f}", int(row.pair_count), row.delta_median, int(row.cliff_delta_ge_1_count), percent(row.cliff_delta_ge_1_fraction) if np.isfinite(row.cliff_delta_ge_1_fraction) else "NA", int(row.cliff_delta_ge_2_count)])

    aux_rows = []
    for dataset in ("single_concentration", "Emax_direct", "TDI_vs_direct_pIC50"):
        for cyp in CYPS:
            row = auxiliary.loc[(auxiliary["auxiliary_dataset"] == dataset) & (auxiliary["CYP"] == cyp)].iloc[0]
            aux_rows.append([dataset, cyp, int(row.overlap_N), row.spearman])

    cv_aggregate = cv.groupby("scheme").agg(
        fold_min=("validation_molecules", "min"),
        fold_max=("validation_molecules", "max"),
        validation_nn=("validation_to_train_NN_median", "mean"),
        test_nn=("test_to_same_fold_train_NN_median", "mean"),
        median_gap=("validation_minus_test_NN_median", "mean"),
        scaffold_seen=("validation_scaffold_seen_fraction", "mean"),
        low_fraction=("validation_fraction_NN_lt_0.4", "mean"),
        high_fraction=("validation_fraction_NN_ge_0.6", "mean"),
    )
    cv_rows = [[scheme, int(row.fold_min), int(row.fold_max), row.validation_nn, row.test_nn, row.median_gap, percent(row.scaffold_seen), percent(row.low_fraction), percent(row.high_fraction)] for scheme, row in cv_aggregate.iterrows()]

    nearest_overlap = {(row.representation_1, row.representation_2): row.same_nearest_neighbor_fraction for row in representation_overlap.itertuples(index=False)}
    pair_rank = {(row.representation_1, row.representation_2): row.spearman for row in representation_rank.itertuples(index=False)}

    summary_rows = [
        {"analysis": "Molecule integrity", "main result": f"0 invalid molecules; 0 duplicate canonical molecules in the {int(inventory.loc['train_union', 'rows']):,}-molecule train union; {int(inventory.loc['train_union', 'multi_fragment'])} multi-fragment molecule", "implication for modelling": "Retain every row; preserve salts/parent identities as audited metadata", "priority": "High"},
        {"analysis": "Label sparsity", "main result": f"{int(label_counts.loc[1, 'molecules']):,}/{int(inventory.loc['direct', 'rows']):,} direct rows have one CYP and only {int(label_counts.loc[4, 'molecules'])} have all four", "implication for modelling": "Masked multitask loss is necessary; complete-case training is unacceptable", "priority": "Critical"},
        {"analysis": "Cross-CYP signal", "main result": "CYP2C9/CYP3A4 Spearman 0.688, but CYP2D6 pairs are near zero", "implication for modelling": "Use a shared encoder with CYP-specific heads; retain single-task controls", "priority": "High"},
        {"analysis": "Activity cliffs", "main result": f"At ECFP6 >=0.6, delta-pIC50 >=1 occurs in {int(cliffs[(cliffs.CYP=='CYP3A4') & (cliffs.pair_scope=='all') & np.isclose(cliffs.similarity_threshold,0.6)].iloc[0].cliff_delta_ge_1_count)}/260 CYP3A4 pairs", "implication for modelling": "Similarity is not a smooth-label guarantee; use robust losses and series-grouped diagnostics", "priority": "Critical"},
        {"analysis": "Per-CYP applicability", "main result": f"CYP2D6 has weakest test coverage (median ECFP6 NN {cyp_coverage.loc['CYP2D6', 'test_to_CYP_training_NN_median']:.3f})", "implication for modelling": "Expect CYP2D6 extrapolation risk and report applicability-stratified errors", "priority": "High"},
        {"analysis": "Scaffolds", "main result": f"Only {int(scaffold.loc['test', 'molecules_with_scaffold_in_other_split'])}/750 test molecules share a train scaffold", "implication for modelling": "Scaffold CV is a useful stress test but is harsher than the actual test mixture", "priority": "High"},
        {"analysis": "Single-concentration assay", "main result": "Per-CYP Spearman magnitude 0.828-0.936 versus direct pIC50", "implication for modelling": "Strongest auxiliary pretraining target; preserve CYP-specific direction/scale", "priority": "Critical"},
        {"analysis": "Emax/TDI", "main result": "Emax signal is CYP-dependent; TDI/direct pIC50 Spearman 0.906-0.990", "implication for modelling": "Multi-assay transfer is justified, but use separate heads and prevent fold-level molecule leakage", "priority": "High"},
        {"analysis": "Experimental uncertainty", "main result": "CI width is strongly activity-dependent; raw inverse-variance ESS falls to 32.1% for CYP3A4", "implication for modelling": "Prefer official interval loss or clipped/learned uncertainty weighting over raw inverse variance", "priority": "High"},
        {"analysis": "Representation complementarity", "main result": f"ECFP6/USRCAT share only {percent(nearest_overlap[('ECFP6','USRCAT')])} of test NNs; sampled pair-rank rho {pair_rank[('ECFP6','USRCAT')]:.3f}", "implication for modelling": "3D may diversify an ensemble but raw similarity scales are not comparable", "priority": "Medium"},
        {"analysis": "Pmapper hashing", "main result": f"{int(saturation.loc['test_NNs_that_are_fully_saturated', 'count'])}/750 hashed-pmapper NNs are fully saturated", "implication for modelling": "Exclude pmapper-2048; use unhashed counts only", "priority": "Critical"},
        {"analysis": "Conformer sensitivity", "main result": f"NN identity changed for {percent(conformer.loc['USRCAT','NN_changed_fraction'])} USRCAT and {percent(conformer.loc['pmapper_count','NN_changed_fraction'])} pmapper queries", "implication for modelling": "Do not treat single-conformer 3D features as stable; aggregate conformers", "priority": "High"},
        {"analysis": "CV schemes", "main result": f"ECFP component 0.6 validation median NN {cv_aggregate.loc['ecfp_component_0.6','validation_nn']:.3f} versus same-reference test {cv_aggregate.loc['ecfp_component_0.6','test_nn']:.3f}", "implication for modelling": "Use component-0.6 CV as conservative primary, random as interpolation control, scaffold as stress test", "priority": "Critical"},
        {"analysis": "Train/test property shift", "main result": f"Largest shift is Fsp3: SMD {shifts.loc['Fsp3','standardized_mean_difference_test_minus_train']:.3f}, KS {shifts.loc['Fsp3','ks_statistic']:.3f}", "implication for modelling": "Monitor lower-Fsp3/smaller test chemistry and avoid uncalibrated extrapolation", "priority": "High"},
    ]
    write_csv(pd.DataFrame(summary_rows), "analysis_summary.csv", ["priority", "analysis"])

    report = f"""# OpenADMET CYP exploratory and validation analysis

> Generated 2026-09-12 from the checked-out challenge CSVs. This is a pre-model
> analysis: no final predictive model and no test-label proxy was trained. Every
> summary table is CSV under [`tables/`](tables/) and every new figure is PNG under
> [`figures/`](figures/).

## Executive findings

1. **The labels are sparse and missing non-randomly.** Of 4,905 direct-inhibition
   molecules, {int(label_counts.loc[1, 'molecules']):,} have one CYP label,
   {int(label_counts.loc[2, 'molecules']):,} have two, {int(label_counts.loc[3, 'molecules'])}
   have three, and only {int(label_counts.loc[4, 'molecules'])} have all four. A masked
   multitask setup can use this matrix; a complete-case model would discard 99.2%.
2. **Multitask sharing should be selective.** CYP2C9/CYP3A4 has paired Spearman
   rho={cross.loc[(cross.CYP_1=='CYP2C9') & (cross.CYP_2=='CYP3A4'),'spearman'].iloc[0]:.3f},
   while all CYP2D6 pairwise correlations are near zero. A shared encoder is
   scientifically plausible, but CYP-specific heads and single-task controls are
   required.
3. **The primary screen is unusually informative.** Its per-CYP Spearman correlation
   with direct pIC50 ranges from -0.828 to -0.936. It is the first auxiliary target to
   test for internal transfer learning.
4. **Close 2D analogs can still be activity cliffs.** Among 260 CYP3A4 pairs at ECFP6
   similarity >=0.6, 54 ({percent(54/260)}) differ by at least 1 pIC50 and 15 differ by
   at least 2. Similarity generally improves SAR smoothness, but does not guarantee it.
5. **CYP2D6 has the weakest test coverage.** Its test-to-CYP-training nearest ECFP6
   median is {cyp_coverage.loc['CYP2D6','test_to_CYP_training_NN_median']:.3f}, versus
   {cyp_coverage.loc['CYP3A4','test_to_CYP_training_NN_median']:.3f} for CYP3A4.
6. **Single-conformer 3D neighbor identities are unstable.** On the declared 40-query,
   237-reference, five-conformer subset, the NN changed for
   {percent(conformer.loc['USR','NN_changed_fraction'])} of USR,
   {percent(conformer.loc['USRCAT','NN_changed_fraction'])} of USRCAT, and
   {percent(conformer.loc['pmapper_count','NN_changed_fraction'])} of pmapper-count
   queries.

The ranked modelling consequences appear at the end of this report. The compact
machine-readable version is [`analysis_summary.csv`](tables/analysis_summary.csv).

## 1. Integrity and molecular identity

All {int(inventory.loc['train_union','rows']):,} unique-name training molecules and
all {int(inventory.loc['test','rows'])} test molecules parse with RDKit. There are no
duplicate full isomeric canonical molecules within either set and no duplicated raw
SMILES. Exact canonical train/test overlap is
{int(cross_split_identity.loc['canonical_smiles','shared_identities'])}, and there are
{len(cross_source_conflicts)} molecule-name/canonical-identity conflicts across input
files. The single-concentration file's repeated identities are the expected four CYP
rows per molecule, not replicate measurements. No duplicate molecule/CYP/assay keys
or conflicting replicates were found.

The train union has {len(stereo_train)} nonisomeric identity groups containing more
than one stereoisomer, versus {len(stereo_test)} in test. There are
{len(tautomer_train)} obvious train tautomer-parent groups with multiple canonical
forms, and {int(inventory.loc['train_union','multi_fragment'])} multi-fragment train
molecule. These groups are reported but not merged. RDKit parsing—not optional
fragment/charge/tautomer parenting—defines whether a molecule is valid.

The direct-pIC50 columns duplicated in the direct and TDI files agree exactly:
{int(agreement['numeric_conflicts'].sum())} numeric conflicts and
{int(agreement['missingness_disagreements'].sum())} missingness conflicts. See
[`dataset_inventory.csv`](tables/dataset_inventory.csv),
[`input_file_hashes.csv`](tables/input_file_hashes.csv),
[`identity_resolution_counts.csv`](tables/identity_resolution_counts.csv),
[`duplicate_identity_groups.csv`](tables/duplicate_identity_groups.csv), and
[`cross_file_measurement_agreement.csv`](tables/cross_file_measurement_agreement.csv).

## 2. Direct-pIC50 targets

{markdown_table(['CYP','N','mean','median','SD','IQR','range','skew','active >=4','Tukey outliers'], target_rows)}

The endpoints differ materially: CYP3A4 has the lowest mean and only
{percent(targets.loc['CYP3A4','active_fraction_ge_4'])} at pIC50 >=4, whereas CYP2D6
has {percent(targets.loc['CYP2D6','active_fraction_ge_4'])}. All distributions are
left-skewed. Outliers are reported rather than removed.

![Target distributions](figures/target_pic50_distributions.png)

## 3. Missingness and suitability for multitask learning

Availability counts are 1/2/3/4 CYPs =
{int(label_counts.loc[1,'molecules']):,}/{int(label_counts.loc[2,'molecules']):,}/
{int(label_counts.loc[3,'molecules'])}/{int(label_counts.loc[4,'molecules'])}. Missingness
is chemically structured. The largest labeled-versus-missing property effects are:

{markdown_table(['CYP','property','labeled N','missing N','SMD missing-labeled','p'], [[row.CYP,row.property,int(row.labeled_N),int(row.missing_N),row.SMD_missing_minus_labeled,row.pvalue] for row in strongest_missing.itertuples(index=False)])}

Scaffold/missingness Cramer's V ranges from
{missing_scaffold['cramers_v'].min():.3f} to {missing_scaffold['cramers_v'].max():.3f}.
This supports masked multitask learning but makes random missing-label assumptions
unsafe: preprocessing and task weighting must be fold-local.

![Missingness](figures/missingness_and_assay_coverage.png)

## 4. Cross-CYP relationships

{markdown_table(['CYP 1','CYP 2','paired N','Pearson','Spearman'], cross_rows)}

Using explicit exploratory definitions (at least three observed CYPs all >=5 for
*broad strong*; max >=5, min <=4, range >=1.5 for *selective*; range >=2 for
*discordant*), there are
{int(categories['broad_strong_all_observed_ge_5'].sum())} broad-strong,
{int(categories['selective_max_ge_5_min_le_4_range_ge_1.5'].sum())} selective, and
{int(categories['discordant_range_ge_2'].sum())} discordant compounds. Categories
overlap and are diagnostic, not new labels.

![Cross-CYP relationships](figures/cross_cyp_pairwise_pic50.png)

## 5. Experimental uncertainty

Median 95% interval widths are
{', '.join(f'{cyp}={uncertainty.loc[(uncertainty.CYP==cyp) & (uncertainty.uncertainty_metric=="CI_width"),"median"].iloc[0]:.3f}' for cyp in CYPS)}.
Interval width is strongly anticorrelated with activity: Spearman rho is
{', '.join(f'{cyp}={uncertainty_rel.loc[(uncertainty_rel.CYP==cyp) & (uncertainty_rel.uncertainty_metric=="CI_width") & (uncertainty_rel.variable=="pIC50"),"spearman"].iloc[0]:.3f}' for cyp in CYPS)}.
Uncertainty is therefore not independent noise; it partly encodes assay dynamic
range.

Raw inverse-variance weighting reduces effective sample size to
{', '.join(f'{cyp}={percent(weights.loc[(weights.CYP==cyp) & (weights.weighting=="inverse_variance"),"effective_sample_fraction"].iloc[0])}' for cyp in CYPS)}.
The official interval-aware loss is the primary future experiment. If weights are
tested, clip them within each training fold; do not use raw inverse variance
uncritically. A heteroscedastic head is reasonable only as a matched ablation because
the supplied uncertainty is highly activity-dependent. Replicate counts are not
available, so no uncertainty-versus-replicate analysis is claimed.

![Measurement uncertainty](figures/measurement_uncertainty.png)

## 6. ECFP6 activity cliffs

Every unique measured pair was evaluated exactly: 996,166 CYP1A2, 824,970 CYP2C9,
1,113,778 CYP2D6, and 2,724,945 CYP3A4 pairs.

{markdown_table(['CYP','similarity','pairs','median delta','delta >=1 N','delta >=1 %','delta >=2 N'], cliff_rows)}

At high similarity, counts for CYP1A2/CYP2C9 are small, so their percentages are
unstable. CYP3A4 provides the strongest evidence: median delta falls with similarity,
yet meaningful same- and different-scaffold cliffs persist. Increasing structural
similarity implies smoother average SAR, not a smooth-label guarantee. See the full
same-scaffold stratification in
[`activity_cliff_threshold_statistics.csv`](tables/activity_cliff_threshold_statistics.csv).

![Activity cliffs](figures/activity_cliffs_ecfp6_hexbin.png)

![Activity cliff examples](figures/activity_cliff_examples.png)

## 7. Scaffold structure

The 6,147-molecule training union contains {int(scaffold.loc['train_union','unique_scaffolds']):,}
scaffolds; {percent(scaffold.loc['train_union','singleton_scaffold_fraction'])} of
scaffold identities are singletons. Test has {int(scaffold.loc['test','unique_scaffolds'])}
scaffolds and {percent(scaffold.loc['test','singleton_scaffold_fraction'])} singleton
scaffolds. Only {int(scaffold.loc['test','molecules_with_scaffold_in_other_split'])}/750
test molecules and {int(scaffold.loc['test','scaffolds_shared_between_train_test'])}
test scaffold identities overlap training.

Per-CYP test molecule scaffold coverage is
{', '.join(f'{cyp}={percent(cyp_scaffold.loc[cyp,"test_molecule_scaffold_coverage_fraction"])}' for cyp in CYPS)}.
Within-scaffold variances and high-range/selective scaffold examples are retained in
CSV rather than reduced to cherry-picked pictures.

![Scaffold summary](figures/scaffold_coverage_and_activity_ranges.png)

## 8. CYP-specific chemical-space coverage

{markdown_table(['CYP','training N','within-train NN median','test NN median','test >=0.5','test >=0.6'], [[cyp,int(cyp_coverage.loc[cyp,'training_molecules']),cyp_coverage.loc[cyp,'within_training_NN_median'],cyp_coverage.loc[cyp,'test_to_CYP_training_NN_median'],percent(cyp_coverage.loc[cyp,'test_fraction_ge_0.5']),percent(cyp_coverage.loc[cyp,'test_fraction_ge_0.6'])] for cyp in CYPS])}

CYP2D6 is the weakest domain by test median and exact scaffold coverage; CYP3A4 is the
best covered, consistent with its larger labeled set. This must be evaluated per CYP,
not inferred from the 6,147-molecule union.

![Per-CYP test coverage](figures/per_cyp_test_chemical_space_coverage.png)

## 9. Auxiliary assays

{markdown_table(['assay relationship','CYP','paired N','Spearman'], aux_rows)}

Single-concentration data are at one concentration per CYP in this snapshot and are
strongly monotonic with direct pIC50. Emax is weaker and strongly CYP-dependent;
CYP2D6 has the opposite correlation sign from the fluorescence CYPs, consistent with
its different assay readout and requiring separate scaling/heads. Direct and
TDI-condition pIC50 are very strongly correlated, but their shift is the TDI-defining
signal.

After rank-residual control for direct pIC50, TDI-condition Emax retains association
with scored TDI labels for CYP2D6 (partial rho=
{partial.loc[(partial.CYP=='CYP2D6') & (partial.variable=='tdi_emax'),'partial_spearman'].iloc[0]:.3f})
and CYP3A4 ({partial.loc[(partial.CYP=='CYP3A4') & (partial.variable=='tdi_emax'),'partial_spearman'].iloc[0]:.3f}).
This justifies a leakage-controlled multi-assay/transfer experiment, not blind
concatenation of assays as rows.

![Single concentration and pIC50](figures/single_concentration_vs_pic50.png)

![Emax and pIC50](figures/emax_vs_pic50.png)

## 10. Representation complementarity

Similarity values were never compared directly across descriptor types. On the same
750 test queries, ECFP6 shares nearest-neighbor identity with USR for
{percent(nearest_overlap[('ECFP6','USR')])}, USRCAT for
{percent(nearest_overlap[('ECFP6','USRCAT')])}, pmapper count for
{percent(nearest_overlap[('ECFP6','pmapper_count')])}, and hashed pmapper for
{percent(nearest_overlap[('ECFP6','pmapper_bit')])}. Across 20,000 fixed-seed
test/train pairs, ECFP6 similarity-rank correlations are respectively
{pair_rank[('ECFP6','USR')]:.3f}, {pair_rank[('ECFP6','USRCAT')]:.3f},
{pair_rank[('ECFP6','pmapper_count')]:.3f}, and
{pair_rank[('ECFP6','pmapper_bit')]:.3f}.

USR remains over-permissive (nearest median {representation.loc['USR','nearest_median']:.3f}).
USRCAT is more discriminating (median {representation.loc['USRCAT','nearest_median']:.3f})
but conformer-sensitive. Hashed pmapper is invalid as a primary representation:
{int(saturation.loc['train_fully_saturated_2048_bits','count'])} train molecules set all
2,048 bits, and these fully saturated structures are selected for
{int(saturation.loc['test_NNs_that_are_fully_saturated','count'])}/750 test queries.
Single-conformer embedding succeeded for {int(single_conformer['success'].sum()):,}/
{len(single_conformer):,} molecules; both failures are training molecules and are
retained with their error messages in
[`single_conformer_3d_descriptors.csv`](tables/single_conformer_3d_descriptors.csv).

![Representation complementarity](figures/representation_complementarity.png)

## 11. Multi-conformer sensitivity

The deterministic subset contains 40 test queries stratified over four ECFP6
coverage bands and {int((conformer_metadata['split']=='train').sum())} candidate train
references: single-representation NNs plus fixed-seed random references. All received
five conformers. It is deliberately manageable and does not claim exhaustive
full-training-space 3D neighbors.

{markdown_table(['representation','queries','NN changed','median unique NNs','median modal fraction','median best-score SD','p95 best-score SD'], [[name,int(conformer.loc[name,'queries']),percent(conformer.loc[name,'NN_changed_fraction']),conformer.loc[name,'median_unique_NNs'],conformer.loc[name,'median_modal_neighbor_fraction'],conformer.loc[name,'median_best_score_SD'],conformer.loc[name,'p95_best_score_SD']] for name in ['USR','USRCAT','pmapper_count']])}

Single-conformer 3D descriptors are not sufficiently stable for primary modeling.
If retained later, use deterministic conformer ensembles with an aggregation rule
chosen inside CV.

![Conformer sensitivity](figures/conformer_sensitivity.png)

## 12. Candidate CV splits and applicability domain

The ECFP schemes use exact threshold-graph connected components, an order-independent
alternative to Butina series clustering. At thresholds 0.5/0.6 they produce
{int(cluster.loc['ecfp_component_0.5','groups']):,}/
{int(cluster.loc['ecfp_component_0.6','groups']):,} groups; their largest components
contain only {int(cluster.loc['ecfp_component_0.5','largest_group'])}/
{int(cluster.loc['ecfp_component_0.6','largest_group'])} molecules, so component
chaining is controlled.

{markdown_table(['scheme','fold min','fold max','validation NN','test NN same ref','val-test gap','scaffold seen','val NN <0.4','val NN >=0.6'], cv_rows)}

All validation schemes are chemically harder than test against the same fold-training
reference. Random CV is the closest but permits analog leakage. Scaffold CV forces
0% scaffold overlap although the actual test has {percent(scaffold.loc['test','molecules_with_scaffold_in_other_split']/750)}.
The recommended primary scheme is therefore ECFP component 0.6: it blocks pairs at
similarity >=0.6 across folds, is less extreme than 0.5, and preserves balanced fold
sizes. Report random CV as an interpolation ceiling and scaffold CV as a novelty
stress test; do not tune against whichever split gives the best score. Fold-training
mean predictions score approximately 1 under the exact official ST-RAE, providing an
implementation sanity check.

![CV novelty](figures/cv_split_novelty_and_fold_sizes.png)

## 13. Train/test physicochemical shifts

{markdown_table(['property','train mean','test mean','SMD test-train','KS','Wasserstein'], [[prop,shifts.loc[prop,'train_mean'],shifts.loc[prop,'test_mean'],shifts.loc[prop,'standardized_mean_difference_test_minus_train'],shifts.loc[prop,'ks_statistic'],shifts.loc[prop,'wasserstein_distance']] for prop in shifts.index])}

Fsp3 is the largest univariate shift (SMD
{shifts.loc['Fsp3','standardized_mean_difference_test_minus_train']:.3f}, KS
{shifts.loc['Fsp3','ks_statistic']:.3f}); test is also smaller by about
{shifts.loc['MW','train_mean']-shifts.loc['MW','test_mean']:.1f} Da. TPSA and aromatic
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

The scripts use seed {20260912}, stable sorting for output tables, the official
ST-RAE logic from tutorial revision
`858ae63ce79934113bccdb7fc65467de5f7b1935`, and project-local matplotlib/output
directories. No analytical figure is written under `/tmp`.
"""
    (ANALYSIS_DIR / "report.md").write_text(report, encoding="utf-8")
    print(f"Wrote {ANALYSIS_DIR / 'report.md'} and analysis_summary.csv", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
