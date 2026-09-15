#!/usr/bin/env python3
"""Core integrity, target, missingness, uncertainty, scaffold, and assay EDA."""

from __future__ import annotations

import hashlib
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from common import (
    CYPS,
    DATA_DIR,
    DIRECT_TARGETS,
    FIGURES_DIR,
    SEED,
    TABLES_DIR,
    add_structure_features,
    distribution_shift,
    ensure_output_dirs,
    environment_table,
    load_sources,
    save_figure,
    standardized_mean_difference,
    write_csv,
)

import matplotlib.pyplot as plt


PROPERTIES = (
    "MW",
    "cLogP",
    "TPSA",
    "Fsp3",
    "HBD",
    "HBA",
    "RotatableBonds",
    "AromaticRings",
)
IDENTITY_COLUMNS = (
    "SMILES",
    "canonical_smiles",
    "nonisomeric_smiles",
    "fragment_parent_smiles",
    "charge_parent_smiles",
    "tautomer_parent_smiles",
)


def safe_correlation(x: pd.Series, y: pd.Series, method: str) -> tuple[int, float, float]:
    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return len(pair), np.nan, np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = stats.pearsonr(pair.iloc[:, 0], pair.iloc[:, 1]) if method == "pearson" else stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
    return len(pair), float(result.statistic), float(result.pvalue)


def identity_group_rows(frame: pd.DataFrame, source: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    valid = frame.loc[frame["valid"]].copy()
    for column in IDENTITY_COLUMNS:
        for identity, group in valid.dropna(subset=[column]).groupby(column, sort=True):
            if len(group) < 2:
                continue
            rows.append(
                {
                    "source": source,
                    "identity_level": column,
                    "identity": identity,
                    "row_count": len(group),
                    "unique_names": group["Molecule_Name"].nunique(),
                    "unique_raw_smiles": group["SMILES"].nunique(),
                    "unique_canonical_smiles": group["canonical_smiles"].nunique(),
                    "molecule_names": "|".join(sorted(group["Molecule_Name"].astype(str).unique())),
                }
            )
    return rows


def source_measurement_duplicates(source: str, frame: pd.DataFrame) -> pd.DataFrame:
    target_columns: list[tuple[str, str]] = []
    if source == "single":
        work = frame.loc[frame["valid"]].copy()
        keys = ["canonical_smiles", "enzyme", "concentration_M"]
        rows = []
        for key, group in work.groupby(keys, dropna=False, sort=True):
            measured = group["log2fc_estimate"].dropna()
            if len(group) > 1:
                rows.append(
                    {
                        "source": source,
                        "canonical_smiles": key[0],
                        "assay": f"{key[1]}_single_concentration_{key[2]}",
                        "row_count": len(group),
                        "measured_count": len(measured),
                        "value_min": measured.min() if len(measured) else np.nan,
                        "value_max": measured.max() if len(measured) else np.nan,
                        "value_range": measured.max() - measured.min() if len(measured) else np.nan,
                        "conflict": bool(len(measured) > 1 and measured.max() - measured.min() > 1e-8),
                        "molecule_names": "|".join(sorted(group["Molecule_Name"].astype(str).unique())),
                    }
                )
        return pd.DataFrame(rows)

    for column in frame.columns:
        if (
            ("pIC50" in column or "EmaxVsPosCtrl" in column or column.endswith("_is_TDI"))
            and not column.endswith(("_conf_high", "_conf_low", "_std"))
        ):
            target_columns.append((column, column))
    rows = []
    valid = frame.loc[frame["valid"]]
    for assay, column in target_columns:
        for identity, group in valid.groupby("canonical_smiles", sort=True):
            if len(group) < 2:
                continue
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            if not len(values):
                continue
            value_range = float(values.max() - values.min()) if len(values) > 1 else 0.0
            rows.append(
                {
                    "source": source,
                    "canonical_smiles": identity,
                    "assay": assay,
                    "row_count": len(group),
                    "measured_count": len(values),
                    "value_min": float(values.min()),
                    "value_max": float(values.max()),
                    "value_range": value_range,
                    "conflict": bool(len(values) > 1 and value_range > 1e-8),
                    "molecule_names": "|".join(sorted(group["Molecule_Name"].astype(str).unique())),
                }
            )
    return pd.DataFrame(rows)


def run_integrity(
    sources: dict[str, pd.DataFrame], train: pd.DataFrame, test: pd.DataFrame
) -> None:
    provenance_rows = []
    for path in sorted(DATA_DIR.glob("*.csv")):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        provenance_rows.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    write_csv(pd.DataFrame(provenance_rows), "input_file_hashes.csv", ["file"])

    source_unique = pd.concat(
        [
            frame.drop_duplicates("Molecule_Name", keep="first")
            .loc[:, ["Molecule_Name", "canonical_smiles"]]
            .assign(source=source)
            for source, frame in sources.items()
        ],
        ignore_index=True,
    )
    union_counts = [
        {
            "stage": f"{source}: input rows",
            "count": len(frame),
        }
        for source, frame in sources.items()
    ]
    union_counts.extend(
        {
            "stage": f"{source}: unique Molecule_Name rows",
            "count": frame["Molecule_Name"].nunique(),
        }
        for source, frame in sources.items()
    )
    union_counts.extend(
        [
            {"stage": "concatenated source-unique rows", "count": len(source_unique)},
            {
                "stage": "cross-source repeated Molecule_Name rows removed",
                "count": len(source_unique) - source_unique["Molecule_Name"].nunique(),
            },
            {"stage": "final training union", "count": len(train)},
        ]
    )
    write_csv(pd.DataFrame(union_counts), "training_union_construction_counts.csv")

    cross_source_conflicts = []
    for name, group in source_unique.groupby("Molecule_Name", sort=True):
        identities = group["canonical_smiles"].dropna().unique()
        if len(identities) > 1:
            cross_source_conflicts.append(
                {
                    "Molecule_Name": name,
                    "source_count": group["source"].nunique(),
                    "canonical_identity_count": len(identities),
                    "sources": "|".join(sorted(group["source"].unique())),
                    "canonical_smiles": "|".join(sorted(identities)),
                }
            )
    write_csv(
        pd.DataFrame(
            cross_source_conflicts,
            columns=[
                "Molecule_Name",
                "source_count",
                "canonical_identity_count",
                "sources",
                "canonical_smiles",
            ],
        ),
        "cross_source_name_identity_conflicts.csv",
        ["Molecule_Name"],
    )

    cross_split_rows = []
    for identity_level in IDENTITY_COLUMNS:
        train_values = set(train[identity_level].dropna())
        test_values = set(test[identity_level].dropna())
        shared = train_values & test_values
        cross_split_rows.append(
            {
                "identity_level": identity_level,
                "shared_identities": len(shared),
                "train_molecules_with_shared_identity": int(train[identity_level].isin(shared).sum()),
                "test_molecules_with_shared_identity": int(test[identity_level].isin(shared).sum()),
            }
        )
    write_csv(
        pd.DataFrame(cross_split_rows),
        "cross_split_identity_overlap.csv",
        ["identity_level"],
    )

    frames = {**sources, "train_union": train, "test": test}
    inventory = []
    counts = []
    identity_rows: list[dict[str, object]] = []
    problem_rows = []
    for source, frame in frames.items():
        derived_columns = {
            "valid", "error", "standardization_error", "canonical_smiles",
            "nonisomeric_smiles", "fragment_parent_smiles", "charge_parent_smiles",
            "tautomer_parent_smiles", "fragment_count", "scaffold", "scaffold_atoms",
            "HeavyAtoms", *PROPERTIES,
        }
        original_columns = [column for column in frame.columns if column not in derived_columns]
        inventory.append(
            {
                "dataset": source,
                "rows": len(frame),
                "columns": len(original_columns),
                "missing_cells_original_columns": int(frame[original_columns].isna().sum().sum()),
                "unique_molecule_names": frame["Molecule_Name"].nunique(dropna=True),
                "unique_raw_smiles": frame["SMILES"].nunique(dropna=True),
                "rdkit_valid": int(frame["valid"].sum()),
                "invalid": int((~frame["valid"]).sum()),
                "multi_fragment": int((frame["fragment_count"].fillna(0) > 1).sum()),
                "standardization_errors": int(frame["standardization_error"].fillna("").ne("").sum()),
            }
        )
        valid = frame.loc[frame["valid"]]
        stage_values = [
            ("input rows", len(frame)),
            ("nonmissing SMILES", int(frame["SMILES"].notna().sum())),
            ("RDKit-valid rows", len(valid)),
            ("unique Molecule_Name among valid", valid["Molecule_Name"].nunique()),
            ("unique raw SMILES among valid", valid["SMILES"].nunique()),
            ("unique canonical isomeric molecules", valid["canonical_smiles"].nunique()),
            ("unique nonisomeric molecules", valid["nonisomeric_smiles"].nunique()),
            ("unique fragment parents", valid["fragment_parent_smiles"].nunique()),
            ("unique charge parents", valid["charge_parent_smiles"].nunique()),
            ("unique tautomer parents", valid["tautomer_parent_smiles"].nunique()),
        ]
        for stage, remaining in stage_values:
            counts.append({"dataset": source, "stage": stage, "count": int(remaining)})
        identity_rows.extend(identity_group_rows(frame, source))
        for row in frame.loc[(~frame["valid"]) | frame["standardization_error"].fillna("").ne("")].itertuples(index=False):
            problem_rows.append(
                {
                    "dataset": source,
                    "Molecule_Name": row.Molecule_Name,
                    "SMILES": row.SMILES,
                    "parse_error": row.error,
                    "standardization_error": row.standardization_error,
                }
            )

    write_csv(pd.DataFrame(inventory), "dataset_inventory.csv", ["dataset"])
    write_csv(pd.DataFrame(counts), "identity_resolution_counts.csv", ["dataset", "stage"])
    write_csv(pd.DataFrame(identity_rows), "duplicate_identity_groups.csv", ["source", "identity_level", "identity"])
    problem_frame = pd.DataFrame(
        problem_rows,
        columns=["dataset", "Molecule_Name", "SMILES", "parse_error", "standardization_error"],
    )
    write_csv(problem_frame, "molecule_problems.csv", ["dataset", "Molecule_Name"])

    duplicate_frames = [source_measurement_duplicates(name, frame) for name, frame in sources.items()]
    duplicate_frames = [frame for frame in duplicate_frames if not frame.empty]
    duplicate_measurements = pd.concat(duplicate_frames, ignore_index=True) if duplicate_frames else pd.DataFrame(
        columns=["source", "canonical_smiles", "assay", "row_count", "measured_count", "value_min", "value_max", "value_range", "conflict", "molecule_names"]
    )
    write_csv(duplicate_measurements, "duplicate_measurements.csv", ["source", "assay", "canonical_smiles"])

    # The direct columns appear in both direct and TDI files. Audit rather than count
    # them as independent replicates.
    direct = sources["direct"]
    tdi = sources["tdi"]
    merged = direct[["Molecule_Name", *DIRECT_TARGETS]].merge(
        tdi[["Molecule_Name", *DIRECT_TARGETS]], on="Molecule_Name", suffixes=("_direct_file", "_tdi_file"), how="inner", validate="one_to_one"
    )
    agreement = []
    conflicts = []
    for target in DIRECT_TARGETS:
        left = merged[f"{target}_direct_file"]
        right = merged[f"{target}_tdi_file"]
        paired = left.notna() & right.notna()
        diff = (left - right).abs()
        agreement.append(
            {
                "target": target,
                "shared_rows": len(merged),
                "paired_nonmissing": int(paired.sum()),
                "missingness_disagreements": int((left.isna() != right.isna()).sum()),
                "numeric_conflicts": int((paired & diff.gt(1e-8)).sum()),
                "max_absolute_difference": float(diff[paired].max()) if paired.any() else np.nan,
            }
        )
        bad = merged.loc[paired & diff.gt(1e-8), ["Molecule_Name", f"{target}_direct_file", f"{target}_tdi_file"]].copy()
        if len(bad):
            bad.insert(1, "target", target)
            bad = bad.rename(
                columns={
                    f"{target}_direct_file": "direct_file_value",
                    f"{target}_tdi_file": "tdi_file_value",
                }
            )
            conflicts.append(bad)
    write_csv(pd.DataFrame(agreement), "cross_file_measurement_agreement.csv", ["target"])
    conflict_frame = pd.concat(conflicts, ignore_index=True) if conflicts else pd.DataFrame(columns=["Molecule_Name", "target", "direct_file_value", "tdi_file_value"])
    write_csv(conflict_frame, "cross_file_measurement_conflicts.csv", ["target", "Molecule_Name"])


def run_target_distributions(direct: pd.DataFrame) -> pd.DataFrame:
    stats_rows = []
    outlier_rows = []
    for cyp, target in zip(CYPS, DIRECT_TARGETS):
        values = direct[target].dropna().astype(float)
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        stats_rows.append(
            {
                "CYP": cyp,
                "N": len(values),
                "mean": values.mean(),
                "median": values.median(),
                "SD": values.std(ddof=1),
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
                "min": values.min(),
                "max": values.max(),
                "skewness": stats.skew(values, bias=False),
                "tukey_lower": low,
                "tukey_upper": high,
                "tukey_outliers": int(((values < low) | (values > high)).sum()),
                "inactive_lt_4": int((values < 4).sum()),
                "inactive_fraction_lt_4": float((values < 4).mean()),
                "active_ge_4": int((values >= 4).sum()),
                "active_fraction_ge_4": float((values >= 4).mean()),
            }
        )
        marked = direct.loc[direct[target].notna() & ((direct[target] < low) | (direct[target] > high)), ["Molecule_Name", "canonical_smiles", "scaffold", target]].copy()
        marked.insert(0, "CYP", cyp)
        marked = marked.rename(columns={target: "pIC50"})
        marked["outlier_rule"] = "outside Q1-1.5*IQR or Q3+1.5*IQR"
        outlier_rows.append(marked)
    result = pd.DataFrame(stats_rows)
    write_csv(result, "target_distribution_summary.csv", ["CYP"])
    write_csv(pd.concat(outlier_rows, ignore_index=True), "target_outliers.csv", ["CYP", "pIC50", "Molecule_Name"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (cyp, target) in zip(axes.flat, zip(CYPS, DIRECT_TARGETS)):
        values = direct[target].dropna()
        sns.histplot(values, bins=30, kde=True, ax=ax, color="#277da1")
        ax.axvline(4, color="#d62828", linestyle="--", label="pIC50 = 4")
        ax.set(title=f"{cyp} direct inhibition (N={len(values)})", xlabel="pIC50")
        ax.legend()
    save_figure(fig, "target_pic50_distributions.png")

    long = direct.melt(value_vars=list(DIRECT_TARGETS), var_name="target", value_name="pIC50").dropna()
    long["CYP"] = long["target"].str.extract(r"^(CYP[^_]+)")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(data=long, x="CYP", y="pIC50", inner="quartile", cut=0, ax=ax)
    ax.axhline(4, color="#d62828", linestyle="--")
    ax.set(title="Direct-inhibition endpoint distributions", xlabel="", ylabel="pIC50")
    save_figure(fig, "target_distribution_comparison.png")
    return result


def run_missingness(direct: pd.DataFrame) -> None:
    available = direct[list(DIRECT_TARGETS)].notna()
    n_labels = available.sum(axis=1)
    counts = n_labels.value_counts().reindex(range(5), fill_value=0).rename_axis("number_of_CYP_labels").reset_index(name="molecules")
    write_csv(counts, "label_count_per_molecule.csv", ["number_of_CYP_labels"])

    patterns = available.copy()
    patterns.columns = CYPS
    patterns["pattern"] = patterns.apply(lambda row: "+".join(cyp for cyp in CYPS if row[cyp]) or "none", axis=1)
    pattern_counts = patterns["pattern"].value_counts().rename_axis("availability_pattern").reset_index(name="molecules")
    write_csv(pattern_counts, "label_availability_patterns.csv", ["molecules", "availability_pattern"])

    order = np.lexsort(tuple((~available[target]).to_numpy() for target in reversed(DIRECT_TARGETS)) + (-n_labels.to_numpy(),))
    matrix = available.iloc[order].to_numpy(dtype=int)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1]}, constrained_layout=True)
    axes[0].imshow(matrix, aspect="auto", interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    axes[0].set(xticks=range(4), xticklabels=CYPS, xlabel="Endpoint", ylabel="Molecules (sorted)", title="Direct-pIC50 availability")
    sns.barplot(data=pattern_counts.sort_values("molecules", ascending=False), y="availability_pattern", x="molecules", ax=axes[1], color="#4d908e")
    axes[1].set(title="Availability patterns", xlabel="Molecules", ylabel="")
    save_figure(fig, "missingness_and_assay_coverage.png")

    association_rows = []
    for target, cyp in zip(DIRECT_TARGETS, CYPS):
        observed = available[target]
        for prop in PROPERTIES:
            labeled = direct.loc[observed, prop].dropna()
            missing = direct.loc[~observed, prop].dropna()
            test = stats.mannwhitneyu(labeled, missing, alternative="two-sided") if len(labeled) and len(missing) else None
            association_rows.append(
                {
                    "analysis": "endpoint_missingness",
                    "CYP": cyp,
                    "property": prop,
                    "labeled_N": len(labeled),
                    "missing_N": len(missing),
                    "labeled_mean": labeled.mean(),
                    "missing_mean": missing.mean(),
                    "SMD_missing_minus_labeled": standardized_mean_difference(labeled, missing),
                    "mannwhitney_U": test.statistic if test else np.nan,
                    "pvalue": test.pvalue if test else np.nan,
                    "spearman_with_label_count": np.nan,
                    "spearman_pvalue": np.nan,
                }
            )
    for prop in PROPERTIES:
        corr = stats.spearmanr(direct[prop], n_labels, nan_policy="omit")
        association_rows.append(
            {
                "analysis": "number_of_labels",
                "CYP": "all",
                "property": prop,
                "labeled_N": len(direct),
                "missing_N": 0,
                "labeled_mean": direct[prop].mean(),
                "missing_mean": np.nan,
                "SMD_missing_minus_labeled": np.nan,
                "mannwhitney_U": np.nan,
                "pvalue": np.nan,
                "spearman_with_label_count": corr.statistic,
                "spearman_pvalue": corr.pvalue,
            }
        )
    write_csv(pd.DataFrame(association_rows), "missingness_property_associations.csv", ["analysis", "CYP", "property"])

    scaffold_rows = []
    scaffold = direct["scaffold"].where(direct["scaffold"].map(direct["scaffold"].value_counts()) >= 5, "<OTHER_LT5>")
    for target, cyp in zip(DIRECT_TARGETS, CYPS):
        contingency = pd.crosstab(scaffold, available[target])
        if contingency.shape[1] < 2:
            chi2 = pvalue = cramers_v = np.nan
        else:
            chi2, pvalue, _, _ = stats.chi2_contingency(contingency)
            n = contingency.to_numpy().sum()
            cramers_v = np.sqrt(chi2 / (n * max(min(contingency.shape) - 1, 1)))
        scaffold_rows.append({"CYP": cyp, "scaffold_groups_ge_5_plus_other": contingency.shape[0], "chi2": chi2, "pvalue": pvalue, "cramers_v": cramers_v})
    write_csv(pd.DataFrame(scaffold_rows), "missingness_scaffold_association.csv", ["CYP"])


def run_cross_cyp(direct: pd.DataFrame) -> None:
    correlations = []
    for left_cyp, right_cyp in combinations(CYPS, 2):
        left = f"{left_cyp}_pIC50_direct_inhibition"
        right = f"{right_cyp}_pIC50_direct_inhibition"
        n_p, pearson, pearson_p = safe_correlation(direct[left], direct[right], "pearson")
        n_s, spearman, spearman_p = safe_correlation(direct[left], direct[right], "spearman")
        correlations.append({"CYP_1": left_cyp, "CYP_2": right_cyp, "overlap_N": min(n_p, n_s), "pearson": pearson, "pearson_pvalue": pearson_p, "spearman": spearman, "spearman_pvalue": spearman_p})
    write_csv(pd.DataFrame(correlations), "cross_cyp_correlations.csv", ["CYP_1", "CYP_2"])

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for ax, (left_cyp, right_cyp) in zip(axes.flat, combinations(CYPS, 2)):
        left = f"{left_cyp}_pIC50_direct_inhibition"
        right = f"{right_cyp}_pIC50_direct_inhibition"
        paired = direct[[left, right]].dropna()
        ax.hexbin(paired[left], paired[right], gridsize=25, mincnt=1, cmap="viridis")
        ax.axvline(4, color="white", alpha=0.6, linestyle="--")
        ax.axhline(4, color="white", alpha=0.6, linestyle="--")
        rho = stats.spearmanr(paired[left], paired[right]).statistic if len(paired) > 2 else np.nan
        ax.set(title=f"N={len(paired)}, Spearman={rho:.2f}", xlabel=left_cyp, ylabel=right_cyp)
    save_figure(fig, "cross_cyp_pairwise_pic50.png")

    values = direct[list(DIRECT_TARGETS)]
    observed_n = values.notna().sum(axis=1)
    active_n = values.ge(4).sum(axis=1)
    strong_n = values.ge(5).sum(axis=1)
    activity_range = values.max(axis=1) - values.min(axis=1)
    broad = (observed_n >= 3) & (strong_n == observed_n)
    selective = (observed_n >= 2) & values.max(axis=1).ge(5) & values.min(axis=1).le(4) & activity_range.ge(1.5)
    discordant = (observed_n >= 2) & activity_range.ge(2)
    categories = direct.loc[broad | selective | discordant, ["Molecule_Name", "canonical_smiles", "scaffold", *DIRECT_TARGETS]].copy()
    categories["observed_CYPs"] = observed_n[broad | selective | discordant].to_numpy()
    categories["active_ge_4_CYPs"] = active_n[broad | selective | discordant].to_numpy()
    categories["activity_range"] = activity_range[broad | selective | discordant].to_numpy()
    categories["broad_strong_all_observed_ge_5"] = broad[broad | selective | discordant].to_numpy()
    categories["selective_max_ge_5_min_le_4_range_ge_1.5"] = selective[broad | selective | discordant].to_numpy()
    categories["discordant_range_ge_2"] = discordant[broad | selective | discordant].to_numpy()
    write_csv(categories, "cross_cyp_compound_categories.csv", ["discordant_range_ge_2", "activity_range", "Molecule_Name"])

    breadth = pd.DataFrame({"observed_CYPs": observed_n, "active_ge_4_CYPs": active_n, "strong_ge_5_CYPs": strong_n}).value_counts().reset_index(name="molecules")
    write_csv(breadth, "cross_cyp_activity_breadth.csv", ["observed_CYPs", "active_ge_4_CYPs", "strong_ge_5_CYPs"])


def run_uncertainty(direct: pd.DataFrame) -> None:
    long_rows = []
    summary_rows = []
    relation_rows = []
    weighting_rows = []
    noisy_rows = []
    for cyp, target in zip(CYPS, DIRECT_TARGETS):
        lower_col = f"{target}_conf_low"
        upper_col = f"{target}_conf_high"
        sd_col = f"{target}_std"
        subset = direct.loc[direct[target].notna(), ["Molecule_Name", "canonical_smiles", "scaffold", *PROPERTIES, "scaffold_atoms", target, lower_col, upper_col, sd_col]].copy()
        subset["CYP"] = cyp
        subset["pIC50"] = subset[target]
        subset["CI_width"] = subset[upper_col] - subset[lower_col]
        subset["SD"] = subset[sd_col]
        subset["invalid_CI_width"] = subset["CI_width"].notna() & subset["CI_width"].lt(0)
        long_rows.append(subset[["CYP", "Molecule_Name", "canonical_smiles", "scaffold", "pIC50", "CI_width", "SD", "invalid_CI_width", *PROPERTIES, "scaffold_atoms"]])

        for metric in ("CI_width", "SD"):
            values = subset[metric].dropna()
            q1, q3 = values.quantile([0.25, 0.75]) if len(values) else (np.nan, np.nan)
            iqr = q3 - q1
            cutoff = q3 + 1.5 * iqr
            summary_rows.append(
                {
                    "CYP": cyp,
                    "uncertainty_metric": metric,
                    "N": len(values),
                    "missing": int(subset[metric].isna().sum()),
                    "mean": values.mean(),
                    "median": values.median(),
                    "SD": values.std(ddof=1),
                    "Q1": q1,
                    "Q3": q3,
                    "IQR": iqr,
                    "min": values.min() if len(values) else np.nan,
                    "max": values.max() if len(values) else np.nan,
                    "noisy_cutoff_Q3_plus_1.5IQR": cutoff,
                    "unusually_noisy_count": int((values > cutoff).sum()) if len(values) else 0,
                }
            )
            marked = subset.loc[subset[metric].gt(cutoff), ["Molecule_Name", "canonical_smiles", "scaffold", "pIC50", metric]].copy()
            marked.insert(0, "CYP", cyp)
            marked.insert(1, "uncertainty_metric", metric)
            marked = marked.rename(columns={metric: "uncertainty_value"})
            marked["cutoff"] = cutoff
            noisy_rows.append(marked)

            for property_name in ("pIC50", *PROPERTIES, "scaffold_atoms"):
                n, coefficient, pvalue = safe_correlation(subset[metric], subset[property_name], "spearman")
                relation_rows.append({"CYP": cyp, "uncertainty_metric": metric, "variable": property_name, "N": n, "spearman": coefficient, "pvalue": pvalue})

        valid_sd = subset["SD"].dropna()
        valid_sd = valid_sd[valid_sd > 0]
        if len(valid_sd):
            raw_weight = 1.0 / np.square(valid_sd.to_numpy())
            lo, hi = np.quantile(raw_weight, [0.05, 0.95])
            clipped = np.clip(raw_weight, lo, hi)
            for label, weights in (("inverse_variance", raw_weight), ("clipped_5_95pct", clipped)):
                weighting_rows.append(
                    {
                        "CYP": cyp,
                        "weighting": label,
                        "N": len(weights),
                        "weight_min": weights.min(),
                        "weight_median": np.median(weights),
                        "weight_max": weights.max(),
                        "max_to_median_ratio": weights.max() / np.median(weights),
                        "effective_sample_size": np.square(weights.sum()) / np.square(weights).sum(),
                        "effective_sample_fraction": (np.square(weights.sum()) / np.square(weights).sum()) / len(weights),
                    }
                )

    uncertainty = pd.concat(long_rows, ignore_index=True)
    write_csv(uncertainty, "measurement_uncertainty_long.csv", ["CYP", "Molecule_Name"])
    write_csv(pd.DataFrame(summary_rows), "uncertainty_summary.csv", ["CYP", "uncertainty_metric"])
    write_csv(pd.DataFrame(relation_rows), "uncertainty_relationships.csv", ["CYP", "uncertainty_metric", "variable"])
    write_csv(pd.DataFrame(weighting_rows), "uncertainty_weighting_diagnostics.csv", ["CYP", "weighting"])
    noisy = pd.concat(noisy_rows, ignore_index=True) if noisy_rows else pd.DataFrame(columns=["CYP", "uncertainty_metric", "Molecule_Name", "uncertainty_value"])
    write_csv(noisy, "unusually_noisy_measurements.csv", ["CYP", "uncertainty_metric", "uncertainty_value", "Molecule_Name"])
    between_cyp_rows = []
    for metric in ("CI_width", "SD"):
        groups = [uncertainty.loc[uncertainty["CYP"] == cyp, metric].dropna() for cyp in CYPS]
        comparison = stats.kruskal(*groups)
        between_cyp_rows.append(
            {
                "uncertainty_metric": metric,
                "test": "Kruskal-Wallis",
                "statistic": comparison.statistic,
                "pvalue": comparison.pvalue,
                "CYP1A2_median": groups[0].median(),
                "CYP2C9_median": groups[1].median(),
                "CYP2D6_median": groups[2].median(),
                "CYP3A4_median": groups[3].median(),
            }
        )
    write_csv(pd.DataFrame(between_cyp_rows), "uncertainty_between_cyp_comparison.csv", ["uncertainty_metric"])

    assay_availability = pd.DataFrame(
        [
            {"field": "credible_interval_bounds", "available": True, "interpretation": "per fitted dose-response measurement"},
            {"field": "pIC50_std", "available": True, "interpretation": "per fitted dose-response measurement"},
            {"field": "standard_error", "available": False, "interpretation": "not supplied for direct-pIC50 fits"},
            {"field": "replicate_count", "available": False, "interpretation": "not supplied; cannot analyze uncertainty versus replicate count"},
        ]
    )
    write_csv(assay_availability, "uncertainty_field_availability.csv", ["field"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, metric in zip(axes, ("CI_width", "SD")):
        sns.violinplot(data=uncertainty, x="CYP", y=metric, cut=0, inner="quartile", ax=ax[0])
        ax[0].set(title=f"{metric} by CYP", xlabel="")
        sns.scatterplot(data=uncertainty, x="pIC50", y=metric, hue="CYP", s=10, alpha=0.35, ax=ax[1], legend=(metric == "CI_width"))
        ax[1].set(title=f"{metric} versus pIC50")
    save_figure(fig, "measurement_uncertainty.png")

    selected_props = ("MW", "cLogP", "TPSA", "Fsp3", "scaffold_atoms")
    fig, axes = plt.subplots(2, len(selected_props), figsize=(19, 7), constrained_layout=True)
    for row_index, metric in enumerate(("CI_width", "SD")):
        for ax, prop in zip(axes[row_index], selected_props):
            sns.scatterplot(data=uncertainty, x=prop, y=metric, hue="CYP", s=8, alpha=0.3, legend=False, ax=ax)
            ax.set_title(f"{metric} vs {prop}")
    save_figure(fig, "uncertainty_vs_molecular_properties.png")


def run_scaffolds(train: pd.DataFrame, test: pd.DataFrame, direct: pd.DataFrame) -> None:
    train_counts = train["scaffold"].value_counts()
    test_counts = test["scaffold"].value_counts()
    train_set = set(train_counts.index)
    test_set = set(test_counts.index)
    summary = pd.DataFrame(
        [
            {
                "split": "train_union",
                "molecules": len(train),
                "unique_scaffolds": len(train_counts),
                "singleton_scaffolds": int((train_counts == 1).sum()),
                "singleton_scaffold_fraction": float((train_counts == 1).mean()),
                "molecules_on_singleton_scaffolds": int(train["scaffold"].map(train_counts).eq(1).sum()),
                "scaffolds_shared_between_train_test": len(train_set & test_set),
                "molecules_with_scaffold_in_other_split": int(train["scaffold"].isin(test_set).sum()),
            },
            {
                "split": "test",
                "molecules": len(test),
                "unique_scaffolds": len(test_counts),
                "singleton_scaffolds": int((test_counts == 1).sum()),
                "singleton_scaffold_fraction": float((test_counts == 1).mean()),
                "molecules_on_singleton_scaffolds": int(test["scaffold"].map(test_counts).eq(1).sum()),
                "scaffolds_shared_between_train_test": len(train_set & test_set),
                "molecules_with_scaffold_in_other_split": int(test["scaffold"].isin(train_set).sum()),
            },
        ]
    )
    write_csv(summary, "scaffold_summary.csv", ["split"])

    size_rows = []
    for split, counts in (("train_union", train_counts), ("test", test_counts)):
        for scaffold, count in counts.items():
            size_rows.append({"split": split, "scaffold": scaffold, "molecules": count, "shared_train_test": scaffold in train_set and scaffold in test_set})
    write_csv(pd.DataFrame(size_rows), "scaffold_sizes.csv", ["split", "molecules", "scaffold"])

    coverage_rows = []
    variance_rows = []
    for cyp, target in zip(CYPS, DIRECT_TARGETS):
        subset = direct.loc[direct[target].notna()]
        counts = subset["scaffold"].value_counts()
        coverage_rows.append(
            {
                "CYP": cyp,
                "labeled_molecules": len(subset),
                "unique_scaffolds": len(counts),
                "singleton_scaffolds": int((counts == 1).sum()),
                "singleton_scaffold_fraction": float((counts == 1).mean()),
                "labeled_molecules_on_test_seen_scaffolds": int(subset["scaffold"].isin(test_set).sum()),
                "test_molecules_covered_by_CYP_scaffolds": int(test["scaffold"].isin(set(counts.index)).sum()),
                "test_molecule_scaffold_coverage_fraction": float(test["scaffold"].isin(set(counts.index)).mean()),
            }
        )
        for scaffold, group in subset.groupby("scaffold", sort=True):
            if len(group) < 2:
                continue
            values = group[target]
            variance_rows.append(
                {
                    "CYP": cyp,
                    "scaffold": scaffold,
                    "N": len(group),
                    "pIC50_mean": values.mean(),
                    "pIC50_SD": values.std(ddof=1),
                    "pIC50_min": values.min(),
                    "pIC50_max": values.max(),
                    "pIC50_range": values.max() - values.min(),
                    "molecule_names": "|".join(sorted(group["Molecule_Name"].astype(str))),
                }
            )
    write_csv(pd.DataFrame(coverage_rows), "per_cyp_scaffold_coverage.csv", ["CYP"])
    variances = pd.DataFrame(variance_rows)
    write_csv(variances, "within_scaffold_activity_variance.csv", ["CYP", "pIC50_range", "N", "scaffold"])

    scaffold_means = direct.groupby("scaffold")[list(DIRECT_TARGETS)].mean()
    scaffold_ns = direct.groupby("scaffold")[list(DIRECT_TARGETS)].count()
    selective_rows = []
    for scaffold in scaffold_means.index:
        eligible = scaffold_means.loc[scaffold].where(scaffold_ns.loc[scaffold] >= 2).dropna()
        if len(eligible) >= 2:
            selective_rows.append(
                {
                    "scaffold": scaffold,
                    "eligible_CYPs_with_at_least_2": len(eligible),
                    "max_mean_CYP": eligible.idxmax().split("_")[0],
                    "min_mean_CYP": eligible.idxmin().split("_")[0],
                    "mean_pIC50_range_across_CYPs": eligible.max() - eligible.min(),
                    "total_labeled_measurements": int(scaffold_ns.loc[scaffold].sum()),
                }
            )
    write_csv(pd.DataFrame(selective_rows), "scaffold_cyp_selectivity.csv", ["mean_pIC50_range_across_CYPs", "scaffold"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    sns.histplot(train_counts.clip(upper=20), bins=20, ax=axes[0], color="#277da1")
    axes[0].set(title="Train scaffold sizes (clipped at 20)", xlabel="Molecules per scaffold")
    top_variance = variances.sort_values(["pIC50_range", "N"], ascending=False).groupby("CYP", as_index=False).head(10)
    sns.scatterplot(data=top_variance, x="N", y="pIC50_range", hue="CYP", ax=axes[1])
    axes[1].set(title="Highest within-scaffold activity ranges", xlabel="Labeled molecules", ylabel="pIC50 range")
    save_figure(fig, "scaffold_coverage_and_activity_ranges.png")


def partial_spearman(data: pd.DataFrame, x: str, y: str, control: str) -> tuple[int, float, float]:
    subset = data[[x, y, control]].dropna()
    if len(subset) < 5 or any(subset[column].nunique() < 2 for column in (x, y, control)):
        return len(subset), np.nan, np.nan
    ranks = subset.rank(method="average")
    design = np.column_stack([np.ones(len(ranks)), ranks[control].to_numpy()])
    x_resid = ranks[x].to_numpy() - design @ np.linalg.lstsq(design, ranks[x].to_numpy(), rcond=None)[0]
    y_resid = ranks[y].to_numpy() - design @ np.linalg.lstsq(design, ranks[y].to_numpy(), rcond=None)[0]
    result = stats.pearsonr(x_resid, y_resid)
    return len(subset), float(result.statistic), float(result.pvalue)


def run_auxiliary(sources: dict[str, pd.DataFrame], direct: pd.DataFrame) -> None:
    relations = []
    overlap_rows = []
    label_associations = []
    partial_rows = []

    def add_relation(dataset: str, cyp: str, x_name: str, y_name: str, frame: pd.DataFrame) -> None:
        n_p, pearson, pearson_p = safe_correlation(frame[x_name], frame[y_name], "pearson")
        n_s, spearman, spearman_p = safe_correlation(frame[x_name], frame[y_name], "spearman")
        relations.append({"auxiliary_dataset": dataset, "CYP": cyp, "x": x_name, "y": y_name, "overlap_N": min(n_p, n_s), "pearson": pearson, "pearson_pvalue": pearson_p, "spearman": spearman, "spearman_pvalue": spearman_p})

    identifier = direct[["Molecule_Name", "canonical_smiles", *DIRECT_TARGETS]].copy()
    single = sources["single"].merge(identifier[["Molecule_Name", "canonical_smiles"]], on="Molecule_Name", suffixes=("", "_direct"), how="left", validate="many_to_one")
    single["canonical_match"] = single["canonical_smiles"] == single["canonical_smiles_direct"]
    concentrations = single.groupby("enzyme")["concentration_M"].agg(["count", "nunique", "min", "max"]).reset_index().rename(columns={"enzyme": "CYP"})
    write_csv(concentrations, "single_concentration_levels.csv", ["CYP"])
    single_merged = single.merge(direct[["Molecule_Name", *DIRECT_TARGETS]], on="Molecule_Name", how="inner", validate="many_to_one")
    for cyp, target in zip(CYPS, DIRECT_TARGETS):
        subset = single_merged.loc[single_merged["enzyme"] == cyp]
        overlap_rows.append({"auxiliary_dataset": "single_concentration", "CYP": cyp, "auxiliary_nonmissing": int(single.loc[(single["enzyme"] == cyp), "log2fc_estimate"].notna().sum()), "paired_with_direct_pIC50": int(subset[["log2fc_estimate", target]].dropna().shape[0]), "canonical_identity_mismatches_on_name_join": int((~single.loc[single["enzyme"] == cyp, "canonical_match"]).sum())})
        add_relation("single_concentration", cyp, "log2fc_estimate", target, subset)

    emax = sources["emax"].merge(identifier, on="Molecule_Name", suffixes=("_emax_file", "_direct"), how="left", validate="one_to_one")
    tdi = sources["tdi"]
    combined = emax.merge(tdi, on="Molecule_Name", suffixes=("_emax_file", "_tdi_file"), how="inner", validate="one_to_one")
    for cyp, direct_target in zip(CYPS, DIRECT_TARGETS):
        direct_emax = f"{cyp}_EmaxVsPosCtrl_direct_inhibition"
        tdi_emax = f"{cyp}_EmaxVsPosCtrl_TDI_condition"
        tdi_pic50 = f"{cyp}_pIC50_TDI_condition"
        direct_pic50_combined = f"{direct_target}_tdi_file"
        overlap_rows.append({"auxiliary_dataset": "Emax_direct", "CYP": cyp, "auxiliary_nonmissing": int(emax[direct_emax].notna().sum()), "paired_with_direct_pIC50": int(emax[[direct_emax, direct_target]].dropna().shape[0]), "canonical_identity_mismatches_on_name_join": int((emax["canonical_smiles_emax_file"] != emax["canonical_smiles_direct"]).fillna(True).sum())})
        add_relation("Emax_direct", cyp, direct_emax, direct_target, emax)
        add_relation("Emax_TDI", cyp, tdi_emax, tdi_pic50, combined)
        add_relation("TDI_vs_direct_pIC50", cyp, tdi_pic50, direct_pic50_combined, combined)

        label_col = f"{cyp}_is_TDI"
        label_candidates = [column for column in combined.columns if column.startswith(label_col)]
        if label_candidates:
            label = pd.to_numeric(combined[label_candidates[-1]], errors="coerce")
            subset = pd.DataFrame({"label": label, "direct_pIC50": combined[direct_pic50_combined], "tdi_pIC50": combined[tdi_pic50], "direct_emax": combined[direct_emax], "tdi_emax": combined[tdi_emax]})
            subset["emax_shift"] = subset["tdi_emax"] - subset["direct_emax"]
            subset["pic50_shift"] = subset["tdi_pIC50"] - subset["direct_pIC50"]
            for variable in ("direct_pIC50", "tdi_pIC50", "pic50_shift", "direct_emax", "tdi_emax", "emax_shift"):
                observed = subset[["label", variable]].dropna()
                positives = observed.loc[observed["label"] == 1, variable]
                negatives = observed.loc[observed["label"] == 0, variable]
                if len(positives) and len(negatives):
                    point = stats.pointbiserialr(observed["label"], observed[variable])
                    mann = stats.mannwhitneyu(positives, negatives, alternative="two-sided")
                    label_associations.append({"CYP": cyp, "variable": variable, "N": len(observed), "positive_N": len(positives), "negative_N": len(negatives), "positive_mean": positives.mean(), "negative_mean": negatives.mean(), "point_biserial": point.statistic, "point_biserial_pvalue": point.pvalue, "mannwhitney_U": mann.statistic, "mannwhitney_pvalue": mann.pvalue})
            for variable in ("direct_emax", "tdi_emax", "emax_shift"):
                n, coefficient, pvalue = partial_spearman(subset, variable, "label", "direct_pIC50")
                partial_rows.append({"CYP": cyp, "variable": variable, "outcome": "is_TDI", "control": "direct_pIC50", "N": n, "partial_spearman": coefficient, "pvalue": pvalue})

    write_csv(pd.DataFrame(overlap_rows), "auxiliary_assay_overlap.csv", ["auxiliary_dataset", "CYP"])
    write_csv(pd.DataFrame(relations), "auxiliary_assay_correlations.csv", ["auxiliary_dataset", "CYP", "x", "y"])
    write_csv(pd.DataFrame(label_associations), "tdi_label_associations.csv", ["CYP", "variable"])
    write_csv(pd.DataFrame(partial_rows), "emax_partial_associations.csv", ["CYP", "variable"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ax, (cyp, target) in zip(axes.flat, zip(CYPS, DIRECT_TARGETS)):
        subset = single_merged.loc[single_merged["enzyme"] == cyp]
        ax.hexbin(subset["log2fc_estimate"], subset[target], gridsize=30, mincnt=1, cmap="magma")
        rho = stats.spearmanr(subset["log2fc_estimate"], subset[target], nan_policy="omit").statistic
        ax.set(title=f"{cyp}: N={subset[[target, 'log2fc_estimate']].dropna().shape[0]}, rho={rho:.2f}", xlabel="Single-concentration log2 fold change", ylabel="Direct pIC50")
    save_figure(fig, "single_concentration_vs_pic50.png")

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    for column_index, (cyp, target) in enumerate(zip(CYPS, DIRECT_TARGETS)):
        direct_emax = f"{cyp}_EmaxVsPosCtrl_direct_inhibition"
        tdi_emax = f"{cyp}_EmaxVsPosCtrl_TDI_condition"
        tdi_pic50 = f"{cyp}_pIC50_TDI_condition"
        axes[0, column_index].hexbin(emax[direct_emax], emax[target], gridsize=25, mincnt=1, cmap="viridis")
        axes[0, column_index].set(title=f"{cyp} direct Emax", xlabel="Emax", ylabel="Direct pIC50")
        axes[1, column_index].hexbin(combined[tdi_emax], combined[tdi_pic50], gridsize=25, mincnt=1, cmap="viridis")
        axes[1, column_index].set(title=f"{cyp} TDI Emax", xlabel="Emax", ylabel="TDI-condition pIC50")
    save_figure(fig, "emax_vs_pic50.png")


def run_train_test_properties(train: pd.DataFrame, test: pd.DataFrame) -> None:
    rows = []
    for prop in PROPERTIES:
        shift = distribution_shift(train[prop], test[prop])
        rows.append({"property": prop, "train_N": train[prop].notna().sum(), "test_N": test[prop].notna().sum(), "train_mean": train[prop].mean(), "train_SD": train[prop].std(ddof=1), "train_median": train[prop].median(), "test_mean": test[prop].mean(), "test_SD": test[prop].std(ddof=1), "test_median": test[prop].median(), **shift})
    write_csv(pd.DataFrame(rows), "train_test_property_shift_statistics.csv", ["property"])

    plot_data = pd.concat([train.assign(split="train"), test.assign(split="test")], ignore_index=True)
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    for ax, prop in zip(axes.flat, PROPERTIES):
        sns.kdeplot(data=plot_data, x=prop, hue="split", common_norm=False, ax=ax)
        ax.set_title(prop)
    save_figure(fig, "train_test_property_distributions.png")


def main() -> int:
    ensure_output_dirs()
    print("Loading and canonicalizing all sources...", flush=True)
    sources, train, test = load_sources()
    print(f"Loaded train union={len(train)}, test={len(test)}", flush=True)
    train = add_structure_features(train)
    test = add_structure_features(test)
    direct = sources["direct"].merge(
        train[["Molecule_Name", "canonical_smiles", "scaffold", "scaffold_atoms", *PROPERTIES]],
        on="Molecule_Name",
        suffixes=("", "_union"),
        how="left",
        validate="one_to_one",
    )
    if not (direct["canonical_smiles"] == direct["canonical_smiles_union"]).all():
        raise ValueError("Direct-file and union canonical identities disagree")
    direct = direct.drop(columns=["canonical_smiles_union"])
    write_csv(environment_table(), "environment_versions.csv", ["package"])
    write_csv(train[["Molecule_Name", "SMILES", "canonical_smiles", "nonisomeric_smiles", "fragment_parent_smiles", "charge_parent_smiles", "tautomer_parent_smiles", "fragment_count", "valid", "error", "standardization_error", "scaffold", "scaffold_atoms", *PROPERTIES]], "train_molecule_identity_and_properties.csv", ["Molecule_Name"])
    write_csv(test[["Molecule_Name", "SMILES", "canonical_smiles", "nonisomeric_smiles", "fragment_parent_smiles", "charge_parent_smiles", "tautomer_parent_smiles", "fragment_count", "valid", "error", "standardization_error", "scaffold", "scaffold_atoms", *PROPERTIES]], "test_molecule_identity_and_properties.csv", ["Molecule_Name"])

    run_integrity(sources, train, test)
    run_target_distributions(direct)
    run_missingness(direct)
    run_cross_cyp(direct)
    run_uncertainty(direct)
    run_scaffolds(train, test, direct)
    run_auxiliary(sources, direct)
    run_train_test_properties(train, test)
    print(f"Core EDA complete: tables={TABLES_DIR}, figures={FIGURES_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
