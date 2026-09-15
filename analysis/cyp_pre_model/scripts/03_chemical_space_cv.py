#!/usr/bin/env python3
"""Per-CYP chemical coverage, applicability domain, and candidate CV splits."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import DataStructs
from scipy import stats
from sklearn.model_selection import GroupKFold, KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CYPS,
    DATA_DIR,
    DIRECT_TARGETS,
    MORGAN_GENERATOR,
    SEED,
)

# Import explicitly instead of exporting an analysis-specific list from common.
from common import (  # noqa: E402
    distribution_shift,
    ensure_output_dirs,
    nearest_similarities,
    official_st_rae,
    save_figure,
    valid_mol,
    within_nearest_similarities,
    write_csv,
)

import matplotlib.pyplot as plt  # noqa: E402


PROPERTIES = ("MW", "cLogP", "TPSA", "Fsp3", "HBD", "HBA", "RotatableBonds", "AromaticRings")
N_FOLDS = 5
CLUSTER_THRESHOLDS = (0.5, 0.6)


class UnionFind:
    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int32)
        self.rank = np.zeros(size, dtype=np.int8)

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[value] != value:
            next_value = int(self.parent[value])
            self.parent[value] = root
            value = next_value
        return root

    def union(self, left: int, right: int) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def threshold_components(fingerprints: list, threshold: float) -> np.ndarray:
    """Exact threshold-graph components, an order-independent Butina alternative."""

    union_find = UnionFind(len(fingerprints))
    for i in range(1, len(fingerprints)):
        similarities = np.asarray(DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[:i]))
        for j in np.flatnonzero(similarities >= threshold):
            union_find.union(i, int(j))
    roots = [union_find.find(i) for i in range(len(fingerprints))]
    root_to_group = {root: group for group, root in enumerate(sorted(set(roots)))}
    return np.asarray([root_to_group[root] for root in roots], dtype=np.int32)


def group_folds(groups: np.ndarray) -> np.ndarray:
    splitter = GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = np.full(len(groups), -1, dtype=np.int8)
    placeholder = np.zeros(len(groups))
    for fold, (_, validation) in enumerate(splitter.split(placeholder, groups=groups)):
        folds[validation] = fold
    if (folds < 0).any():
        raise RuntimeError("Some rows were not assigned to a group fold")
    return folds


def load_analysis_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    tables = Path(__file__).resolve().parents[1] / "tables"
    train = pd.read_csv(tables / "train_molecule_identity_and_properties.csv")
    test = pd.read_csv(tables / "test_molecule_identity_and_properties.csv")
    direct = pd.read_csv(DATA_DIR / "cyp-challenge-TRAIN_inhibition.csv")
    tdi = pd.read_csv(DATA_DIR / "cyp-challenge-TRAIN_TDI.csv")
    label_columns = ["Molecule_Name", *DIRECT_TARGETS]
    train = train.merge(direct[label_columns], on="Molecule_Name", how="left", validate="one_to_one")
    train = train.merge(tdi[["Molecule_Name", "CYP2D6_is_TDI", "CYP3A4_is_TDI"]], on="Molecule_Name", how="left", validate="one_to_one")
    if not train["valid"].all() or not test["valid"].all():
        raise ValueError("Invalid molecules require an explicit fingerprint fallback")
    return train, test


def per_cyp_coverage(train: pd.DataFrame, test: pd.DataFrame, train_fps: list, test_fps: list) -> None:
    property_rows = []
    nearest_rows = []
    summary_rows = []
    for cyp, target in zip(CYPS, DIRECT_TARGETS):
        mask = train[target].notna().to_numpy()
        indices = np.flatnonzero(mask)
        subset = train.loc[mask]
        subset_fps = [train_fps[i] for i in indices]
        within_scores, within_indices = within_nearest_similarities(subset_fps)
        test_scores, test_indices = nearest_similarities(test_fps, subset_fps)
        for prop in PROPERTIES:
            values = subset[prop].dropna()
            property_rows.append({"CYP": cyp, "property": prop, "N": len(values), "mean": values.mean(), "SD": values.std(ddof=1), "median": values.median(), "Q1": values.quantile(0.25), "Q3": values.quantile(0.75), "min": values.min(), "max": values.max()})
        summary_rows.append(
            {
                "CYP": cyp,
                "training_molecules": len(subset),
                "unique_canonical_training_molecules": subset["canonical_smiles"].nunique(),
                "within_training_NN_mean": np.mean(within_scores),
                "within_training_NN_median": np.median(within_scores),
                "within_training_NN_p05": np.quantile(within_scores, 0.05),
                "within_training_NN_p95": np.quantile(within_scores, 0.95),
                "test_to_CYP_training_NN_mean": np.mean(test_scores),
                "test_to_CYP_training_NN_median": np.median(test_scores),
                "test_to_CYP_training_NN_p05": np.quantile(test_scores, 0.05),
                "test_to_CYP_training_NN_p95": np.quantile(test_scores, 0.95),
                "test_fraction_ge_0.4": np.mean(test_scores >= 0.4),
                "test_fraction_ge_0.5": np.mean(test_scores >= 0.5),
                "test_fraction_ge_0.6": np.mean(test_scores >= 0.6),
                "test_fraction_ge_0.7": np.mean(test_scores >= 0.7),
            }
        )
        for query_index, score in enumerate(test_scores):
            neighbor_local = int(test_indices[query_index])
            nearest_rows.append(
                {
                    "split": "test",
                    "CYP_training_domain": cyp,
                    "Molecule_Name": test.iloc[query_index]["Molecule_Name"],
                    "max_ECFP6_similarity": float(score),
                    "nearest_training_name": subset.iloc[neighbor_local]["Molecule_Name"],
                    "scaffold_seen": bool(test.iloc[query_index]["scaffold"] in set(subset["scaffold"])),
                    "similarity_regime": "low_<0.4" if score < 0.4 else "moderate_0.4_0.6" if score < 0.6 else "high_ge_0.6",
                }
            )
    write_csv(pd.DataFrame(property_rows), "per_cyp_chemical_property_summary.csv", ["CYP", "property"])
    write_csv(pd.DataFrame(summary_rows), "per_cyp_chemical_space_coverage.csv", ["test_to_CYP_training_NN_median", "CYP"])
    applicability = pd.DataFrame(nearest_rows)
    write_csv(applicability, "test_per_cyp_applicability_domain.csv", ["CYP_training_domain", "Molecule_Name"])

    long = train.melt(id_vars=["Molecule_Name", *PROPERTIES], value_vars=list(DIRECT_TARGETS), var_name="target", value_name="pIC50").dropna(subset=["pIC50"])
    long["CYP"] = long["target"].str.extract(r"^(CYP[^_]+)")
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    for ax, prop in zip(axes.flat, PROPERTIES):
        sns.violinplot(data=long, x="CYP", y=prop, cut=0, inner="quartile", ax=ax)
        ax.set(title=prop, xlabel="")
    save_figure(fig, "per_cyp_chemical_property_distributions.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    sns.violinplot(data=applicability, x="CYP_training_domain", y="max_ECFP6_similarity", cut=0, inner="quartile", ax=axes[0])
    axes[0].set(title="Test coverage by CYP-specific training subset", xlabel="", ylabel="Test-to-training ECFP6 NN")
    regime = applicability.groupby(["CYP_training_domain", "similarity_regime"]).size().reset_index(name="molecules")
    regime["fraction"] = regime["molecules"] / 750
    sns.barplot(data=regime, x="CYP_training_domain", y="fraction", hue="similarity_regime", ax=axes[1])
    axes[1].set(title="Test applicability regimes", xlabel="", ylabel="Fraction")
    save_figure(fig, "per_cyp_test_chemical_space_coverage.png")


def build_fold_assignments(train: pd.DataFrame, train_fps: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignments = train[["Molecule_Name", "canonical_smiles", "scaffold"]].rename(columns={"scaffold": "scaffold_smiles"}).copy()
    random_folds = np.full(len(train), -1, dtype=np.int8)
    for fold, (_, validation) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=SEED).split(train)):
        random_folds[validation] = fold
    assignments["random"] = random_folds
    assignments["scaffold_group"] = group_folds(pd.factorize(train["scaffold"], sort=True)[0])

    cluster_rows = []
    for threshold in CLUSTER_THRESHOLDS:
        groups = threshold_components(train_fps, threshold)
        label = f"ecfp_component_{threshold:.1f}"
        assignments[label] = group_folds(groups)
        sizes = pd.Series(groups).value_counts()
        cluster_rows.append(
            {
                "scheme": label,
                "similarity_edge_threshold": threshold,
                "algorithm": "exact ECFP6 threshold-graph connected components",
                "groups": sizes.size,
                "singleton_groups": int((sizes == 1).sum()),
                "largest_group": int(sizes.max()),
                "largest_group_fraction": float(sizes.max() / len(train)),
                "median_group_size": float(sizes.median()),
            }
        )
    write_csv(assignments, "cv_fold_assignments.csv", ["Molecule_Name"])
    cluster_summary = pd.DataFrame(cluster_rows)
    write_csv(cluster_summary, "ecfp_cluster_summary.csv", ["similarity_edge_threshold"])
    return assignments, cluster_summary


def analyze_folds(train: pd.DataFrame, test: pd.DataFrame, train_fps: list, test_fps: list, assignments: pd.DataFrame) -> None:
    schemes = [column for column in assignments.columns if column not in {"Molecule_Name", "canonical_smiles", "scaffold_smiles"}]
    summary_rows = []
    label_rows = []
    missing_rows = []
    shift_rows = []
    applicability_rows = []

    for scheme in schemes:
        folds = assignments[scheme].to_numpy()
        for fold in range(N_FOLDS):
            validation_mask = folds == fold
            training_mask = ~validation_mask
            train_idx = np.flatnonzero(training_mask)
            val_idx = np.flatnonzero(validation_mask)
            reference_fps = [train_fps[i] for i in train_idx]
            query_fps = [train_fps[i] for i in val_idx]
            nn_scores, nn_indices = nearest_similarities(query_fps, reference_fps)
            test_nn_scores, _ = nearest_similarities(test_fps, reference_fps)
            training_scaffolds = set(train.loc[training_mask, "scaffold"])
            summary_rows.append(
                {
                    "scheme": scheme,
                    "fold": fold,
                    "train_molecules": int(training_mask.sum()),
                    "validation_molecules": int(validation_mask.sum()),
                    "validation_unique_scaffolds": train.loc[validation_mask, "scaffold"].nunique(),
                    "validation_scaffold_seen_fraction": float(train.loc[validation_mask, "scaffold"].isin(training_scaffolds).mean()),
                    "validation_canonical_seen_fraction": float(train.loc[validation_mask, "canonical_smiles"].isin(set(train.loc[training_mask, "canonical_smiles"])).mean()),
                    "validation_to_train_NN_mean": float(np.mean(nn_scores)),
                    "validation_to_train_NN_median": float(np.median(nn_scores)),
                    "validation_to_train_NN_p05": float(np.quantile(nn_scores, 0.05)),
                    "validation_to_train_NN_p95": float(np.quantile(nn_scores, 0.95)),
                    "validation_fraction_NN_lt_0.4": float(np.mean(nn_scores < 0.4)),
                    "validation_fraction_NN_ge_0.6": float(np.mean(nn_scores >= 0.6)),
                    "test_to_same_fold_train_NN_mean": float(np.mean(test_nn_scores)),
                    "test_to_same_fold_train_NN_median": float(np.median(test_nn_scores)),
                    "test_to_same_fold_train_NN_p05": float(np.quantile(test_nn_scores, 0.05)),
                    "test_to_same_fold_train_NN_p95": float(np.quantile(test_nn_scores, 0.95)),
                    "validation_minus_test_NN_median": float(np.median(nn_scores) - np.median(test_nn_scores)),
                }
            )
            for local_index, score in enumerate(nn_scores):
                original_index = val_idx[local_index]
                neighbor_index = train_idx[int(nn_indices[local_index])]
                applicability_rows.append(
                    {
                        "scheme": scheme,
                        "fold": fold,
                        "Molecule_Name": train.iloc[original_index]["Molecule_Name"],
                        "max_ECFP6_similarity": float(score),
                        "nearest_fold_training_name": train.iloc[neighbor_index]["Molecule_Name"],
                        "scaffold_seen": bool(train.iloc[original_index]["scaffold"] in training_scaffolds),
                        "similarity_regime": "low_<0.4" if score < 0.4 else "moderate_0.4_0.6" if score < 0.6 else "high_ge_0.6",
                    }
                )
            for prop in PROPERTIES:
                shift_rows.append({"scheme": scheme, "fold": fold, "property": prop, **distribution_shift(train.loc[training_mask, prop], train.loc[validation_mask, prop])})

            availability = train.loc[validation_mask, list(DIRECT_TARGETS)].notna().sum(axis=1)
            for label_count in range(5):
                missing_rows.append({"scheme": scheme, "fold": fold, "direct_CYP_labels": label_count, "validation_molecules": int((availability == label_count).sum())})
            for cyp, target in zip(CYPS, DIRECT_TARGETS):
                train_values = train.loc[training_mask, target].dropna()
                label_rows.append({"scheme": scheme, "fold": fold, "track": "regression", "endpoint": cyp, "train_N": len(train_values), "validation_N": int((validation_mask & train[target].notna()).sum()), "train_mean": train_values.mean(), "validation_mean": train.loc[validation_mask, target].mean(), "validation_median": train.loc[validation_mask, target].median(), "validation_positive_N": np.nan, "validation_positive_fraction": np.nan, "fold_mean_baseline_ST_RAE": np.nan})
            for endpoint in ("CYP2D6_is_TDI", "CYP3A4_is_TDI"):
                train_label = pd.to_numeric(train.loc[training_mask, endpoint], errors="coerce").dropna()
                val_label = pd.to_numeric(train.loc[validation_mask, endpoint], errors="coerce").dropna()
                label_rows.append({"scheme": scheme, "fold": fold, "track": "classification", "endpoint": endpoint.split("_")[0], "train_N": len(train_label), "validation_N": len(val_label), "train_mean": train_label.mean(), "validation_mean": val_label.mean(), "validation_median": val_label.median(), "validation_positive_N": int(val_label.sum()), "validation_positive_fraction": val_label.mean(), "fold_mean_baseline_ST_RAE": np.nan})

    write_csv(pd.DataFrame(summary_rows), "cv_split_summary.csv", ["scheme", "fold"])
    write_csv(pd.DataFrame(label_rows), "cv_fold_label_balance.csv", ["scheme", "fold", "track", "endpoint"])
    write_csv(pd.DataFrame(missing_rows), "cv_fold_missing_label_balance.csv", ["scheme", "fold", "direct_CYP_labels"])
    write_csv(pd.DataFrame(shift_rows), "cv_fold_property_shifts.csv", ["scheme", "fold", "property"])
    applicability = pd.DataFrame(applicability_rows)
    write_csv(applicability, "cv_applicability_domain.csv", ["scheme", "fold", "Molecule_Name"])

    summary = pd.DataFrame(summary_rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    sns.boxplot(data=applicability, x="scheme", y="max_ECFP6_similarity", ax=axes[0])
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set(title="Validation novelty by split scheme", xlabel="", ylabel="Validation-to-training ECFP6 NN")
    sns.barplot(data=summary, x="scheme", y="validation_molecules", hue="fold", ax=axes[1])
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set(title="Fold-size balance", xlabel="", ylabel="Validation molecules")
    save_figure(fig, "cv_split_novelty_and_fold_sizes.png")


def add_bounds_and_official_baseline(train: pd.DataFrame, assignments: pd.DataFrame) -> None:
    """Compute the exact official ST-RAE for fold-training-mean predictions."""

    direct = pd.read_csv(DATA_DIR / "cyp-challenge-TRAIN_inhibition.csv")
    bounded = train[["Molecule_Name"]].merge(direct, on="Molecule_Name", how="left", validate="one_to_one")
    rows = []
    schemes = [column for column in assignments.columns if column not in {"Molecule_Name", "canonical_smiles", "scaffold_smiles"}]
    for scheme in schemes:
        folds = assignments[scheme].to_numpy()
        for fold in range(N_FOLDS):
            for cyp, target in zip(CYPS, DIRECT_TARGETS):
                training = bounded.loc[(folds != fold) & bounded[target].notna()]
                validation = bounded.loc[(folds == fold) & bounded[target].notna()]
                prediction = np.repeat(training[target].mean(), len(validation))
                score = official_st_rae(validation[target], prediction, validation[f"{target}_conf_low"], validation[f"{target}_conf_high"])
                rows.append({"scheme": scheme, "fold": fold, "CYP": cyp, "train_N": len(training), "validation_N": len(validation), "training_mean_prediction": training[target].mean(), "official_ST_RAE": score, "official_tutorial_revision": "858ae63ce79934113bccdb7fc65467de5f7b1935"})
    write_csv(pd.DataFrame(rows), "cv_official_mean_baseline.csv", ["scheme", "fold", "CYP"])


def main() -> int:
    ensure_output_dirs()
    train, test = load_analysis_data()
    # Join confidence bounds for fold-level official metric diagnostics.
    direct = pd.read_csv(DATA_DIR / "cyp-challenge-TRAIN_inhibition.csv")
    bound_columns = ["Molecule_Name", *[f"{target}_{suffix}" for target in DIRECT_TARGETS for suffix in ("conf_low", "conf_high")]]
    train = train.merge(direct[bound_columns], on="Molecule_Name", how="left", validate="one_to_one")
    train_fps = [MORGAN_GENERATOR.GetFingerprint(valid_mol(smiles)) for smiles in train["canonical_smiles"]]
    test_fps = [MORGAN_GENERATOR.GetFingerprint(valid_mol(smiles)) for smiles in test["canonical_smiles"]]
    print("Computing per-CYP chemical-space coverage...", flush=True)
    per_cyp_coverage(train, test, train_fps, test_fps)
    print("Constructing random/scaffold/ECFP component folds...", flush=True)
    assignments, _ = build_fold_assignments(train, train_fps)
    analyze_folds(train, test, train_fps, test_fps, assignments)
    add_bounds_and_official_baseline(train, assignments)
    print("Chemical-space and CV analysis complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
