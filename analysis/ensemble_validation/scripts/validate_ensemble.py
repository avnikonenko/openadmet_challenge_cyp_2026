"""Compare the fixed three-family CYP ensemble on two frozen CV schemes.

Run from the repository root. All inputs are saved OOF predictions; no model is fit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.metrics import regression_metrics  # noqa: E402
from src.features import MORGAN_GENERATOR  # noqa: E402


FAMILIES = ("transfer", "joint", "lightgbm")
MODELS = {
    "ecfp_cluster": {
        "transfer": "singleconc_to_pic50_full",
        "joint": "joint_multiassay",
        "lightgbm": "lightgbm_reference",
    },
    "scaffold": {
        "transfer": "singleconc_to_pic50_scaffold",
        "joint": "joint_multiassay_scaffold",
        "lightgbm": "lightgbm_reference_scaffold",
    },
}
ASSIGNMENT_COLUMNS = {"ecfp_cluster": "ecfp_component_0.6", "scaffold": "scaffold_group"}
KEY = ["molecule_id", "fold", "CYP"]
TRUTH = ["canonical_smiles", "y_true", "lower", "upper"]


def read_ensemble_inputs(results_root: Path, scheme: str) -> pd.DataFrame:
    merged = None
    expected_seeds = {1, 2, 3}
    for family in FAMILIES:
        path = results_root / MODELS[scheme][family] / "evaluation" / "oof_predictions.csv"
        frame = pd.read_csv(path)
        needed = KEY + TRUTH + ["seed", "y_pred", "prediction_scale"]
        missing = set(needed) - set(frame)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        if frame[KEY + ["seed"]].duplicated().any():
            raise ValueError(f"{path}: duplicate molecule/fold/CYP/seed rows")
        if set(frame.seed) != expected_seeds or set(frame.fold) != set(range(5)):
            raise ValueError(f"{path}: expected folds 0-4 and seeds 1-3")
        if set(frame.prediction_scale) != {"original"}:
            raise ValueError(f"{path}: predictions must be on the original pIC50 scale")
        for seed, part in frame.groupby("seed"):
            if set(part[KEY].itertuples(index=False, name=None)) != set(
                frame.loc[frame.seed.eq(1), KEY].itertuples(index=False, name=None)
            ):
                raise ValueError(f"{path}: seed {seed} has different OOF coverage")
        if not np.isfinite(frame[["y_true", "y_pred", "lower", "upper"]].to_numpy()).all():
            raise ValueError(f"{path}: non-finite OOF values")
        for column in TRUTH:
            if frame.groupby(KEY, sort=False)[column].nunique(dropna=False).gt(1).any():
                raise ValueError(f"{path}: {column} changes between seeds")
        average = frame.groupby(KEY, as_index=False, sort=True).agg(
            **{family: ("y_pred", "mean")},
            **{column: (column, "first") for column in TRUTH},
        )
        if merged is None:
            merged = average
        else:
            if len(merged) != len(average):
                raise ValueError(f"{path}: OOF row count differs between families")
            merged = merged.merge(average, on=KEY, how="inner", validate="one_to_one", suffixes=("", "_other"))
            if len(merged) != len(average):
                raise ValueError(f"{path}: OOF keys differ between families")
            for column in TRUTH:
                other = merged.pop(f"{column}_other")
                equal = merged[column].eq(other) if column == "canonical_smiles" else np.isclose(
                    merged[column], other, rtol=0, atol=1e-5
                )
                if not np.asarray(equal).all():
                    raise ValueError(f"{path}: {column} differs between families")
    assert merged is not None
    merged["ensemble"] = merged[list(FAMILIES)].mean(axis=1)
    merged["scheme"] = scheme
    return merged


def add_applicability(
    frame: pd.DataFrame, tables_dir: Path, results_root: Path, scheme: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignments = pd.read_csv(tables_dir / "cv_fold_assignments.csv")
    column = ASSIGNMENT_COLUMNS[scheme]
    assignment = assignments[["Molecule_Name", column, "canonical_smiles", "scaffold_smiles"]].rename(
        columns={"Molecule_Name": "molecule_id", column: "assigned_fold", "canonical_smiles": "assigned_smiles"}
    )
    molecules = frame[["molecule_id", "fold", "canonical_smiles"]].drop_duplicates()
    joined = molecules.merge(assignment, on="molecule_id", how="left", validate="one_to_one")
    if joined.assigned_fold.isna().any() or not joined.fold.eq(joined.assigned_fold).all():
        raise ValueError(f"{scheme}: saved predictions disagree with frozen fold assignments")
    if not joined.canonical_smiles.eq(joined.assigned_smiles).all():
        raise ValueError(f"{scheme}: canonical SMILES disagree with frozen assignments")
    by_id = assignment.set_index("molecule_id")
    fingerprints = {}

    def fingerprint(smiles: str):
        if smiles not in fingerprints:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid saved canonical SMILES: {smiles}")
            fingerprints[smiles] = MORGAN_GENERATOR.GetFingerprint(mol)
        return fingerprints[smiles]

    domain_rows = []
    for fold in range(5):
        split_path = results_root / MODELS[scheme]["transfer"] / f"fold{fold}/seed1/split_molecules.csv"
        split = pd.read_csv(split_path)
        if split.Molecule_Name.duplicated().any() or set(split.role) != {"train", "validation"}:
            raise ValueError(f"{split_path}: invalid split molecule roles or duplicate IDs")
        if set(split.Molecule_Name) != set(frame.molecule_id):
            raise ValueError(f"{split_path}: missing or extra molecules relative to OOF predictions")
        training = split.loc[split.role.eq("train")]
        validation = split.loc[split.role.eq("validation")]
        observed_validation = set(frame.loc[frame.fold.eq(fold), "molecule_id"])
        if set(validation.Molecule_Name) != observed_validation:
            raise ValueError(f"{split_path}: validation IDs differ from OOF predictions")
        if set(training.Molecule_Name) & observed_validation:
            raise ValueError(f"{split_path}: train/validation ID leakage")
        if set(training.canonical_smiles) & set(validation.canonical_smiles):
            raise ValueError(f"{split_path}: train/validation canonical molecule leakage")
        if not by_id.loc[training.Molecule_Name, "assigned_fold"].ne(fold).all():
            raise ValueError(f"{split_path}: training IDs disagree with frozen folds")
        if not split.canonical_smiles.eq(by_id.loc[split.Molecule_Name, "assigned_smiles"].to_numpy()).all():
            raise ValueError(f"{split_path}: canonical SMILES disagree with frozen assignments")
        train_fps = [fingerprint(s) for s in training.canonical_smiles]
        train_ids = training.Molecule_Name.to_list()
        train_scaffolds = set(by_id.loc[train_ids, "scaffold_smiles"])
        for molecule_id, smiles in validation[["Molecule_Name", "canonical_smiles"]].itertuples(index=False, name=None):
            similarities = DataStructs.BulkTanimotoSimilarity(fingerprint(smiles), train_fps)
            nearest = int(np.argmax(similarities))
            domain_rows.append({
                "scheme": scheme, "fold": fold, "molecule_id": molecule_id,
                "max_ECFP6_similarity": float(similarities[nearest]),
                "nearest_direct_training_name": train_ids[nearest],
                "scaffold_seen": by_id.loc[molecule_id, "scaffold_smiles"] in train_scaffolds,
            })
    domain = pd.DataFrame(domain_rows)
    if domain[["molecule_id", "fold"]].duplicated().any():
        raise ValueError(f"{scheme}: duplicate applicability-domain rows")
    merged = frame.merge(
        domain.drop(columns="scheme"), on=["molecule_id", "fold"], how="left", validate="many_to_one"
    )
    if merged.max_ECFP6_similarity.isna().any() or len(merged) != len(frame):
        raise ValueError(f"{scheme}: missing applicability-domain results")
    similarity = merged.max_ECFP6_similarity
    merged["similarity_bin"] = pd.cut(
        similarity, bins=[-0.001, 0.4, 0.7, 1.001], labels=["low <0.4", "moderate 0.4-<0.7", "high >=0.7"], right=False
    )
    if merged.similarity_bin.isna().any():
        raise ValueError(f"{scheme}: invalid ECFP6 similarity")
    return merged, domain


def add_cliff_examples(frame: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    """Flag members of the pre-model report's selected strong-cliff examples only."""
    examples = pd.read_csv(tables_dir / "activity_cliff_top_pairs.csv")
    selected = examples.loc[examples.similarity.ge(0.6) & examples.delta_pIC50.ge(1)]
    members = pd.concat(
        [selected[["CYP", "molecule_1"]].rename(columns={"molecule_1": "molecule_id"}),
         selected[["CYP", "molecule_2"]].rename(columns={"molecule_2": "molecule_id"})],
        ignore_index=True,
    ).drop_duplicates()
    members["selected_cliff_example"] = True
    result = frame.merge(members, on=["molecule_id", "CYP"], how="left", validate="many_to_one")
    result["selected_cliff_example"] = result.selected_cliff_example.fillna(False).astype(bool)
    return result


def metric_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = []
    folds = []
    for scheme, part in frame.groupby("scheme", sort=True):
        for family in (*FAMILIES, "ensemble"):
            measured = part.rename(columns={family: "y_pred"})
            per_cyp, macro = regression_metrics(measured)
            for row in per_cyp.to_dict("records"):
                metrics.append({"scheme": scheme, "model": family, **row})
            metrics.append({"scheme": scheme, "model": family, "CYP": "macro", "N": len(part), **{k.removeprefix("macro_"): v for k, v in macro.items()}})
            for fold, fold_rows in part.groupby("fold", sort=True):
                per_cyp, macro = regression_metrics(fold_rows.rename(columns={family: "y_pred"}))
                for row in per_cyp.to_dict("records"):
                    folds.append({"scheme": scheme, "model": family, "fold": fold, **row})
                folds.append({"scheme": scheme, "model": family, "fold": fold, "CYP": "macro", "N": len(fold_rows), **{k.removeprefix("macro_"): v for k, v in macro.items()}})
    return pd.DataFrame(metrics), pd.DataFrame(folds)


def bootstrap_differences(frame: pd.DataFrame, repeats: int, seed: int) -> pd.DataFrame:
    """Paired molecule bootstrap; all CYP observations of a molecule share one weight."""
    rng = np.random.default_rng(seed)
    rows = []
    for scheme, part in frame.groupby("scheme", sort=True):
        ids, molecule_index = np.unique(part.molecule_id.to_numpy(), return_inverse=True)
        arrays = {}
        for cyp, cyp_rows in part.groupby("CYP", sort=True):
            idx = cyp_rows.index.to_numpy()
            local = part.index.get_indexer(idx)
            arrays[cyp] = (
                molecule_index[local],
                cyp_rows.y_true.to_numpy(float),
                cyp_rows.lower.to_numpy(float),
                cyp_rows.upper.to_numpy(float),
                cyp_rows.transfer.to_numpy(float),
                cyp_rows.ensemble.to_numpy(float),
            )
        deltas = {cyp: np.empty(repeats) for cyp in arrays}
        deltas["macro"] = np.empty(repeats)
        for iteration in range(repeats):
            weights = np.bincount(rng.integers(len(ids), size=len(ids)), minlength=len(ids))
            current = []
            for cyp, (mol_idx, y, low, high, transfer, ensemble) in arrays.items():
                w = weights[mol_idx]
                mean = np.dot(w, y) / w.sum()
                denominator = np.dot(w, np.maximum(low - mean, 0) + np.maximum(mean - high, 0))
                if denominator <= 0:
                    raise ValueError(f"{scheme}/{cyp}: ST-RAE denominator is zero")
                loss = lambda pred: np.dot(w, np.maximum(low - pred, 0) + np.maximum(pred - high, 0)) / denominator
                value = loss(ensemble) - loss(transfer)
                deltas[cyp][iteration] = value
                current.append(value)
            deltas["macro"][iteration] = np.mean(current)
        for cyp, values in deltas.items():
            rows.append({"scheme": scheme, "CYP": cyp, "bootstrap_replicates": repeats, "delta_mean": values.mean(), "ci_low": np.quantile(values, 0.025), "ci_high": np.quantile(values, 0.975)})
    return pd.DataFrame(rows)


def error_by_domain(frame: pd.DataFrame) -> pd.DataFrame:
    result = []
    for scheme, part in frame.groupby("scheme", sort=True):
        for cyp, subset in part.groupby("CYP", sort=True):
            for category, column in [
                ("similarity", "similarity_bin"),
                ("scaffold_seen", "scaffold_seen"),
                ("selected_cliff_example", "selected_cliff_example"),
            ]:
                for level, rows in subset.groupby(column, observed=True, sort=True):
                    result.append({
                        "scheme": scheme, "CYP": cyp, "category": category, "level": str(level),
                        "N": len(rows), "molecules": rows.molecule_id.nunique(),
                        "transfer_MAE": abs(rows.transfer - rows.y_true).mean(),
                        "ensemble_MAE": abs(rows.ensemble - rows.y_true).mean(),
                        "transfer_large_error_fraction": (abs(rows.transfer - rows.y_true) >= 1).mean(),
                        "ensemble_large_error_fraction": (abs(rows.ensemble - rows.y_true) >= 1).mean(),
                    })
    return pd.DataFrame(result)


def make_figures(metrics: pd.DataFrame, domain: pd.DataFrame, figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, scheme in zip(axes, MODELS, strict=True):
        part = metrics[(metrics.scheme == scheme) & (metrics.CYP == "macro")]
        ax.bar(part.model, part.ST_RAE, color=["#377eb8", "#4daf4a", "#999999", "#e41a1c"])
        ax.set_title(scheme)
        ax.set_ylim(0, 0.85)
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylabel("OOF macro ST-RAE (lower is better)")
    fig.tight_layout()
    fig.savefig(figures / "ensemble_by_split.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    order = ["low <0.4", "moderate 0.4-<0.7", "high >=0.7"]
    for ax, scheme in zip(axes, MODELS, strict=True):
        part = domain[(domain.scheme == scheme) & (domain.category == "similarity")]
        aggregated = part.groupby("level", sort=False).apply(
            lambda group: pd.Series({
                "N": group.N.sum(),
                "transfer_MAE": np.average(group.transfer_MAE, weights=group.N),
                "ensemble_MAE": np.average(group.ensemble_MAE, weights=group.N),
            }), include_groups=False
        ).reindex(order)
        x = np.arange(len(order))
        ax.bar(x - 0.18, aggregated.transfer_MAE, 0.36, label="transfer")
        ax.bar(x + 0.18, aggregated.ensemble_MAE, 0.36, label="ensemble")
        ax.set_xticks(
            x,
            [f"{label}\nN={int(aggregated.loc[label, 'N']) if pd.notna(aggregated.loc[label, 'N']) else 0}" for label in order],
            rotation=15,
        )
        ax.set_title(scheme)
        ax.set_ylabel("MAE (pIC50)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(figures / "error_by_similarity.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--tables-dir", type=Path, default=ROOT / "analysis/cyp_pre_model/tables")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis/ensemble_validation")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("--bootstrap must be at least 100")
    frames = []
    domain_rows = []
    for scheme in MODELS:
        enriched, novelty = add_applicability(
            read_ensemble_inputs(args.results_root, scheme), args.tables_dir, args.results_root, scheme
        )
        frames.append(add_cliff_examples(enriched, args.tables_dir))
        domain_rows.append(novelty)
    combined = pd.concat(frames, ignore_index=True)
    metrics, folds = metric_tables(combined)
    bootstrap = bootstrap_differences(combined, args.bootstrap, args.seed)
    domain = error_by_domain(combined)
    combined["ensemble_abs_error"] = abs(combined.ensemble - combined.y_true)
    combined["transfer_abs_error"] = abs(combined.transfer - combined.y_true)
    top_errors = combined.sort_values(
        ["scheme", "CYP", "ensemble_abs_error", "molecule_id"],
        ascending=[True, True, False, True],
    ).groupby(["scheme", "CYP"], sort=False).head(20)
    tables = args.output_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    combined.to_csv(tables / "oof_ensemble_diagnostics.csv", index=False)
    metrics.to_csv(tables / "ensemble_metrics.csv", index=False)
    folds.to_csv(tables / "ensemble_fold_metrics.csv", index=False)
    bootstrap.to_csv(tables / "ensemble_bootstrap_ci.csv", index=False)
    domain.to_csv(tables / "ensemble_error_by_domain.csv", index=False)
    pd.concat(domain_rows, ignore_index=True).to_csv(tables / "cv_direct_training_applicability.csv", index=False)
    top_errors.to_csv(tables / "ensemble_top_errors.csv", index=False)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    (tables / "analysis_metadata.json").write_text(json.dumps({
        "results_root": str(args.results_root.resolve()), "tables_dir": str(args.tables_dir.resolve()),
        "models": MODELS, "bootstrap_replicates": args.bootstrap, "bootstrap_seed": args.seed,
        "source_code_revision": revision,
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "aggregation": "mean across seeds within family, then equal-weight mean across families",
        "similarity_reference": "saved direct-pIC50 training molecules per fold; Morgan radius 3, 2048 bits",
        "cliff_flag": "membership in selected pre-model top-pair examples with ECFP6 >=0.6 and delta pIC50 >=1; not an exhaustive cliff census",
    }, indent=2) + "\n")
    make_figures(metrics, domain, args.output_dir / "figures")
    primary = metrics[(metrics.scheme == "ecfp_cluster") & (metrics.model == "ensemble") & (metrics.CYP == "macro")].ST_RAE.iloc[0]
    print(f"ECFP OOF ensemble macro ST-RAE: {primary:.6f}")
    if abs(primary - 0.6500201247434944) > 1e-6:
        raise ValueError("Primary ensemble does not reproduce the saved 0.650020 reference")
    print(metrics.loc[metrics.CYP.eq("macro"), ["scheme", "model", "ST_RAE", "MAE"]].to_string(index=False))


if __name__ == "__main__":
    main()
