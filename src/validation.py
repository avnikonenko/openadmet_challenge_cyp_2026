from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import ANALYSIS_TABLES
from .data import DataBundle
from .features import molecule_graph
from .utils import json_dump


def _identity_sets(frame: pd.DataFrame, mask: pd.Series) -> tuple[set[str], set[str]]:
    selected = frame.loc[mask]
    return set(selected["Molecule_Name"]), set(selected["canonical_smiles"])


def leakage_checks(
    bundle: DataBundle, output_dir: Path, auxiliary_checkpoint: str | Path | None = None
) -> dict[str, object]:
    train_ids, train_structures = _identity_sets(bundle.frame, bundle.train_mask)
    validation_ids, validation_structures = _identity_sets(bundle.frame, bundle.validation_mask)
    test = pd.read_csv(ANALYSIS_TABLES / "test_molecule_identity_and_properties.csv")
    test_ids, test_structures = set(test["Molecule_Name"]), set(test["canonical_smiles"])
    violations = {
        "train_validation_id_overlap": sorted(train_ids & validation_ids),
        "train_validation_canonical_overlap": sorted(train_structures & validation_structures),
        "training_test_id_overlap": sorted(train_ids & test_ids),
        "training_test_canonical_overlap": sorted(train_structures & test_structures),
        "validation_test_id_overlap": sorted(validation_ids & test_ids),
        "validation_test_canonical_overlap": sorted(validation_structures & test_structures),
        "validation_auxiliary_id_overlap": [],
        "validation_auxiliary_canonical_overlap": [],
    }
    auxiliary_summary: dict[str, object] = {"checked": False}
    if auxiliary_checkpoint:
        checkpoint = Path(auxiliary_checkpoint).resolve()
        source_run = checkpoint.parent.parent
        split_path = source_run / "split_molecules.csv"
        metadata_path = source_run / "metadata.json"
        if not split_path.exists() or not metadata_path.exists():
            violations["auxiliary_audit_files_missing"] = [
                str(path) for path in (split_path, metadata_path) if not path.exists()
            ]
            auxiliary_summary = {
                "checked": False, "checkpoint": str(checkpoint), "source_run": str(source_run)
            }
        else:
            source_split = pd.read_csv(split_path)
            source_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if int(source_metadata.get("fold", -1)) != int(bundle.split_metadata["outer_fold"]):
                violations["auxiliary_fold_mismatch"] = [
                    f"source={source_metadata.get('fold')}, target={bundle.split_metadata['outer_fold']}"
                ]
            auxiliary_ids = set(source_split["Molecule_Name"])
            auxiliary_structures = set(source_split["canonical_smiles"])
            violations["validation_auxiliary_id_overlap"] = sorted(validation_ids & auxiliary_ids)
            violations["validation_auxiliary_canonical_overlap"] = sorted(
                validation_structures & auxiliary_structures
            )
            auxiliary_summary = {
                "checked": True,
                "checkpoint": str(checkpoint),
                "source_run": str(source_run),
                "source_molecules_used": len(auxiliary_ids),
            }
    failed = {name: values for name, values in violations.items() if values}
    summary = {
        "status": "failed" if failed else "passed",
        "train_molecules": len(train_ids),
        "validation_molecules": len(validation_ids),
        "test_molecules": len(test_ids),
        "auxiliary": auxiliary_summary,
        "violation_counts": {name: len(values) for name, values in violations.items()},
        "violations": failed,
    }
    json_dump(summary, output_dir / "leakage_check.json")
    if failed:
        details = ", ".join(f"{name}={len(values)}" for name, values in failed.items())
        raise ValueError(f"Leakage check failed: {details}; see {output_dir / 'leakage_check.json'}")
    return summary


def audit_featurization(
    bundle: DataBundle, output_dir: Path, policy: str = "fail"
) -> dict[str, object]:
    if policy not in {"fail", "skip_with_log"}:
        raise ValueError("featurization.policy must be fail or skip_with_log")
    requested = bundle.train_mask | bundle.validation_mask
    failures = []
    for index, row in bundle.frame.loc[requested].iterrows():
        try:
            molecule_graph(str(row["canonical_smiles"]))
        except Exception as exc:
            failures.append(
                {
                    "row_index": index,
                    "molecule_id": row["Molecule_Name"],
                    "canonical_smiles": row.get("canonical_smiles"),
                    "original_smiles": row.get("SMILES"),
                    "error": f"{type(exc).__name__}: {exc}",
                    "action": "failed_run" if policy == "fail" else "skipped",
                }
            )
    failure_frame = pd.DataFrame(
        failures,
        columns=["row_index", "molecule_id", "canonical_smiles", "original_smiles", "error", "action"],
    )
    failure_frame.to_csv(output_dir / "featurization_failures.csv", index=False)
    if failures and policy == "skip_with_log":
        failed_indices = {row["row_index"] for row in failures}
        bundle.train_mask &= ~bundle.frame.index.isin(failed_indices)
        bundle.validation_mask &= ~bundle.frame.index.isin(failed_indices)
    summary = {
        "policy": policy,
        "total_requested": int(requested.sum()),
        "successfully_featurized": int(requested.sum()) - len(failures),
        "failed": len(failures),
        "run_action": "failed" if failures and policy == "fail" else "continued",
        "failed_row_indices": [int(row["row_index"]) for row in failures],
    }
    json_dump(summary, output_dir / "featurization_summary.json")
    if failures and policy == "fail":
        raise ValueError(
            f"Featurization failed for {len(failures)} molecules; "
            f"see {output_dir / 'featurization_failures.csv'}"
        )
    return summary


def audit_prediction_featurization(frame: pd.DataFrame, output_dir: Path) -> dict[str, object]:
    failures = []
    for index, row in frame.iterrows():
        try:
            molecule_graph(str(row["canonical_smiles"]))
        except Exception as exc:
            failures.append(
                {
                    "row_index": index, "molecule_id": row["Molecule_Name"],
                    "canonical_smiles": row.get("canonical_smiles"),
                    "original_smiles": row.get("SMILES"),
                    "error": f"{type(exc).__name__}: {exc}", "action": "failed_run",
                }
            )
    pd.DataFrame(
        failures,
        columns=["row_index", "molecule_id", "canonical_smiles", "original_smiles", "error", "action"],
    ).to_csv(output_dir / "prediction_featurization_failures.csv", index=False)
    summary = {
        "policy": "fail", "total_requested": len(frame),
        "successfully_featurized": len(frame) - len(failures), "failed": len(failures),
    }
    json_dump(summary, output_dir / "prediction_featurization_summary.json")
    if failures:
        raise ValueError(
            f"Prediction featurization failed for {len(failures)} molecules; output completeness is mandatory"
        )
    return summary


def validate_predictions(
    frame: pd.DataFrame,
    expected_keys: Iterable[tuple[str, str]],
    required_tasks: Iterable[str],
    inverse_transformed: bool,
) -> dict[str, object]:
    required = {"molecule_id", "CYP", "y_pred"}
    if not required.issubset(frame):
        raise ValueError(f"Predictions lack columns: {sorted(required - set(frame))}")
    if frame.duplicated(["molecule_id", "CYP"]).any():
        raise ValueError("Predictions contain duplicate molecule/CYP rows")
    values = frame["y_pred"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Predictions contain NaN or infinite values")
    actual = set(map(tuple, frame[["molecule_id", "CYP"]].itertuples(index=False, name=None)))
    expected = set(expected_keys)
    if actual != expected:
        raise ValueError(
            f"Prediction keys differ from expectation: missing={len(expected - actual)}, "
            f"unexpected={len(actual - expected)}"
        )
    tasks = set(frame["CYP"])
    required_task_set = set(required_tasks)
    if tasks != required_task_set:
        raise ValueError(
            f"Prediction targets differ: missing={sorted(required_task_set - tasks)}, "
            f"unexpected={sorted(tasks - required_task_set)}"
        )
    if not inverse_transformed:
        raise ValueError("Predictions have not been inverse-transformed to the original target scale")
    if "prediction_scale" not in frame:
        raise ValueError("Predictions lack prediction_scale provenance")
    scales = set(frame["prediction_scale"].dropna().astype(str))
    if scales != {"original"} or frame["prediction_scale"].isna().any():
        raise ValueError(
            f"Predictions are not uniformly on the original target scale: {sorted(scales)}"
        )
    return {
        "rows": len(frame), "unique_molecules": frame["molecule_id"].nunique(),
        "tasks": sorted(tasks), "finite": True, "inverse_transformed": True,
    }
