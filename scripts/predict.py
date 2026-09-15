#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from models.multitask_gnn import batch_graphs
from models.transfer_model import validate_encoder_checkpoint
from src.constants import FEATURE_SCHEMA_VERSION
from src.data import load_test_frame
from src.features import lightgbm_features
from src.training import build_model
from src.utils import file_sha256, json_dump, load_config, normalize_device, seed_everything
from src.validation import audit_prediction_featurization, validate_predictions


def predict_dmpnn(run_dir: Path, frame: pd.DataFrame, device: str, batch_size: int, metadata: dict) -> pd.DataFrame:
    import torch

    checkpoint_path = run_dir / "checkpoints" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required_checkpoint_fields = {"target_normalization", "target_mean", "target_std", "model_state_dict"}
    missing = required_checkpoint_fields - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint compatibility metadata are missing: {sorted(missing)}")
    validate_encoder_checkpoint(checkpoint, checkpoint.get("model_config"), partial_load=False)
    model_config = checkpoint["model_config"]
    task_names = tuple(checkpoint["task_names"])
    model = build_model(task_names, {"model": model_config})
    model.load_state_dict(checkpoint["model_state_dict"])
    normalized = normalize_device(device)
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {normalized}")
    model.to(normalized).eval()
    means = np.asarray(checkpoint["target_mean"], dtype=float)
    stds = np.asarray(checkpoint["target_std"], dtype=float)
    if means.shape != (len(task_names),) or stds.shape != (len(task_names),):
        raise ValueError("Checkpoint target normalization dimensions do not match task_names")
    if not np.isfinite(means).all() or not np.isfinite(stds).all() or (stds <= 0).any():
        raise ValueError("Checkpoint target normalization statistics are invalid")
    normalization = checkpoint["target_normalization"]
    if normalization.get("mode") not in {"none", "zscore_per_target"}:
        raise ValueError("Checkpoint target normalization mode is unsupported")
    if not np.allclose(np.asarray(normalization.get("mean"), dtype=float), means) or not np.allclose(
        np.asarray(normalization.get("std"), dtype=float), stds
    ):
        raise ValueError("Checkpoint target normalization fields are internally inconsistent")
    values = []
    with torch.no_grad():
        for start in range(0, len(frame), batch_size):
            batch = frame.iloc[start : start + batch_size]
            output = model(batch_graphs(batch["canonical_smiles"].tolist()).to(normalized))
            values.append(output.cpu().numpy() * stds + means)
    predictions = np.concatenate(values)
    return long_test_predictions(frame, task_names, predictions, metadata)


def predict_lightgbm(run_dir: Path, frame: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    features, _ = lightgbm_features(frame)
    task_names, predictions = [], []
    for model_path in sorted((run_dir / "checkpoints").glob("CYP*.pkl")):
        sidecar = model_path.with_suffix(".meta.json")
        if not sidecar.exists():
            raise ValueError(f"LightGBM checkpoint metadata are missing: {sidecar}")
        checkpoint_metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        expected = {
            "architecture_type": "lightgbm",
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "model_code_version": "openadmet-lightgbm-v1",
            "task_names": [model_path.stem],
        }
        mismatches = {
            key: {"source": checkpoint_metadata.get(key), "expected": value}
            for key, value in expected.items() if checkpoint_metadata.get(key) != value
        }
        source_features = checkpoint_metadata.get("feature_config", {})
        for key, expected_value in (("morgan_radius", 3), ("morgan_bits", 2048)):
            if int(source_features.get(key, -1)) != expected_value:
                mismatches[f"feature_config.{key}"] = {
                    "source": source_features.get(key), "expected": expected_value,
                }
        if mismatches:
            raise ValueError(f"Incompatible LightGBM checkpoint metadata: {sidecar}")
        with model_path.open("rb") as handle:
            model = pickle.load(handle)
        task_names.append(model_path.stem)
        predictions.append(model.predict(features))
    if not predictions:
        raise FileNotFoundError(f"No LightGBM checkpoints in {run_dir / 'checkpoints'}")
    return long_test_predictions(frame, tuple(task_names), np.column_stack(predictions), metadata)


def _validate_existing_predictions(
    prediction_path: Path, run_dir: Path, frame: pd.DataFrame, output_dir: Path,
    task_names: tuple[str, ...],
) -> None:
    metadata_path = output_dir / "prediction_metadata.json"
    if not metadata_path.exists():
        raise ValueError(f"Existing predictions lack provenance: {metadata_path}")
    prediction_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if Path(prediction_metadata.get("run_dir", "")).resolve() != run_dir:
        raise ValueError("Existing predictions were produced from a different run directory")
    current_hashes = _checkpoint_hashes(run_dir)
    if prediction_metadata.get("checkpoint_hashes") != current_hashes:
        raise ValueError("Existing prediction checkpoint hashes are missing or no longer match")
    predictions = pd.read_csv(prediction_path)
    expected_keys = {
        (molecule_id, task)
        for molecule_id in frame["Molecule_Name"]
        for task in task_names
    }
    integrity = validate_predictions(predictions, expected_keys, task_names, True)
    json_dump(integrity, output_dir / "prediction_integrity.json")


def _checkpoint_hashes(run_dir: Path) -> dict[str, str]:
    checkpoints = run_dir / "checkpoints"
    paths = [checkpoints / "best.pt"] if (checkpoints / "best.pt").exists() else sorted(
        checkpoints.glob("CYP*.pkl")
    )
    if not paths:
        raise FileNotFoundError(f"No prediction checkpoint found in {checkpoints}")
    return {path.name: file_sha256(path) for path in paths}


def _checkpoint_task_names(run_dir: Path, model_type: str) -> tuple[str, ...]:
    if model_type == "lightgbm":
        tasks = []
        for model_path in sorted((run_dir / "checkpoints").glob("CYP*.pkl")):
            sidecar = model_path.with_suffix(".meta.json")
            if not sidecar.exists():
                raise ValueError(f"LightGBM checkpoint metadata are missing: {sidecar}")
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            if metadata.get("task_names") != [model_path.stem]:
                raise ValueError(f"LightGBM task metadata do not match checkpoint: {sidecar}")
            tasks.append(model_path.stem)
        if not tasks:
            raise FileNotFoundError(f"No LightGBM checkpoints in {run_dir / 'checkpoints'}")
        return tuple(tasks)
    import torch

    checkpoint_path = run_dir / "checkpoints" / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_encoder_checkpoint(checkpoint, checkpoint.get("model_config"), partial_load=False)
    tasks = tuple(checkpoint.get("task_names", ()))
    if not tasks:
        raise ValueError(f"Checkpoint has no task list: {checkpoint_path}")
    return tasks


def long_test_predictions(frame, task_names, values, metadata):
    rows = []
    for task_index, cyp in enumerate(task_names):
        for row_index, row in enumerate(frame.itertuples(index=False)):
            rows.append(
                {
                    "molecule_id": row.Molecule_Name,
                    "canonical_smiles": row.canonical_smiles,
                    "CYP": cyp,
                    "y_pred": float(values[row_index, task_index]),
                    "model": metadata.get("experiment", "unknown"),
                    "fold": metadata.get("fold"),
                    "seed": metadata.get("seed"),
                    "prediction_scale": "original",
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict blinded OpenADMET molecules from one completed run")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    run_dir = args.run_dir.resolve()
    config = load_config(args.config or run_dir / "config.yaml")
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Run metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "completed":
        raise ValueError(
            f"Refusing prediction from run status={metadata.get('status')!r}; a completed run is required"
        )
    if not (run_dir / "COMPLETED").exists():
        raise ValueError("Run metadata says completed but the COMPLETED marker is missing")
    output_dir = (args.output_dir or run_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "test_predictions.csv"
    seed_everything(args.seed)
    frame = load_test_frame()
    model_type = config.get("model", {}).get("type", "dmpnn")
    if prediction_path.exists():
        if args.resume:
            _validate_existing_predictions(
                prediction_path, run_dir, frame, output_dir,
                _checkpoint_task_names(run_dir, model_type),
            )
            print(f"Validated existing predictions: {prediction_path}")
            return 0
        raise FileExistsError(f"Predictions already exist: {prediction_path}; use --resume for validation")
    audit_prediction_featurization(frame, output_dir)
    predictions = (
        predict_lightgbm(run_dir, frame, metadata)
        if model_type == "lightgbm"
        else predict_dmpnn(run_dir, frame, args.device, args.batch_size, metadata)
    )
    task_names = tuple(sorted(predictions["CYP"].unique()))
    expected_keys = {
        (molecule_id, task)
        for molecule_id in frame["Molecule_Name"]
        for task in task_names
    }
    integrity = validate_predictions(predictions, expected_keys, task_names, True)
    json_dump(integrity, output_dir / "prediction_integrity.json")
    predictions.to_csv(prediction_path, index=False)
    wide = predictions.pivot(index="molecule_id", columns="CYP", values="y_pred").reset_index()
    wide.to_csv(output_dir / "test_predictions_wide.csv", index=False)
    json_dump(
        {
            "run_dir": str(run_dir), "rows": len(predictions),
            "checkpoint_hashes": _checkpoint_hashes(run_dir),
        },
        output_dir / "prediction_metadata.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
