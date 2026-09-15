#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import _bootstrap  # noqa: F401
from models.lightgbm_model import build_lightgbm, require_lightgbm
from src.constants import FEATURE_SCHEMA_VERSION
from src.data import load_direct_data, save_split_ids, smoke_subset
from src.features import lightgbm_features
from src.metrics import regression_metrics
from src.resource_monitor import ResourceMonitor
from src.run_management import begin_run, finish_run
from src.utils import (
    GracefulStop, RuntimeGuard, add_common_args, json_dump, load_config,
    normalize_device, seed_everything,
)
from src.validation import audit_featurization, leakage_checks, validate_predictions


def main() -> int:
    parser = argparse.ArgumentParser(description="Train per-CYP Morgan-r3/RDKit LightGBM models")
    add_common_args(parser, "configs/lightgbm.yaml")
    parser.add_argument("--cyp", action="append", default=[])
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    if args.experiment:
        config["experiment"] = args.experiment
    if args.smoke_test:
        args.run_name = args.run_name or "smoke"
        config["lightgbm"]["n_estimators"] = min(
            20, int(config["lightgbm"].get("n_estimators", 20))
        )
    feature_config = config.get("features", {})
    if int(feature_config.get("morgan_radius", 3)) != 3 or int(
        feature_config.get("morgan_bits", 2048)
    ) != 2048:
        raise ValueError("This pipeline requires Morgan radius 3 with 2048 bits")
    lightgbm = require_lightgbm()
    seed_everything(args.seed)
    run, metadata, started = begin_run(args, config)
    if metadata.get("already_completed"):
        print(f"Run already completed: {run.run_dir}")
        return 0
    cyps = args.cyp or config.get("data", {}).get("cyps")
    bundle = load_direct_data(config.get("data", {}).get("split_scheme", "ecfp_cluster"), args.fold, cyps)
    if args.smoke_test:
        bundle = smoke_subset(bundle)
    leakage_checks(bundle, run.run_dir)
    featurization = audit_featurization(
        bundle, run.run_dir, config.get("featurization", {}).get("policy", "fail")
    )
    save_split_ids(bundle, run.run_dir)
    unused_indices = set(bundle.frame.index[~(bundle.train_mask | bundle.validation_mask)])
    skipped_indices = unused_indices | set(featurization["failed_row_indices"])
    features, feature_names = lightgbm_features(bundle.frame, skipped_indices)
    json_dump({"feature_count": len(feature_names), "features": feature_names}, run.run_dir / "feature_manifest.json")
    guard = RuntimeGuard(args.max_runtime_minutes, started=started)
    rows = []
    completed_tasks = []
    history_path = run.run_dir / "training_history.csv"
    target_history = (
        pd.read_csv(history_path).to_dict("records")
        if args.resume and history_path.exists() else []
    )
    estimated_target_seconds = (
        max(float(row["runtime_seconds"]) for row in target_history)
        if target_history else config.get("walltime", {}).get("initial_target_estimate_seconds")
    )
    runtime_margin = float(config.get("walltime", {}).get("safety_margin_seconds", 60))
    normalized_device = normalize_device(args.device)
    monitor = ResourceMonitor(normalized_device)
    monitor.start()
    graceful_stop = GracefulStop()
    graceful_stop.__enter__()
    for index, (cyp, target, lower, upper, uncertainty) in enumerate(
        zip(bundle.task_names, bundle.target_columns, bundle.lower_columns, bundle.upper_columns, bundle.uncertainty_columns)
    ):
        model_path = run.checkpoints / f"{cyp}.pkl"
        resuming_model = args.resume and model_path.exists()
        if not resuming_model and (
            graceful_stop.requested or not guard.can_start_epoch(
                estimated_target_seconds, runtime_margin
            )
        ):
            run.log(
                "training_stopped_before_target", CYP=cyp,
                reason=f"signal:{graceful_stop.signal_name}" if graceful_stop.signal_name else "insufficient_runtime",
            )
            break
        target_started = time.monotonic()
        train_mask = bundle.train_mask & bundle.frame[target].notna()
        validation_mask = bundle.validation_mask & bundle.frame[target].notna()
        if int(train_mask.sum()) < 2 or int(validation_mask.sum()) < 2:
            raise ValueError(
                f"Insufficient observed {cyp} labels: train={int(train_mask.sum())}, "
                f"validation={int(validation_mask.sum())}"
            )
        if resuming_model:
            with model_path.open("rb") as handle:
                model = pickle.load(handle)
        else:
            model = build_lightgbm(config.get("lightgbm", {}), args.seed, args.device)
            fit_kwargs = {}
            loss_mode = config.get("loss", {}).get("loss_mode", "standard")
            if loss_mode == "clipped_uncertainty_weighted":
                sd = bundle.frame.loc[train_mask, uncertainty].to_numpy(float)
                weights = np.ones(len(sd), dtype=float)
                valid = np.isfinite(sd) & (sd > 0)
                raw = 1 / np.square(sd[valid])
                if len(raw):
                    raw /= np.median(raw)
                    weights[valid] = np.clip(
                        raw, config["loss"].get("weight_clip_min", 0.2),
                        config["loss"].get("weight_clip_max", 5.0),
                    )
                fit_kwargs["sample_weight"] = weights
                json_dump(
                    {
                        "CYP": cyp, "observations": len(weights),
                        "weighted_observations": int(valid.sum()),
                        "min": float(weights.min()), "median": float(np.median(weights)),
                        "max": float(weights.max()),
                    },
                    run.run_dir / f"uncertainty_weight_summary_{cyp}.json",
                )
            elif loss_mode == "interval_aware":
                raise ValueError("interval_aware training is implemented for D-MPNN; use standard or clipped weighting for LightGBM")
            model.fit(features[train_mask], bundle.frame.loc[train_mask, target], **fit_kwargs)
            temporary = model_path.with_suffix(".pkl.tmp")
            with temporary.open("wb") as handle:
                pickle.dump(model, handle)
            temporary.replace(model_path)
        json_dump(
            {
                "architecture_type": "lightgbm",
                "model_config": model.get_params(deep=False),
                "requested_model_config": config.get("lightgbm", {}),
                "feature_config": config.get("features", {}),
                "task_names": [cyp],
                "target_normalization": {"mode": "none", "mean": [0.0], "std": [1.0]},
                "model_code_version": "openadmet-lightgbm-v1",
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "lightgbm_version": lightgbm.__version__,
                "pytorch_version": torch.__version__,
                "chemprop_version": "not-applicable",
            },
            run.checkpoints / f"{cyp}.meta.json",
        )
        prediction = model.predict(features[validation_mask])
        selected = bundle.frame.loc[validation_mask]
        for row, pred in zip(selected.itertuples(index=False), prediction):
            rows.append(
                {
                    "molecule_id": row.Molecule_Name, "canonical_smiles": row.canonical_smiles,
                    "fold": args.fold, "CYP": cyp, "y_true": getattr(row, target),
                    "y_pred": float(pred), "experimental_uncertainty": getattr(row, uncertainty),
                    "lower": getattr(row, lower), "upper": getattr(row, upper),
                    "model": config.get("experiment", "lightgbm"), "seed": args.seed,
                    "prediction_scale": "original",
                }
            )
        completed_tasks.append(cyp)
        target_seconds = time.monotonic() - target_started
        estimated_target_seconds = max(target_seconds, estimated_target_seconds or 0)
        if cyp not in {row["CYP"] for row in target_history}:
            target_history.append({"CYP": cyp, "runtime_seconds": target_seconds})
        pd.DataFrame(target_history).to_csv(run.run_dir / "training_history.csv", index=False)
        run.log("target_completed", CYP=cyp, runtime_seconds=target_seconds)
        if guard.should_stop() or graceful_stop.requested:
            break
    graceful_stop.__exit__(None, None, None)
    resource_summary = monitor.finish()
    prediction_columns = [
        "molecule_id", "canonical_smiles", "fold", "CYP", "y_true", "y_pred",
        "experimental_uncertainty", "lower", "upper", "model", "seed", "prediction_scale",
    ]
    predictions = pd.DataFrame(rows, columns=prediction_columns)
    expected_keys = {
        (row.Molecule_Name, cyp)
        for cyp, target in zip(bundle.task_names, bundle.target_columns)
        if cyp in completed_tasks
        for row in bundle.frame.loc[bundle.validation_mask & bundle.frame[target].notna()].itertuples()
    }
    integrity = (
        validate_predictions(predictions, expected_keys, completed_tasks, True)
        if completed_tasks else {
            "rows": 0, "unique_molecules": 0, "tasks": [], "finite": True,
            "inverse_transformed": True, "note": "no target started before runtime guard",
        }
    )
    json_dump(integrity, run.run_dir / "prediction_integrity.json")
    predictions.to_csv(run.run_dir / "predictions.csv", index=False)
    if completed_tasks:
        per_cyp, macro = regression_metrics(predictions)
    else:
        per_cyp, macro = pd.DataFrame(), {}
    per_cyp.to_csv(run.run_dir / "metrics_per_cyp.csv", index=False)
    json_dump(macro, run.run_dir / "metrics.json")
    finished = len(completed_tasks) == len(bundle.task_names)
    result = {
        "completed_CYPs": completed_tasks,
        "best_epoch": None,
        "selection_metric": "macro_ST_RAE",
        "best_validation_metric": macro.get("macro_ST_RAE"),
        "final_metrics": macro,
        "precision_mode": "fp32",
        "metric_direction": "minimize",
        "stop_signal": graceful_stop.signal_name,
        **resource_summary,
    }
    finish_run(run, metadata, started, result, completed=finished)
    print(f"Run {'completed' if finished else 'checkpointed'}: {run.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
