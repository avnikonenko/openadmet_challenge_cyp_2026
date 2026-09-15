#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.metrics import regression_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate and evaluate completed fold/seed runs")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--aggregate-oof", action="store_true")
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.input.resolve()
    output = (args.output_dir or root / "evaluation").resolve()
    output.mkdir(parents=True, exist_ok=True)
    prediction_paths = sorted(root.glob("**/predictions.csv"))
    prediction_by_run = {path.parent: path for path in prediction_paths}
    metadata_by_run = {path.parent: path for path in root.glob("**/metadata.json")}
    records, status_rows, macro_rows, completed_prediction_paths = [], [], [], []
    for run_dir in sorted(set(prediction_by_run) | set(metadata_by_run)):
        path = prediction_by_run.get(run_dir)
        metadata = {}
        try:
            metadata_path = metadata_by_run.get(run_dir)
            metadata = json.loads(metadata_path.read_text()) if metadata_path else {}
            run_status = metadata.get("status", "unknown")
            if run_status != "completed":
                status_rows.append({
                    "run_dir": str(run_dir), "model": metadata.get("experiment", "unknown"),
                    "fold": metadata.get("fold"), "seed": metadata.get("seed"),
                    "status": run_status,
                    "error": "Excluded from metrics and OOF because run is not completed",
                })
                continue
            if path is None:
                raise FileNotFoundError("predictions.csv is missing")
            frame = pd.read_csv(path)
            required = {
                "molecule_id", "fold", "CYP", "y_true", "y_pred", "lower", "upper",
                "seed", "model", "prediction_scale",
            }
            if not required.issubset(frame.columns):
                raise ValueError(f"missing columns {sorted(required - set(frame.columns))}")
            if frame.duplicated(["molecule_id", "CYP"]).any():
                raise ValueError("duplicate molecule/CYP predictions")
            if frame["fold"].nunique() != 1 or frame["seed"].nunique() != 1 or frame["model"].nunique() != 1:
                raise ValueError("predictions.csv mixes multiple folds, seeds, or models")
            if not np.isfinite(frame[["y_true", "y_pred"]].to_numpy(dtype=float)).all():
                raise ValueError("non-finite truth or prediction values")
            if "prediction_scale" not in frame or set(frame["prediction_scale"].dropna()) != {"original"}:
                raise ValueError("predictions are not marked as original target scale")
            per_cyp, macro = regression_metrics(frame)
            model = frame["model"].iloc[0]
            fold = int(frame["fold"].iloc[0])
            seed = int(frame["seed"].iloc[0])
            if int(metadata.get("fold", -1)) != fold or int(metadata.get("seed", -1)) != seed:
                raise ValueError("prediction fold/seed provenance disagrees with metadata.json")
            if metadata.get("experiment") != model:
                raise ValueError("prediction model name disagrees with metadata.json experiment")
            pretraining = bool(metadata.get("pretrained_encoder"))
            finetune_mode = metadata.get("transfer_mode", "none")
            for row in per_cyp.to_dict("records"):
                records.append(
                    {
                        "model": model,
                        "pretraining": pretraining,
                        "finetune_mode": finetune_mode,
                        "fold": fold, "seed": seed, **row,
                    }
                )
            macro_rows.append(
                {
                    "model": model, "pretraining": pretraining,
                    "finetune_mode": finetune_mode, "fold": fold, "seed": seed,
                    **macro,
                }
            )
            status_rows.append({"run_dir": str(run_dir), "model": model, "fold": fold, "seed": seed, "status": metadata.get("status", "unknown"), "error": "", **macro})
            completed_prediction_paths.append(path)
        except Exception as exc:
            status_rows.append({"run_dir": str(run_dir), "model": metadata.get("experiment", "unknown"), "fold": metadata.get("fold"), "seed": metadata.get("seed"), "status": "failed_or_incomplete", "error": f"{type(exc).__name__}: {exc}"})
    metrics = pd.DataFrame(records)
    macro_metrics = pd.DataFrame(macro_rows)
    status = pd.DataFrame(
        status_rows,
        columns=None if status_rows else ["run_dir", "model", "fold", "seed", "status", "error"],
    )
    status.to_csv(output / "run_status.csv", index=False)
    metrics.to_csv(output / "metrics_by_fold_seed_cyp.csv", index=False)
    macro_metrics.to_csv(output / "macro_metrics_by_fold_seed.csv", index=False)
    if len(metrics):
        summary = metrics.groupby(["model", "pretraining", "finetune_mode", "CYP"], dropna=False).agg(
            runs=("ST_RAE", "size"), ST_RAE_mean=("ST_RAE", "mean"), ST_RAE_std=("ST_RAE", "std"),
            MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"), RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"), Spearman_mean=("Spearman", "mean"),
            Spearman_std=("Spearman", "std"), R2_mean=("R2", "mean"), R2_std=("R2", "std"),
        ).reset_index()
        summary.to_csv(output / "metrics_summary.csv", index=False)
        metrics[["model", "pretraining", "finetune_mode", "fold", "seed", "CYP", "ST_RAE", "MAE", "RMSE", "Spearman"]].to_csv(output / "transfer_comparison.csv", index=False)
    if len(macro_metrics):
        macro_columns = [column for column in macro_metrics if column.startswith("macro_")]
        aggregations = {
            f"{column}_{stat}": (column, stat)
            for column in macro_columns
            for stat in ("mean", "std")
        }
        macro_summary = macro_metrics.groupby(
            ["model", "pretraining", "finetune_mode"], dropna=False
        ).agg(runs=("fold", "size"), **aggregations).reset_index()
        macro_summary.to_csv(output / "macro_metrics_summary.csv", index=False)
    if args.aggregate_oof and completed_prediction_paths:
        oof = pd.concat([pd.read_csv(path).assign(run_dir=str(path.parent)) for path in completed_prediction_paths], ignore_index=True)
        if oof.duplicated(["model", "seed", "molecule_id", "CYP"]).any():
            raise ValueError("OOF aggregation found duplicate model/seed/molecule/CYP predictions")
        oof.to_csv(output / "oof_predictions.csv", index=False)
        oof_cyp_rows, oof_macro_rows = [], []
        transfer_lookup = (
            metrics[["model", "seed", "pretraining", "finetune_mode"]]
            .drop_duplicates(["model", "seed"])
            .set_index(["model", "seed"])
            if len(metrics) else pd.DataFrame()
        )
        for (model, seed), group in oof.groupby(["model", "seed"], sort=True):
            per_cyp, macro = regression_metrics(group)
            if len(transfer_lookup) and (model, seed) in transfer_lookup.index:
                context = transfer_lookup.loc[(model, seed)].to_dict()
            else:
                context = {"pretraining": False, "finetune_mode": "none"}
            for row in per_cyp.to_dict("records"):
                oof_cyp_rows.append({"model": model, "seed": seed, **context, **row})
            oof_macro_rows.append({"model": model, "seed": seed, **context, **macro})
        oof_cyp_metrics = pd.DataFrame(oof_cyp_rows)
        oof_macro_metrics = pd.DataFrame(oof_macro_rows)
        oof_cyp_metrics.to_csv(output / "oof_metrics_by_seed_cyp.csv", index=False)
        oof_macro_metrics.to_csv(output / "oof_macro_metrics_by_seed.csv", index=False)
        if len(oof_cyp_metrics):
            oof_cyp_summary = oof_cyp_metrics.groupby(
                ["model", "pretraining", "finetune_mode", "CYP"], dropna=False
            ).agg(
                seeds=("seed", "nunique"),
                **{
                    f"{metric}_{stat}": (metric, stat)
                    for metric in ("ST_RAE", "MAE", "RMSE", "Spearman", "R2")
                    for stat in ("mean", "std")
                },
            ).reset_index()
            oof_cyp_summary.to_csv(output / "oof_metrics_summary_across_seeds.csv", index=False)
        if len(oof_macro_metrics):
            macro_columns = [column for column in oof_macro_metrics if column.startswith("macro_")]
            oof_macro_summary = oof_macro_metrics.groupby(
                ["model", "pretraining", "finetune_mode"], dropna=False
            ).agg(
                seeds=("seed", "nunique"),
                **{
                    f"{column}_{stat}": (column, stat)
                    for column in macro_columns
                    for stat in ("mean", "std")
                },
            ).reset_index()
            oof_macro_summary.to_csv(output / "oof_macro_summary_across_seeds.csv", index=False)
    missing_rows = []
    valid_status = status.loc[status["status"].eq("completed")]
    for (model, seed), group in valid_status.groupby(["model", "seed"], dropna=False):
        observed = set(group["fold"].dropna().astype(int))
        for fold in sorted(set(args.folds) - observed):
            missing_rows.append({"model": model, "seed": seed, "missing_fold": fold})
    pd.DataFrame(missing_rows, columns=["model", "seed", "missing_fold"]).to_csv(output / "missing_folds.csv", index=False)
    incomplete = int(status["status"].ne("completed").sum()) if len(status) else 0
    print(
        f"Evaluated {len(completed_prediction_paths)} completed prediction files "
        f"from {len(prediction_paths)} discovered files; "
        f"missing model/seed/fold combinations: {len(missing_rows)}; incomplete/failed runs: {incomplete}"
    )
    if not completed_prediction_paths:
        return 1
    return 2 if missing_rows or incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
