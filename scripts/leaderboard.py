#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an experiment leaderboard from run outputs")
    parser.add_argument("--input", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/experiment_leaderboard.csv"))
    args = parser.parse_args()
    rows = []
    for metadata_path in sorted(args.input.glob("**/metadata.json")):
        run_dir = metadata_path.parent
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metrics_path = run_dir / "metrics_per_cyp.csv"
        metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
        macro_st_rae = metadata.get("final_metrics", {}).get("macro_ST_RAE")
        pretraining = metadata.get("pretrained_encoder") or {}
        base = {
            "experiment": metadata.get("experiment"),
            "model": metadata.get("model_type"),
            "fold": metadata.get("fold"), "seed": metadata.get("seed"),
            "transfer_mode": metadata.get("transfer_mode", "none"),
            "pretraining_source": pretraining.get("checkpoint") if isinstance(pretraining, dict) else None,
            "macro_ST_RAE": macro_st_rae,
            "runtime_seconds": metadata.get(
                "cumulative_runtime_seconds", metadata.get("runtime_seconds")
            ),
            "peak_GPU_memory_MB": metadata.get("peak_gpu_memory_mb"),
            "best_epoch": metadata.get("best_epoch"),
            "completion_status": metadata.get("status", "unknown"),
            "run_dir": str(run_dir.resolve()),
        }
        if len(metrics):
            for metric in metrics.to_dict("records"):
                rows.append({**base, **metric})
        else:
            rows.append({**base, "CYP": None, "ST_RAE": None, "MAE": None, "RMSE": None, "Spearman": None})
    columns = [
        "experiment", "model", "fold", "seed", "transfer_mode", "pretraining_source",
        "CYP", "ST_RAE", "macro_ST_RAE", "MAE", "RMSE", "Spearman",
        "runtime_seconds", "peak_GPU_memory_MB", "best_epoch", "completion_status", "run_dir",
    ]
    leaderboard = pd.DataFrame(rows).reindex(columns=columns)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(args.output, index=False)
    completed = leaderboard.loc[leaderboard["completion_status"].eq("completed") & leaderboard["CYP"].notna()]
    group_columns = ["experiment", "model", "transfer_mode", "pretraining_source", "CYP"]
    summary = completed.groupby(group_columns, dropna=False).agg(
        runs=("ST_RAE", "size"), folds=("fold", "nunique"), seeds=("seed", "nunique"),
        ST_RAE_mean=("ST_RAE", "mean"), ST_RAE_std=("ST_RAE", "std"),
        macro_ST_RAE_mean=("macro_ST_RAE", "mean"), macro_ST_RAE_std=("macro_ST_RAE", "std"),
        MAE_mean=("MAE", "mean"), RMSE_mean=("RMSE", "mean"),
        Spearman_mean=("Spearman", "mean"), runtime_seconds_mean=("runtime_seconds", "mean"),
        peak_GPU_memory_MB_max=("peak_GPU_memory_MB", "max"),
    ).reset_index()
    summary_path = args.output.with_name(args.output.stem + "_aggregated.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Runs={leaderboard['run_dir'].nunique() if len(leaderboard) else 0}; rows={len(leaderboard)}; output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
