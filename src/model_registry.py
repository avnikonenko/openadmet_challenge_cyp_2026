from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import pandas as pd


MODEL_STATISTICS_COLUMNS = [
    "run_id",
    "experiment",
    "model",
    "fold",
    "seed",
    "CYP",
    "N",
    "ST_RAE",
    "MAE",
    "RMSE",
    "Spearman",
    "R2",
    "macro_ST_RAE",
    "macro_MAE",
    "macro_RMSE",
    "macro_Spearman",
    "macro_R2",
    "transfer_mode",
    "pretraining_source",
    "loss_mode",
    "split_scheme",
    "model_hyperparameters",
    "training_hyperparameters",
    "best_epoch",
    "best_validation_metric",
    "selection_metric",
    "runtime_seconds",
    "peak_GPU_memory_MB",
    "average_GPU_utilization_percent",
    "maximum_GPU_utilization_percent",
    "completion_status",
    "run_started_at_utc",
    "run_completed_at_utc",
    "registered_at_utc",
    "registry_updated_at_utc",
    "CV_runs",
    "CV_folds",
    "CV_seeds",
    "CV_ST_RAE_mean",
    "CV_ST_RAE_std",
    "CV_MAE_mean",
    "CV_MAE_std",
    "CV_RMSE_mean",
    "CV_RMSE_std",
    "CV_Spearman_mean",
    "CV_Spearman_std",
    "CV_R2_mean",
    "CV_R2_std",
    "CV_macro_ST_RAE_mean",
    "CV_macro_ST_RAE_std",
    "run_dir",
]

REGISTRY_KEYS = ["run_id", "CYP"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_cell(value: object) -> str:
    return json.dumps(value or {}, sort_keys=True, separators=(",", ":"))


def _run_id(run_dir: Path, input_root: Path) -> str:
    try:
        return run_dir.resolve().relative_to(input_root.resolve()).as_posix()
    except ValueError:
        return str(run_dir.resolve())


def _registry_key(run_id: object, cyp: object) -> tuple[str, str]:
    return str(run_id), "" if pd.isna(cyp) else str(cyp)


def _metric_rows(run_dir: Path) -> list[dict[str, object]]:
    metrics_path = run_dir / "metrics_per_cyp.csv"
    if not metrics_path.exists():
        return [{}]
    metrics = pd.read_csv(metrics_path)
    if metrics.empty:
        return [{}]
    if "CYP" not in metrics or metrics["CYP"].isna().any():
        raise ValueError(f"Metrics file has missing CYP identities: {metrics_path}")
    if metrics["CYP"].duplicated().any():
        raise ValueError(f"Metrics file has duplicate CYP rows: {metrics_path}")
    return metrics.to_dict("records")


def collect_model_statistics(input_root: Path, timestamp: str) -> pd.DataFrame:
    input_root = input_root.resolve()
    rows: list[dict[str, object]] = []
    for metadata_path in sorted(input_root.glob("**/metadata.json")):
        run_dir = metadata_path.parent
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        final_metrics = metadata.get("final_metrics") or {}
        pretraining = metadata.get("pretrained_encoder") or {}
        model_config = metadata.get("model_hyperparameters") or {}
        training_config = metadata.get("training_hyperparameters") or {}
        config: dict[str, object] = {}
        config_path = run_dir / "config.yaml"
        if config_path.exists():
            import yaml

            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        loss_config = metadata.get("loss") or config.get("loss") or {}
        split_scheme = metadata.get("split_scheme") or (config.get("data") or {}).get(
            "split_scheme"
        )
        base = {
            "run_id": _run_id(run_dir, input_root),
            "experiment": metadata.get("experiment"),
            "model": metadata.get("model_type"),
            "fold": metadata.get("fold"),
            "seed": metadata.get("seed"),
            "macro_ST_RAE": final_metrics.get("macro_ST_RAE"),
            "macro_MAE": final_metrics.get("macro_MAE"),
            "macro_RMSE": final_metrics.get("macro_RMSE"),
            "macro_Spearman": final_metrics.get("macro_Spearman"),
            "macro_R2": final_metrics.get("macro_R2"),
            "transfer_mode": metadata.get("transfer_mode", "none"),
            "pretraining_source": (
                pretraining.get("checkpoint") if isinstance(pretraining, dict) else None
            ),
            "loss_mode": loss_config.get("loss_mode"),
            "split_scheme": split_scheme,
            "model_hyperparameters": _json_cell(model_config),
            "training_hyperparameters": _json_cell(training_config),
            "best_epoch": metadata.get("best_epoch"),
            "best_validation_metric": metadata.get("best_validation_metric"),
            "selection_metric": metadata.get("selection_metric"),
            "runtime_seconds": metadata.get(
                "cumulative_runtime_seconds", metadata.get("runtime_seconds")
            ),
            "peak_GPU_memory_MB": metadata.get("peak_gpu_memory_mb"),
            "average_GPU_utilization_percent": metadata.get("average_gpu_utilization_percent"),
            "maximum_GPU_utilization_percent": metadata.get("max_gpu_utilization_percent"),
            "completion_status": metadata.get("status", "unknown"),
            "run_started_at_utc": metadata.get("start_time"),
            "run_completed_at_utc": metadata.get("end_time"),
            "registry_updated_at_utc": timestamp,
            "run_dir": str(run_dir.resolve()),
        }
        for metric in _metric_rows(run_dir):
            rows.append({**base, **metric})
    frame = pd.DataFrame(rows).reindex(columns=MODEL_STATISTICS_COLUMNS)
    if not frame.empty and frame.duplicated(REGISTRY_KEYS).any():
        duplicates = frame.loc[frame.duplicated(REGISTRY_KEYS, keep=False), REGISTRY_KEYS]
        raise ValueError(f"Duplicate model-statistics keys detected: {duplicates.head().to_dict('records')}")
    return frame


def _add_cv_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.reindex(columns=MODEL_STATISTICS_COLUMNS)
    result = frame.drop(
        columns=[column for column in MODEL_STATISTICS_COLUMNS if column.startswith("CV_")],
        errors="ignore",
    )
    completed = result.loc[
        result["completion_status"].eq("completed") & result["CYP"].notna()
    ].copy()
    if completed.empty:
        return result.reindex(columns=MODEL_STATISTICS_COLUMNS)
    signature_columns = [
        "experiment", "model", "transfer_mode", "CYP", "loss_mode", "split_scheme",
        "model_hyperparameters", "training_hyperparameters",
    ]
    result["_cv_group"] = result[signature_columns].fillna("").astype(str).agg("\x1f".join, axis=1)
    completed = result.loc[
        result["completion_status"].eq("completed") & result["CYP"].notna()
    ].copy()
    aggregate = completed.groupby("_cv_group", dropna=False).agg(
        CV_runs=("ST_RAE", "size"),
        CV_folds=("fold", "nunique"),
        CV_seeds=("seed", "nunique"),
        **{
            f"CV_{metric}_{stat}": (metric, stat)
            for metric in ("ST_RAE", "MAE", "RMSE", "Spearman", "R2", "macro_ST_RAE")
            for stat in ("mean", "std")
        },
    ).reset_index()
    result = result.merge(aggregate, on="_cv_group", how="left", validate="many_to_one")
    result = result.drop(columns="_cv_group")
    return result.reindex(columns=MODEL_STATISTICS_COLUMNS)


@contextmanager
def _registry_lock(output_path: Path) -> Iterator[None]:
    import fcntl

    lock_path = output_path.with_suffix(output_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def update_model_statistics(input_root: Path, output_path: Path) -> pd.DataFrame:
    input_root = input_root.resolve()
    output_path = output_path.resolve()
    timestamp = utc_now()
    with _registry_lock(output_path):
        current = collect_model_statistics(input_root, timestamp)
        existing = pd.DataFrame()
        first_seen: dict[tuple[str, str], object] = {}
        if output_path.exists():
            existing = pd.read_csv(output_path)
            required = set(MODEL_STATISTICS_COLUMNS)
            if required.issubset(existing):
                first_seen = {
                    _registry_key(row.run_id, row.CYP): row.registered_at_utc
                    for row in existing.itertuples(index=False)
                }
        if not current.empty:
            current["registered_at_utc"] = [
                first_seen.get(_registry_key(row.run_id, row.CYP), timestamp)
                for row in current.itertuples(index=False)
            ]
        if not existing.empty and set(MODEL_STATISTICS_COLUMNS).issubset(existing):
            current_run_ids = set(current["run_id"].astype(str))
            retained = existing.loc[~existing["run_id"].astype(str).isin(current_run_ids)]
            current = pd.concat([retained, current], ignore_index=True)
        current = _add_cv_statistics(current)
        current = current.reindex(columns=MODEL_STATISTICS_COLUMNS).sort_values(
            ["experiment", "fold", "seed", "CYP", "run_id"],
            kind="mergesort",
            na_position="last",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", prefix=f".{output_path.stem}_", dir=output_path.parent,
            delete=False, encoding="utf-8", newline="",
        ) as handle:
            temporary_path = Path(handle.name)
            current.to_csv(handle, index=False)
        try:
            os.replace(temporary_path, output_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
    return current
