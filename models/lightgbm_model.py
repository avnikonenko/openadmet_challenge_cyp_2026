from __future__ import annotations

import os
from typing import Any


def require_lightgbm():
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError(
            "LightGBM is required. Create the modelling environment from "
            "environment-modeling.yml."
        ) from exc
    return lgb


def build_lightgbm(params: dict[str, Any], seed: int, device: str = "cpu"):
    lgb = require_lightgbm()
    scheduler_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    default_jobs = int(scheduler_cpus) if scheduler_cpus else min(4, os.cpu_count() or 1)
    resolved = {
        "objective": "regression_l1",
        "n_estimators": 1200,
        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": max(1, default_jobs),
        "deterministic": True,
        "verbosity": -1,
        **params,
    }
    if int(resolved["n_jobs"]) < 1:
        raise ValueError("lightgbm.n_jobs must be a positive integer")
    if str(device).lower().startswith(("cuda", "gpu")):
        resolved["device_type"] = "gpu"
        normalized = str(device).lower()
        if ":" in normalized:
            resolved["gpu_device_id"] = int(normalized.split(":", 1)[1])
    return lgb.LGBMRegressor(**resolved)
