from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def official_st_rae(y_true, y_pred, lower, upper) -> float:
    y = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    low = np.asarray(lower, dtype=float)
    high = np.asarray(upper, dtype=float)
    valid = np.isfinite(y) & np.isfinite(prediction) & np.isfinite(low) & np.isfinite(high)
    y, prediction, low, high = y[valid], prediction[valid], low[valid], high[valid]
    if not len(y):
        return np.nan
    numerator = np.clip(prediction - high, 0, None) + np.clip(low - prediction, 0, None)
    baseline = y.mean()
    denominator = np.clip(baseline - high, 0, None) + np.clip(low - baseline, 0, None)
    return float(numerator.sum() / denominator.sum()) if denominator.sum() > 0 else np.nan


def regression_metrics(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    required = {"CYP", "y_true", "y_pred", "lower", "upper"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
    rows = []
    for cyp, group in frame.groupby("CYP", sort=True):
        valid = group[["y_true", "y_pred"]].dropna()
        paired = group[["y_true", "y_pred", "lower", "upper"]].dropna()
        if len(valid) < 2:
            continue
        spearman = (
            stats.spearmanr(valid["y_true"], valid["y_pred"]).statistic
            if valid["y_true"].nunique() > 1 and valid["y_pred"].nunique() > 1
            else np.nan
        )
        rows.append(
            {
                "CYP": cyp,
                "N": len(valid),
                "ST_RAE": official_st_rae(
                    paired["y_true"], paired["y_pred"], paired["lower"], paired["upper"]
                ),
                "MAE": mean_absolute_error(valid["y_true"], valid["y_pred"]),
                "RMSE": mean_squared_error(valid["y_true"], valid["y_pred"]) ** 0.5,
                "Spearman": float(spearman),
                "R2": r2_score(valid["y_true"], valid["y_pred"]),
            }
        )
    metric_names = ("ST_RAE", "MAE", "RMSE", "Spearman", "R2")
    per_cyp = pd.DataFrame(rows, columns=["CYP", "N", *metric_names])
    macro = {
        f"macro_{metric}": float(per_cyp[metric].mean())
        for metric in metric_names
    }
    return per_cyp, macro


def objective_value(metrics: dict[str, float], name: str = "macro_ST_RAE") -> float:
    value = metrics.get(name)
    return float(value) if value is not None and np.isfinite(value) else float("inf")
