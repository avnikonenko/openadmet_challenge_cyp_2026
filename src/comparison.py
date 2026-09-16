"""Model comparison and out-of-fold error-complementarity diagnostics.

Two questions this answers, both of which a single macro number cannot:

1. Is a candidate model actually better than the reference, or is the difference
   inside fold/seed variation? Every comparison is paired on the same (fold, seed)
   units and reported with a bootstrap confidence interval on the paired difference.
2. Do two models make *different* errors? Ensembling near-identical models adds
   overfitting risk without adding information, so residual correlations are computed
   before ensemble members are chosen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


PRIMARY_METRIC = "macro_ST_RAE"
RESIDUAL_KEYS = ["molecule_id", "CYP"]


def bootstrap_ci(
    values: np.ndarray, samples: int = 1000, seed: int = 20260916, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of paired differences."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 2:
        return (float("nan"), float("nan"))
    generator = np.random.default_rng(seed)
    draws = generator.choice(finite, size=(samples, len(finite)), replace=True).mean(axis=1)
    return (
        float(np.quantile(draws, alpha / 2)),
        float(np.quantile(draws, 1 - alpha / 2)),
    )


def dispersion_summary(macro_metrics: pd.DataFrame, metric: str = PRIMARY_METRIC) -> pd.DataFrame:
    """Per-model mean and SD overall, across folds, and across seeds."""
    required = {"model", "fold", "seed", metric}
    missing = required - set(macro_metrics.columns)
    if missing:
        raise ValueError(f"Macro metrics are missing columns: {sorted(missing)}")
    rows = []
    for model, group in macro_metrics.groupby("model", sort=True):
        per_fold = group.groupby("fold")[metric].mean()
        per_seed = group.groupby("seed")[metric].mean()
        rows.append(
            {
                "model": model,
                "runs": len(group),
                "folds": group["fold"].nunique(),
                "seeds": group["seed"].nunique(),
                f"{metric}_mean": float(group[metric].mean()),
                f"{metric}_sd": float(group[metric].std(ddof=1)) if len(group) > 1 else np.nan,
                f"{metric}_sd_across_folds": (
                    float(per_fold.std(ddof=1)) if len(per_fold) > 1 else np.nan
                ),
                f"{metric}_sd_across_seeds": (
                    float(per_seed.std(ddof=1)) if len(per_seed) > 1 else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def paired_comparison(
    macro_metrics: pd.DataFrame, reference: str, metric: str = PRIMARY_METRIC,
    samples: int = 1000, seed: int = 20260916,
) -> pd.DataFrame:
    """Fold/seed-paired differences against a reference model.

    A negative delta means the candidate improves the primary metric, which is
    minimized. Only (fold, seed) units present for both models are compared, and the
    number of improving folds is reported so the experiment design's "at least three
    of five outer folds" rule can be checked directly.
    """
    if reference not in set(macro_metrics["model"]):
        raise ValueError(f"Reference model {reference!r} has no runs to compare against")
    reference_runs = (
        macro_metrics.loc[macro_metrics["model"].eq(reference)]
        .groupby(["fold", "seed"])[metric].mean()
    )
    rows = []
    for model, group in macro_metrics.groupby("model", sort=True):
        candidate = group.groupby(["fold", "seed"])[metric].mean()
        shared = candidate.index.intersection(reference_runs.index)
        deltas = (candidate.loc[shared] - reference_runs.loc[shared]).to_numpy(dtype=float)
        low, high = bootstrap_ci(deltas, samples=samples, seed=seed)
        fold_means = (
            pd.Series(deltas, index=shared).groupby(level="fold").mean()
            if len(shared) else pd.Series(dtype=float)
        )
        rows.append(
            {
                "model": model,
                "reference": reference,
                "paired_runs": int(len(shared)),
                "paired_delta_mean": float(np.mean(deltas)) if len(deltas) else np.nan,
                "paired_delta_sd": (
                    float(np.std(deltas, ddof=1)) if len(deltas) > 1 else np.nan
                ),
                "paired_delta_ci_low": low,
                "paired_delta_ci_high": high,
                "folds_compared": int(fold_means.size),
                "folds_improved": int((fold_means < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def model_comparison_table(
    macro_metrics: pd.DataFrame, reference: str | None = None,
    metric: str = PRIMARY_METRIC, samples: int = 1000, seed: int = 20260916,
) -> tuple[pd.DataFrame, str]:
    """Comparison table combining dispersion, raw delta, and paired delta with CI.

    The reference defaults to the model with the best mean primary metric, which is
    the current best transfer model once its runs are present.
    """
    working = macro_metrics.copy()
    if "CYP_set" not in working:
        working["CYP_set"] = "unspecified"
    coverage_counts = working.groupby("model")["CYP_set"].nunique()
    inconsistent = coverage_counts.loc[coverage_counts.ne(1)]
    if len(inconsistent):
        raise ValueError(
            "A model identity spans different endpoint sets; use distinct model/run names: "
            f"{inconsistent.index.tolist()}"
        )
    coverage = working.groupby("model", sort=True).agg(
        CYP_set=("CYP_set", "first"),
        scored_CYPs=("scored_CYPs", "first") if "scored_CYPs" in working else ("CYP_set", "size"),
    ).reset_index()
    if "scored_CYPs" not in working:
        coverage["scored_CYPs"] = coverage["CYP_set"].map(
            lambda value: 0 if value == "" else len(str(value).split("|"))
        )
    summary = dispersion_summary(working, metric)
    if summary.empty:
        return summary, ""
    summary = summary.merge(coverage, on="model", how="left", validate="one_to_one")
    if reference:
        if reference not in set(summary["model"]):
            raise ValueError(f"Reference model {reference!r} has no runs to compare against")
        resolved = reference
    else:
        maximum_coverage = int(summary["scored_CYPs"].max())
        eligible = summary.loc[summary["scored_CYPs"].eq(maximum_coverage)]
        resolved = str(
            eligible.sort_values(f"{metric}_mean", kind="mergesort").iloc[0]["model"]
        )
    reference_set = str(summary.loc[summary["model"].eq(resolved), "CYP_set"].iloc[0])
    comparable_models = set(summary.loc[summary["CYP_set"].eq(reference_set), "model"])
    comparable = working.loc[working["model"].isin(comparable_models)]
    paired = paired_comparison(comparable, resolved, metric, samples, seed)
    aggregations = {
        output: (column, "mean")
        for output, column in (("MAE", "macro_MAE"), ("Spearman", "macro_Spearman"))
        if column in macro_metrics.columns
    }
    table = summary.merge(paired, on="model", how="left")
    if aggregations:
        secondary = working.groupby("model", sort=True).agg(**aggregations).reset_index()
        table = table.merge(secondary, on="model", how="left")
    reference_mean = float(
        summary.loc[summary["model"].eq(resolved), f"{metric}_mean"].iloc[0]
    )
    table["comparison_status"] = np.where(
        table["CYP_set"].eq(reference_set), "comparable", "incompatible_endpoint_set"
    )
    table["delta_vs_reference"] = np.where(
        table["comparison_status"].eq("comparable"),
        table[f"{metric}_mean"] - reference_mean,
        np.nan,
    )
    ordered = [
        "model", "CYP_set", "scored_CYPs", "comparison_status",
        f"{metric}_mean", f"{metric}_sd", f"{metric}_sd_across_folds",
        f"{metric}_sd_across_seeds", "delta_vs_reference", "paired_delta_mean",
        "paired_delta_ci_low", "paired_delta_ci_high", "MAE", "Spearman",
        "runs", "folds", "seeds", "paired_runs", "folds_compared", "folds_improved",
        "reference",
    ]
    table = table.reindex(columns=ordered).sort_values(
        f"{metric}_mean", kind="mergesort"
    )
    return table, resolved


def _residual_frame(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    required = set(RESIDUAL_KEYS + ["y_true", "y_pred"])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"OOF predictions for {model!r} lack columns: {sorted(missing)}")
    residuals = frame.loc[:, RESIDUAL_KEYS + ["y_true", "y_pred"]].copy()
    if residuals.duplicated(RESIDUAL_KEYS).any():
        raise ValueError(f"OOF predictions for {model!r} contain duplicate molecule/CYP rows")
    residuals["residual"] = residuals["y_true"] - residuals["y_pred"]
    return residuals


def _correlations(left: pd.Series, right: pd.Series) -> tuple[float, float]:
    if len(left) < 3 or left.nunique() < 2 or right.nunique() < 2:
        return (float("nan"), float("nan"))
    return (
        float(stats.pearsonr(left, right).statistic),
        float(stats.spearmanr(left, right).statistic),
    )


def residual_complementarity(
    predictions_by_model: dict[str, pd.DataFrame], per_cyp: bool = True
) -> pd.DataFrame:
    """Pairwise out-of-fold residual agreement between candidate ensemble members.

    High residual correlation with small disagreement means the two models fail on the
    same molecules, so averaging them mostly duplicates one model.
    """
    residuals = {
        model: _residual_frame(frame, model)
        for model, frame in predictions_by_model.items()
    }
    names = sorted(residuals)
    rows = []
    for position, left_name in enumerate(names):
        for right_name in names[position + 1 :]:
            merged = residuals[left_name].merge(
                residuals[right_name], on=RESIDUAL_KEYS, how="inner",
                suffixes=("_left", "_right"), validate="one_to_one",
            )
            if merged.empty:
                continue
            if not np.allclose(
                merged["y_true_left"].to_numpy(dtype=float),
                merged["y_true_right"].to_numpy(dtype=float),
                equal_nan=True,
            ):
                raise ValueError(
                    f"OOF truth values disagree between {left_name!r} and {right_name!r} "
                    "for shared molecule/CYP rows"
                )
            groups = [("all", merged)]
            if per_cyp:
                groups.extend(
                    (str(cyp), group) for cyp, group in merged.groupby("CYP", sort=True)
                )
            for scope, group in groups:
                pearson, spearman = _correlations(
                    group["residual_left"], group["residual_right"]
                )
                disagreement = (group["y_pred_left"] - group["y_pred_right"]).abs()
                rows.append(
                    {
                        "model_a": left_name,
                        "model_b": right_name,
                        "CYP": scope,
                        "N": int(len(group)),
                        "residual_pearson": pearson,
                        "residual_spearman": spearman,
                        "mean_absolute_disagreement": float(disagreement.mean()),
                        "max_absolute_disagreement": float(disagreement.max()),
                        "mean_absolute_residual_a": float(group["residual_left"].abs().mean()),
                        "mean_absolute_residual_b": float(group["residual_right"].abs().mean()),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "model_a", "model_b", "CYP", "N", "residual_pearson", "residual_spearman",
            "mean_absolute_disagreement", "max_absolute_disagreement",
            "mean_absolute_residual_a", "mean_absolute_residual_b",
        ],
    )
