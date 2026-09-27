from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.ensemble_validation.scripts.validate_ensemble import bootstrap_differences, metric_tables


def _synthetic_oof() -> pd.DataFrame:
    rows = []
    for molecule in range(40):
        for cyp in ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"):
            y = 3 + molecule / 10
            rows.append({
                "scheme": "ecfp_cluster", "molecule_id": f"M{molecule}", "fold": molecule % 5,
                "CYP": cyp, "y_true": y, "lower": y - 0.05, "upper": y + 0.05,
                "transfer": y + 0.8, "joint": y + 0.1, "lightgbm": y + 0.3,
                "ensemble": y + 0.4,
            })
    return pd.DataFrame(rows)


def test_ensemble_metric_is_better_than_transfer_on_synthetic_oof() -> None:
    metrics, folds = metric_tables(_synthetic_oof())
    macro = metrics.loc[metrics.CYP.eq("macro")].set_index("model")
    assert macro.loc["ensemble", "ST_RAE"] < macro.loc["transfer", "ST_RAE"]
    assert len(folds.loc[folds.CYP.eq("macro") & folds.model.eq("ensemble")]) == 5


def test_paired_molecule_bootstrap_is_reproducible() -> None:
    frame = _synthetic_oof()
    first = bootstrap_differences(frame, repeats=100, seed=19)
    second = bootstrap_differences(frame, repeats=100, seed=19)
    pd.testing.assert_frame_equal(first, second)
    assert np.isfinite(first[["delta_mean", "ci_low", "ci_high"]].to_numpy()).all()
    assert first.ci_high.lt(0).all()
