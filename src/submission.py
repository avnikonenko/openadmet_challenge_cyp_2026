from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import CYPS, DATA_DIR


SUBMISSION_COLUMNS = [
    "SMILES", "Molecule_Name", *[f"{cyp}_pIC50_direct_inhibition" for cyp in CYPS]
]


def build_submission(predictions: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    required = {"molecule_id", "CYP", "y_pred"}
    if not required.issubset(predictions):
        raise ValueError(f"Prediction input lacks columns {sorted(required - set(predictions))}")
    if predictions.duplicated(["molecule_id", "CYP"]).any():
        raise ValueError("Prediction input contains duplicate molecule/CYP rows")
    expected_keys = {(name, cyp) for name in test["Molecule_Name"] for cyp in CYPS}
    actual_keys = set(
        map(tuple, predictions[["molecule_id", "CYP"]].itertuples(index=False, name=None))
    )
    if actual_keys != expected_keys:
        raise ValueError(
            f"Prediction coverage is incomplete: missing={len(expected_keys - actual_keys)}, "
            f"unexpected={len(actual_keys - expected_keys)}"
        )
    values = predictions.pivot(index="molecule_id", columns="CYP", values="y_pred")
    values = values.rename(columns={cyp: f"{cyp}_pIC50_direct_inhibition" for cyp in CYPS})
    submission = test[["SMILES", "Molecule_Name"]].merge(
        values, left_on="Molecule_Name", right_index=True, how="left", validate="one_to_one",
        sort=False,
    )
    submission = submission.loc[:, SUBMISSION_COLUMNS]
    validate_submission(submission, test)
    return submission


def validate_submission(submission: pd.DataFrame, test: pd.DataFrame) -> dict[str, object]:
    if list(submission.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"Submission columns/order must be exactly {SUBMISSION_COLUMNS}")
    if len(submission) != len(test):
        raise ValueError(f"Submission has {len(submission)} rows; expected {len(test)}")
    if submission["Molecule_Name"].duplicated().any():
        raise ValueError("Submission contains duplicate Molecule_Name values")
    if not submission["Molecule_Name"].equals(test["Molecule_Name"]):
        raise ValueError("Submission molecule identifiers/order differ from blinded test data")
    if not submission["SMILES"].equals(test["SMILES"]):
        raise ValueError("Submission SMILES/order differ from blinded test data")
    targets = SUBMISSION_COLUMNS[2:]
    numeric = submission[targets].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Submission predictions must all be finite numeric values")
    return {"rows": len(submission), "targets": len(targets), "finite_values": int(numeric.size)}


def official_test_frame() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "cyp-challenge-TEST-BLINDED.csv")

