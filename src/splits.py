from __future__ import annotations

import pandas as pd

from .constants import FOLD_COLUMNS


def fold_column(scheme: str) -> str:
    try:
        return FOLD_COLUMNS[scheme]
    except KeyError as exc:
        raise ValueError(f"Unknown split scheme {scheme!r}; choose {sorted(FOLD_COLUMNS)}") from exc


def split_masks(frame: pd.DataFrame, scheme: str, fold: int) -> tuple[pd.Series, pd.Series]:
    column = fold_column(scheme)
    if column not in frame:
        raise ValueError(f"Fold column {column!r} is absent; folds are never regenerated")
    values = frame[column]
    if values.isna().any() or not set(values.astype(int)).issubset(set(range(5))):
        raise ValueError(f"Invalid assignments in {column}")
    return values.ne(fold), values.eq(fold)


def auxiliary_inner_masks(
    frame: pd.DataFrame, scheme: str, outer_fold: int, inner_fold: int | None = None
) -> tuple[pd.Series, pd.Series, int]:
    column = fold_column(scheme)
    inner = (outer_fold + 1) % 5 if inner_fold is None else inner_fold
    if inner == outer_fold:
        raise ValueError("Auxiliary inner fold must differ from the outer fold")
    outer_training = frame[column].ne(outer_fold)
    return outer_training & frame[column].ne(inner), outer_training & frame[column].eq(inner), inner
