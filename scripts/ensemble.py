#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.comparison import residual_complementarity
from src.metrics import regression_metrics
from src.utils import json_dump


KEYS = ["molecule_id", "CYP"]


def cv_score(path: Path) -> float:
    _, macro = regression_metrics(pd.read_csv(path))
    return float(macro["macro_ST_RAE"])


def _run_identity(frame: pd.DataFrame, path: Path) -> tuple[str, int, int]:
    required = {"model", "fold", "seed", "prediction_scale"}
    if not required.issubset(frame):
        raise ValueError(f"{path} lacks run/provenance columns {sorted(required - set(frame))}")
    if set(frame["prediction_scale"].dropna()) != {"original"} or frame["prediction_scale"].isna().any():
        raise ValueError(f"{path} is not uniformly on the original target scale")
    identities = frame[["model", "fold", "seed"]].drop_duplicates()
    if len(identities) != 1:
        raise ValueError(f"{path} mixes multiple model/fold/seed identities")
    row = identities.iloc[0]
    return str(row["model"]), int(row["fold"]), int(row["seed"])


def _consistent_numeric_groups(frame: pd.DataFrame, columns: list[str]) -> bool:
    for _key, group in frame.groupby(KEYS, sort=False):
        for column in columns:
            values = group[column].to_numpy(dtype=float)
            if not np.allclose(values, values[0], rtol=1e-6, atol=1e-6, equal_nan=True):
                return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="OOF-derived arithmetic or weighted prediction ensemble")
    parser.add_argument("--prediction-inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--oof-inputs", nargs="+", type=Path, default=[])
    parser.add_argument("--method", choices=("mean", "weighted"), default="mean")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    existing_outputs = [
        args.output_dir / name
        for name in (
            "ensemble_predictions.csv", "test_ensemble_predictions.csv",
            "oof_ensemble_predictions.csv", "ensemble_metadata.json",
            "oof_residual_complementarity.csv",
        )
        if (args.output_dir / name).exists()
    ]
    if existing_outputs and not args.force:
        raise FileExistsError(
            f"Ensemble outputs already exist: {existing_outputs}; use --force to replace them"
        )
    if args.oof_inputs and len(args.oof_inputs) != len(args.prediction_inputs):
        parser.error("provide one matching --oof-inputs path per prediction input")
    if args.method == "weighted" and not args.oof_inputs:
        parser.error("weighted ensembles require matching --oof-inputs")
    if args.method == "weighted":
        scores = np.asarray([cv_score(path) for path in args.oof_inputs], dtype=float)
        if not np.isfinite(scores).all() or (scores < 0).any():
            raise ValueError("OOF ST-RAE scores must be finite and nonnegative")
        if (scores == 0).any():
            weights = (scores == 0).astype(float) / (scores == 0).sum()
        else:
            weights = (1.0 / scores) / np.sum(1.0 / scores)
    else:
        weights = np.repeat(1 / len(args.prediction_inputs), len(args.prediction_inputs))
    merged = None
    reference_keys = None
    reference_smiles = None
    prediction_identities = []
    for index, path in enumerate(args.prediction_inputs):
        frame = pd.read_csv(path)
        required = set(KEYS + ["canonical_smiles", "y_pred"])
        if not required.issubset(frame):
            raise ValueError(f"{path} lacks {sorted(required - set(frame))}")
        if frame.duplicated(KEYS).any():
            raise ValueError(f"{path} contains duplicate molecule/CYP keys")
        if frame["canonical_smiles"].isna().any():
            raise ValueError(f"{path} contains missing canonical molecule identities")
        if not np.isfinite(frame["y_pred"].to_numpy(dtype=float)).all():
            raise ValueError(f"{path} contains non-finite predictions")
        prediction_identities.append(_run_identity(frame, path))
        keys = set(map(tuple, frame[KEYS].itertuples(index=False, name=None)))
        if reference_keys is None:
            reference_keys = keys
            reference_smiles = frame.set_index(KEYS)["canonical_smiles"].sort_index()
        elif keys != reference_keys:
            raise ValueError("Prediction inputs do not contain identical molecule/CYP keys")
        elif not frame.set_index(KEYS)["canonical_smiles"].sort_index().equals(reference_smiles):
            raise ValueError("Prediction inputs disagree on canonical molecule identity")
        context_columns = [column for column in ("canonical_smiles",) if column in frame]
        values = frame[KEYS + context_columns + ["y_pred"]].rename(
            columns={"y_pred": f"prediction_{index}"}
        )
        if merged is not None:
            values = values.drop(columns=context_columns)
        merged = values if merged is None else merged.merge(values, on=KEYS, how="inner", validate="one_to_one")
    columns = [f"prediction_{index}" for index in range(len(args.prediction_inputs))]
    merged["y_pred"] = merged[columns].to_numpy() @ weights
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged["model"] = "ensemble"
    merged["fold"] = -1
    merged["seed"] = -1
    merged["prediction_scale"] = "original"
    output_columns = KEYS + [column for column in ("canonical_smiles",) if column in merged] + [
        "y_pred", "model", "fold", "seed", "prediction_scale",
    ]
    oof_output = None
    if args.oof_inputs:
        oof_frames = []
        oof_identities = []
        for index, (path, weight) in enumerate(zip(args.oof_inputs, weights)):
            frame = pd.read_csv(path)
            required = set(KEYS + ["canonical_smiles", "y_true", "y_pred", "lower", "upper"])
            if not required.issubset(frame):
                raise ValueError(f"{path} lacks {sorted(required - set(frame))}")
            if frame.duplicated(KEYS).any():
                raise ValueError(f"{path} contains duplicate OOF molecule/CYP keys")
            if frame["canonical_smiles"].isna().any():
                raise ValueError(f"{path} contains missing canonical molecule identities")
            if not np.isfinite(frame["y_pred"].to_numpy(dtype=float)).all():
                raise ValueError(f"{path} contains non-finite OOF predictions")
            oof_identities.append(_run_identity(frame, path))
            oof_frames.append(frame.assign(component=index, component_weight=weight))
        if oof_identities != prediction_identities:
            raise ValueError(
                "Prediction and OOF inputs are not paired in the same model/fold/seed order: "
                f"predictions={prediction_identities}, OOF={oof_identities}"
            )
        complementarity = residual_complementarity(
            {
                f"{identity[0]}|fold{identity[1]}|seed{identity[2]}": frame
                for identity, frame in zip(oof_identities, oof_frames)
            }
        )
        oof = pd.concat(oof_frames, ignore_index=True)
        if (oof.groupby(KEYS)["canonical_smiles"].nunique(dropna=False) > 1).any():
            raise ValueError("OOF components disagree on canonical molecule identity")
        if not _consistent_numeric_groups(oof, ["y_true", "lower", "upper", "fold"]):
            raise ValueError("OOF components disagree on truth or interval values")
        oof["weighted_prediction"] = oof["y_pred"] * oof["component_weight"]
        grouped = oof.groupby(KEYS, sort=True, as_index=False).agg(
            canonical_smiles=("canonical_smiles", "first"),
            y_true=("y_true", "first"), lower=("lower", "first"), upper=("upper", "first"),
            fold=("fold", "first"),
            weighted_sum=("weighted_prediction", "sum"), weight_sum=("component_weight", "sum"),
            components=("component", "nunique"),
        )
        grouped["y_pred"] = grouped["weighted_sum"] / grouped["weight_sum"]
        grouped["model"] = "ensemble"
        grouped["seed"] = -1
        grouped["prediction_scale"] = "original"
        if not np.isfinite(grouped["y_pred"].to_numpy(dtype=float)).all():
            raise ValueError("OOF ensemble produced non-finite predictions")
        oof_output = grouped[KEYS + [
            "canonical_smiles", "fold", "y_true", "y_pred", "lower", "upper", "model", "seed",
            "prediction_scale", "components",
        ]]
    merged[output_columns].to_csv(args.output_dir / "ensemble_predictions.csv", index=False)
    merged[output_columns].to_csv(args.output_dir / "test_ensemble_predictions.csv", index=False)
    if oof_output is not None:
        oof_output.to_csv(args.output_dir / "oof_ensemble_predictions.csv", index=False)
        # Written only once every OOF validation has passed, so a rejected ensemble
        # never leaves a stray file behind that the next run would have to --force past.
        complementarity.to_csv(args.output_dir / "oof_residual_complementarity.csv", index=False)
    json_dump(
        {
            "method": args.method,
            "inputs": [str(path.resolve()) for path in args.prediction_inputs],
            "oof_inputs": [str(path.resolve()) for path in args.oof_inputs],
            "run_identities": prediction_identities,
            "weights": weights.tolist(),
        },
        args.output_dir / "ensemble_metadata.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
