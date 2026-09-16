#!/usr/bin/env python3
"""Train an official Chemprop v2 multitask regression baseline on saved folds."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from src.data import load_direct_data, save_split_ids, smoke_subset
from src.metrics import regression_metrics
from src.run_management import begin_run, finish_run
from src.utils import add_common_args, json_dump, load_config, normalize_device, seed_everything
from src.validation import audit_featurization, leakage_checks, validate_predictions


CHEMPROP_VERSION = "2.2.1"


def require_chemprop() -> tuple[str, str]:
    import torch

    bundled_executable = Path(sys.executable).with_name("chemprop")
    executable = str(bundled_executable) if bundled_executable.is_file() else shutil.which("chemprop")
    if executable is None or not Path(executable).is_file():
        raise RuntimeError(
            "Official Chemprop v2 is required. Create the modelling environment from "
            "environment-modeling.yml or environment-modeling-cuda.yml."
        )
    result = subprocess.run(
        [sys.executable, "-c", "import chemprop; print(chemprop.__version__)"],
        check=True, capture_output=True, text=True,
    )
    version = result.stdout.strip()
    if version != CHEMPROP_VERSION:
        raise RuntimeError(
            f"This baseline is pinned to Chemprop {CHEMPROP_VERSION}; found {version}. "
            "Use the supplied environment file."
        )
    torch_version = tuple(int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2])
    if not (2, 3) <= torch_version < (2, 6):
        raise RuntimeError(
            f"Chemprop {CHEMPROP_VERSION} requires PyTorch >=2.3,<2.6 in this pipeline; "
            f"found {torch.__version__}. Use the supplied environment file."
        )
    return executable, version


def write_chemprop_inputs(bundle, run_dir: Path) -> tuple[Path, Path]:
    selected = bundle.train_mask | bundle.validation_mask
    frame = bundle.frame.loc[selected].copy()
    roles = np.where(bundle.train_mask.loc[frame.index], "train", "val")
    result = pd.DataFrame(
        {
            "Molecule_Name": frame["Molecule_Name"],
            "SMILES": frame["canonical_smiles"],
            "split": roles,
        }
    )
    for target in bundle.target_columns:
        result[target] = frame[target].to_numpy()
    if result["Molecule_Name"].duplicated().any() or result["SMILES"].isna().any():
        raise ValueError("Chemprop input construction produced invalid molecule identities")
    train_path = run_dir / "chemprop_input.csv"
    validation_path = run_dir / "chemprop_validation_input.csv"
    result.to_csv(train_path, index=False)
    result.loc[result["split"].eq("val"), ["Molecule_Name", "SMILES"]].to_csv(
        validation_path, index=False
    )
    return train_path, validation_path


def chemprop_command(
    executable: str, config: dict, input_path: Path, output_dir: Path, seed: int, device: str
) -> list[str]:
    model = config["model"]
    training = config["training"]
    command = [
        executable, "train", "--data-path", str(input_path), "--output-dir", str(output_dir),
        "--smiles-columns", "SMILES", "--target-columns", *[
            f"{cyp}_pIC50_direct_inhibition" for cyp in config["data"]["cyps"]
        ],
        "--splits-column", "split", "--task-type", "regression", "--loss-function", "mse",
        "--metrics", "mae", "rmse", "r2", "--tracking-metric", "mae",
        "--message-hidden-dim", str(model["message_hidden_dim"]),
        "--depth", str(model["message_passing_depth"]), "--dropout", str(model["dropout"]),
        "--aggregation", str(model["aggregation"]), "--ffn-hidden-dim", str(model["ffn_hidden_dim"]),
        "--ffn-num-layers", str(model["ffn_num_layers"]),
        "--batch-size", str(training["batch_size"]), "--epochs", str(training["max_epochs"]),
        "--patience", str(training["early_stopping_patience"]),
        "--warmup-epochs", str(training["warmup_epochs"]),
        "--init-lr", str(training["init_learning_rate"]),
        "--max-lr", str(training["max_learning_rate"]),
        "--final-lr", str(training["final_learning_rate"]),
        "--grad-clip", str(training["gradient_clip_norm"]),
        "--num-workers", str(training.get("num_workers", 0)), "--pytorch-seed", str(seed),
        "--data-seed", str(seed), "--save-smiles-splits",
    ]
    normalized = normalize_device(device)
    if normalized.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA requested but unavailable: {normalized}")
        command.extend(["--accelerator", "gpu", "--devices", normalized.split(":", 1)[1]])
    elif normalized == "cpu":
        command.extend(["--accelerator", "cpu", "--devices", "1"])
    else:
        raise ValueError(f"Unsupported device for official Chemprop: {device}")
    return command


def find_native_model(chemprop_dir: Path) -> Path:
    candidates = sorted(chemprop_dir.glob("model_*/best.pt"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one native Chemprop best.pt under {chemprop_dir}; found {candidates}"
        )
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train official Chemprop v2 multitask CYP regression")
    add_common_args(parser, "configs/chemprop_official_multitask.yaml")
    parser.add_argument("--cyp", action="append", default=[])
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    if args.experiment:
        config["experiment"] = args.experiment
    if args.cyp:
        config["data"]["cyps"] = args.cyp
    if args.max_runtime_minutes is not None:
        raise ValueError(
            "Official Chemprop v2.2 CLI has no walltime-safe epoch-boundary stop or exact resume. "
            "Do not use --max-runtime-minutes for this baseline; request sufficient walltime instead."
        )
    if args.smoke_test:
        args.run_name = args.run_name or "smoke"
        config["training"]["max_epochs"] = min(2, int(config["training"]["max_epochs"]))
        config["training"]["early_stopping_patience"] = 2
        # Chemprop requires at least one post-warmup epoch.  Keep the tiny smoke
        # run valid even when the regular configuration warms up for longer.
        config["training"]["warmup_epochs"] = min(
            int(config["training"].get("warmup_epochs", 0)),
            max(0, int(config["training"]["max_epochs"]) - 1),
        )
    if args.resume:
        raise RuntimeError(
            "Native Chemprop v2.2 CLI does not restore Lightning optimizer/scheduler state. "
            "This baseline deliberately refuses unsafe partial resume; rerun with a new run identity."
        )
    executable, chemprop_version = require_chemprop()
    seed_everything(args.seed, bool(config.get("reproducibility", {}).get("deterministic", True)))
    run, metadata, started = begin_run(args, config)
    if metadata.get("already_completed"):
        print(f"Run already completed: {run.run_dir}")
        return 0
    bundle = load_direct_data(
        config["data"].get("split_scheme", "ecfp_cluster"), args.fold, config["data"].get("cyps")
    )
    if args.smoke_test:
        bundle = smoke_subset(bundle)
    leakage_checks(bundle, run.run_dir)
    audit_featurization(bundle, run.run_dir, config.get("featurization", {}).get("policy", "fail"))
    save_split_ids(bundle, run.run_dir)
    input_path, validation_path = write_chemprop_inputs(bundle, run.run_dir)
    chemprop_dir = run.run_dir / "chemprop"
    command = chemprop_command(executable, config, input_path, chemprop_dir, args.seed, args.device)
    json_dump({"command": command, "chemprop_version": chemprop_version}, run.run_dir / "chemprop_command.json")
    environment = dict(os.environ)
    environment["MPLCONFIGDIR"] = str(run.run_dir / ".matplotlib")
    (run.run_dir / ".matplotlib").mkdir(exist_ok=True)
    with (run.logs / "chemprop_stdout.log").open("w", encoding="utf-8") as handle:
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT, env=environment)
    model_path = find_native_model(chemprop_dir)
    prediction_path = run.run_dir / "chemprop_validation_predictions.csv"
    predict_command = [
        executable, "predict", "--test-path", str(validation_path), "--preds-path", str(prediction_path),
        "--model-paths", str(model_path), "--smiles-columns", "SMILES",
    ]
    if normalize_device(args.device).startswith("cuda"):
        predict_command.extend(["--accelerator", "gpu", "--devices", normalize_device(args.device).split(":", 1)[1]])
    else:
        predict_command.extend(["--accelerator", "cpu", "--devices", "1"])
    with (run.logs / "chemprop_predict_stdout.log").open("w", encoding="utf-8") as handle:
        subprocess.run(predict_command, check=True, stdout=handle, stderr=subprocess.STDOUT, env=environment)
    predicted = pd.read_csv(prediction_path)
    prediction_columns = {
        target: f"prediction_{target}" for target in bundle.target_columns if target in predicted
    }
    predicted = predicted.rename(columns=prediction_columns)
    observed = bundle.frame.loc[bundle.validation_mask].copy()
    # The native prediction file contains the canonical input SMILES, whereas
    # the source frame preserves the supplied SMILES in its SMILES column.
    # Join on the canonical representation used to train Chemprop.
    observed["SMILES"] = observed["canonical_smiles"]
    merged = observed.merge(predicted, on=["Molecule_Name", "SMILES"], how="left", validate="one_to_one")
    rows = []
    for cyp, target, lower, upper, uncertainty in zip(
        bundle.task_names, bundle.target_columns, bundle.lower_columns, bundle.upper_columns,
        bundle.uncertainty_columns,
    ):
        prediction_column = f"prediction_{target}"
        if prediction_column not in predicted:
            raise ValueError(f"Chemprop prediction output lacks target column {target}")
        for row in merged.loc[merged[target].notna()].itertuples(index=False):
            rows.append(
                {
                    "molecule_id": row.Molecule_Name, "canonical_smiles": row.SMILES,
                    "fold": args.fold, "CYP": cyp, "y_true": getattr(row, target),
                    "y_pred": getattr(row, prediction_column),
                    "experimental_uncertainty": getattr(row, uncertainty), "lower": getattr(row, lower),
                    "upper": getattr(row, upper),
                    "model": config.get("model_id", config["experiment"]), "seed": args.seed,
                    "prediction_scale": "original",
                }
            )
    predictions = pd.DataFrame(rows)
    expected_keys = {
        (row.Molecule_Name, cyp)
        for cyp, target in zip(bundle.task_names, bundle.target_columns)
        for row in bundle.frame.loc[bundle.validation_mask & bundle.frame[target].notna()].itertuples()
    }
    integrity = validate_predictions(predictions, expected_keys, bundle.task_names, True)
    json_dump(integrity, run.run_dir / "prediction_integrity.json")
    predictions.to_csv(run.run_dir / "predictions.csv", index=False)
    per_cyp, macro = regression_metrics(predictions)
    per_cyp.to_csv(run.run_dir / "metrics_per_cyp.csv", index=False)
    json_dump(macro, run.run_dir / "metrics.json")
    finish_run(
        run, metadata, started,
        {
            "completed_CYPs": list(bundle.task_names), "best_epoch": None,
            "selection_metric": "MAE", "best_validation_metric": None,
            "final_metrics": macro, "precision_mode": "native_chemprop",
            "metric_direction": "minimize", "chemprop_version": chemprop_version,
            "native_chemprop_checkpoint": str(model_path),
            "native_resume_supported": False,
        },
    )
    print(f"Run completed: {run.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
