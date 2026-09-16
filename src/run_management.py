from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from .constants import ANALYSIS_TABLES, DATA_DIR
from .model_registry import update_model_statistics
from .utils import (
    RunContext, capture_environment, file_sha256, json_dump, require_yaml, resolve_run_dir,
    system_metadata, utc_now,
)


def begin_run(args: argparse.Namespace, config: dict[str, Any]) -> tuple[RunContext, dict[str, Any], float]:
    experiment = str(config.get("experiment", "unnamed_experiment"))
    run_name = args.run_name or config.get("run_name")
    # A run name selects a model/configuration variant inside an experiment family.
    # Keep the family name for directory grouping, but give predictions a distinct
    # identity so OOF aggregation cannot mix different matrix/grid variants.
    model_id = str(config.get("model_id") or (
        f"{experiment}::{run_name}" if run_name else experiment
    ))
    config["model_id"] = model_id
    run_dir = resolve_run_dir(args.output_dir, experiment, args.fold, args.seed, run_name)
    run = RunContext(run_dir, args.resume)
    started = time.monotonic()
    yaml = require_yaml()
    config_path = run_dir / "config.yaml"
    if args.resume and config_path.exists():
        with config_path.open(encoding="utf-8") as handle:
            previous_config = yaml.safe_load(handle) or {}
        # Runs started before model_id was introduced remain resumable; the identity
        # is deterministic from the existing experiment/run-name path.
        previous_config.setdefault("model_id", model_id)
        if previous_config != config:
            raise ValueError("Resolved configuration differs from the interrupted run")
    if run.completed.exists() and args.resume:
        required_artifacts = [
            run_dir / "metadata.json", run_dir / "predictions.csv",
            run_dir / "metrics.json", run_dir / "split_molecules.csv",
        ]
        missing_artifacts = [str(path) for path in required_artifacts if not path.exists()]
        checkpoints = list(run.checkpoints.glob("*"))
        completed_metadata = (
            json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
            if (run_dir / "metadata.json").exists() else {}
        )
        if missing_artifacts or not checkpoints or completed_metadata.get("status") != "completed":
            raise ValueError(
                "COMPLETED marker is inconsistent with run artifacts: "
                f"missing={missing_artifacts}, checkpoints={len(checkpoints)}, "
                f"metadata_status={completed_metadata.get('status')!r}"
            )
        return run, {"already_completed": True}, time.monotonic()
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)
    capture_environment(run_dir / "environment_pip_freeze.txt")
    arguments = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    json_dump(arguments, run_dir / "cli_args.json")
    input_paths = [
        DATA_DIR / "cyp-challenge-TRAIN_inhibition.csv",
        DATA_DIR / "cyp-challenge-single-concentration-TRAIN.csv",
        DATA_DIR / "cyp-challenge-TEST-BLINDED.csv",
        ANALYSIS_TABLES / "train_molecule_identity_and_properties.csv",
        ANALYSIS_TABLES / "test_molecule_identity_and_properties.csv",
        ANALYSIS_TABLES / "cv_fold_assignments.csv",
    ]
    metadata_path = run_dir / "metadata.json"
    previous_metadata = {}
    if args.resume and metadata_path.exists():
        previous_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    command = [sys.executable, *sys.argv]
    command_history = list(previous_metadata.get("command_history", []))
    command_history.append(command)
    model_type = config.get("model", {}).get("type")
    model_hyperparameters = (
        config.get("lightgbm", {}) if model_type == "lightgbm" else config.get("model", {})
    )
    metadata = {
        **previous_metadata,
        **system_metadata(args.device),
        "start_time": previous_metadata.get("start_time", utc_now()),
        "last_resume_time": utc_now() if previous_metadata else None,
        "seed": args.seed,
        "seeds": {
            "python": args.seed, "numpy": args.seed, "pytorch": args.seed,
            "cuda": args.seed if str(args.device).lower().startswith(("cuda", "gpu")) else None,
        },
        "deterministic_mode": bool(config.get("reproducibility", {}).get("deterministic", True)),
        "deterministic_algorithms_warn_only": bool(
            config.get("reproducibility", {}).get("deterministic", True)
        ),
        "fold": args.fold,
        "experiment": experiment,
        "model_id": model_id,
        "output_root": str(Path(args.output_dir).resolve()),
        "model_type": model_type,
        "chemprop_implementation": "self-contained-chemprop-style" if model_type == "dmpnn" else "not-applicable",
        "model_hyperparameters": model_hyperparameters,
        "training_hyperparameters": config.get(
            "training", config.get("pretraining", config.get("transfer", {}))
        ),
        "input_hashes": {str(path.resolve()): file_sha256(path) for path in input_paths},
        "command": command,
        "command_history": command_history,
        "status": "running",
    }
    json_dump(metadata, run_dir / "metadata.json")
    run.log("run_started", experiment=experiment, fold=args.fold, seed=args.seed)
    previous_excepthook = sys.excepthook

    def record_failure(exception_type, exception, traceback) -> None:
        session_runtime = time.monotonic() - started
        previous_runtime = float(
            metadata.get("cumulative_runtime_seconds", metadata.get("runtime_seconds", 0)) or 0
        )
        metadata.update(
            {
                "end_time": utc_now(), "runtime_seconds": session_runtime,
                "cumulative_runtime_seconds": previous_runtime + session_runtime,
                "status": "failed",
                "error": f"{exception_type.__name__}: {exception}",
            }
        )
        json_dump(metadata, run_dir / "metadata.json")
        run.log("run_failed", error=metadata["error"])
        previous_excepthook(exception_type, exception, traceback)

    sys.excepthook = record_failure
    return run, metadata, started


def finish_run(
    run: RunContext, metadata: dict[str, Any], started: float,
    result: dict[str, Any], completed: bool = True,
) -> None:
    session_runtime = time.monotonic() - started
    previous_runtime = float(
        metadata.get("cumulative_runtime_seconds", metadata.get("runtime_seconds", 0)) or 0
    )
    metadata.update(result)
    metadata.update(
        {
            "end_time": utc_now(),
            "runtime_seconds": session_runtime,
            "cumulative_runtime_seconds": previous_runtime + session_runtime,
            "status": "completed" if completed else "checkpointed",
        }
    )
    json_dump(metadata, run.run_dir / "metadata.json")
    run.log("run_finished", status=metadata["status"])
    if completed:
        run.mark_complete()
        output_root_value = metadata.get("output_root")
        if not output_root_value:
            metadata["model_statistics_error"] = "Missing output_root in run metadata"
            json_dump(metadata, run.run_dir / "metadata.json")
            run.log("model_statistics_update_failed", error=metadata["model_statistics_error"])
            print(
                f"WARNING: model completed but statistics registry update failed: "
                f"{metadata['model_statistics_error']}",
                file=sys.stderr,
            )
            return
        output_root = Path(output_root_value)
        registry_path = output_root / "experiment_leaderboard.csv"
        try:
            statistics = update_model_statistics(output_root, registry_path)
            run.log(
                "model_statistics_updated",
                registry=str(registry_path), rows=len(statistics),
            )
        except Exception as exc:
            metadata["model_statistics_error"] = f"{type(exc).__name__}: {exc}"
            json_dump(metadata, run.run_dir / "metadata.json")
            run.log("model_statistics_update_failed", error=metadata["model_statistics_error"])
            print(
                f"WARNING: model completed but statistics registry update failed: "
                f"{metadata['model_statistics_error']}",
                file=sys.stderr,
            )
