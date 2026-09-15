#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import _bootstrap  # noqa: F401
from models.transfer_model import load_encoder_weights
from src.constants import ANALYSIS_TABLES, DATA_DIR
from src.data import load_direct_data
from src.training import build_model
from src.utils import load_config, normalize_device, system_metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenADMET cluster pre-flight checks")
    parser.add_argument("--config", type=Path, default=Path("configs/chemprop_multitask.yaml"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--run-smoke", action="store_true")
    args = parser.parse_args()
    failures = []
    required = [
        DATA_DIR / "cyp-challenge-TRAIN_inhibition.csv",
        DATA_DIR / "cyp-challenge-single-concentration-TRAIN.csv",
        DATA_DIR / "cyp-challenge-TEST-BLINDED.csv",
        ANALYSIS_TABLES / "train_molecule_identity_and_properties.csv",
        ANALYSIS_TABLES / "test_molecule_identity_and_properties.csv",
        ANALYSIS_TABLES / "cv_fold_assignments.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        failures.append(f"missing files: {missing}")
    try:
        config = load_config(args.config)
        load_direct_data(config.get("data", {}).get("split_scheme", "ecfp_cluster"), 0)
        if config.get("model", {}).get("type") == "chemprop_official_v2":
            import chemprop

            if chemprop.__version__ != "2.2.1":
                raise ValueError(f"expected Chemprop 2.2.1, found {chemprop.__version__}")
    except Exception as exc:
        failures.append(f"data/fold validation: {type(exc).__name__}: {exc}")
        config = None
    metadata = system_metadata(args.device)
    device = normalize_device(args.device)
    if device.startswith("cuda") and not metadata.get("cuda_available"):
        failures.append(f"CUDA requested but unavailable: {device}")
    if metadata.get("gpu_metadata_error"):
        failures.append(f"invalid CUDA selection: {metadata['gpu_metadata_error']}")
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=args.output_dir, prefix=".preflight_", delete=True):
            pass
    except OSError as exc:
        failures.append(f"output directory is not writable: {exc}")
    if args.checkpoint and config:
        try:
            if config.get("model", {}).get("type") == "chemprop_official_v2":
                import torch

                checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
                if not checkpoint.get("output_columns") or "state_dict" not in checkpoint:
                    raise ValueError("native Chemprop checkpoint lacks output columns or state")
            else:
                import torch

                checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
                tasks = tuple(checkpoint.get("task_names", ()))
                if not tasks:
                    raise ValueError("checkpoint has no task list")
                model = build_model(tasks, config)
                load_encoder_weights(model, args.checkpoint, config.get("model", {}), partial_load=False)
        except Exception as exc:
            failures.append(f"checkpoint compatibility: {type(exc).__name__}: {exc}")
    if failures:
        print("PRE-FLIGHT FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(
        f"PRE-FLIGHT PASSED: python={metadata.get('python')} torch={metadata.get('torch')} "
        f"rdkit={metadata.get('rdkit')} device={device} data/folds=valid output=writable"
    )
    if args.run_smoke:
        script = (
            "scripts/train_chemprop_official.py"
            if config and config.get("model", {}).get("type") == "chemprop_official_v2"
            else "scripts/train_multitask.py"
        )
        command = [
            sys.executable, script, "--config", str(args.config),
            "--fold", "0", "--seed", "42", "--device", args.device,
            "--output-dir", str(args.output_dir), "--experiment", "preflight_smoke",
            "--run-name", f"run_{time.time_ns()}", "--smoke-test",
        ]
        result = subprocess.run(command, check=False)
        if result.returncode:
            print(f"PRE-FLIGHT SMOKE FAILED: return_code={result.returncode}")
            return result.returncode
        print("PRE-FLIGHT SMOKE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
