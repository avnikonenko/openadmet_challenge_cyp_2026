from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .constants import DEFAULT_OUTPUT_ROOT, PROJECT_ROOT


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
            stderr=subprocess.DEVNULL, timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def require_yaml():
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required for modelling CLIs. Create the modelling environment "
            "from environment-modeling.yml."
        ) from exc
    return yaml


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    yaml = require_yaml()
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override must be key=value: {override}")
        dotted_key, raw = override.split("=", 1)
        value = yaml.safe_load(raw)
        cursor = config
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):
                raise ValueError(f"Cannot set nested override {dotted_key}")
        cursor[parts[-1]] = value
    return config


def add_common_args(parser: argparse.ArgumentParser, default_config: str) -> None:
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-runtime-minutes", type=float, default=None)
    parser.add_argument("--smoke-test", "--dry-run", action="store_true", dest="smoke_test")
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="Repeatable dotted YAML override, e.g. model.message_hidden_dim=600",
    )


def normalize_device(device: str) -> str:
    text = str(device).strip().lower()
    return f"cuda:{text}" if text.isdigit() else text


def seed_everything(seed: int, deterministic: bool = True) -> dict[str, Any]:
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return {"python": seed, "numpy": seed, "pytorch": None, "cuda": None, "deterministic": deterministic}
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=deterministic)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    return {
        "python": seed, "numpy": seed, "pytorch": seed,
        "cuda": seed if torch.cuda.is_available() else None,
        "deterministic": deterministic,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def resolve_run_dir(
    output_root: Path, experiment: str, fold: int, seed: int, run_name: str | None = None
) -> Path:
    path = output_root / experiment / f"fold{fold}" / f"seed{seed}"
    return path / run_name if run_name else path


def json_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
    temporary.replace(path)


def atomic_torch_save(value: Any, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def system_metadata(device: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "git_commit": git_commit(),
        "device_requested": device,
    }
    try:
        import torch

        result.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
            }
        )
        normalized = normalize_device(device)
        if normalized.startswith("cuda") and torch.cuda.is_available():
            index = int(normalized.split(":", 1)[1]) if ":" in normalized else 0
            if index < 0 or index >= torch.cuda.device_count():
                result["gpu_metadata_error"] = (
                    f"requested CUDA index {index}, visible device count {torch.cuda.device_count()}"
                )
            else:
                result["gpu_name"] = torch.cuda.get_device_name(index)
    except ImportError:
        result["torch"] = "not-installed"
    for module_name, output_name in (
        ("numpy", "numpy"), ("pandas", "pandas"), ("sklearn", "scikit_learn"),
        ("rdkit", "rdkit"), ("lightgbm", "lightgbm"), ("chemprop", "chemprop"),
    ):
        try:
            module = __import__(module_name)
            result[output_name] = getattr(module, "__version__", "unknown")
        except (ImportError, RuntimeError, ValueError, AssertionError):
            result[output_name] = "not-installed"
    return result


def capture_environment(path: Path) -> None:
    try:
        output = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True,
            stderr=subprocess.STDOUT, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        output = f"environment capture failed: {type(exc).__name__}: {exc}\n"
    path.write_text(output, encoding="utf-8")


@dataclass
class RuntimeGuard:
    max_minutes: float | None
    started: float = field(default_factory=time.monotonic)

    def should_stop(self) -> bool:
        return self.max_minutes is not None and (
            time.monotonic() - self.started
        ) >= 60 * self.max_minutes

    def can_start_epoch(self, estimated_epoch_seconds: float | None, margin_seconds: float = 30) -> bool:
        if self.max_minutes is None or estimated_epoch_seconds is None:
            return True
        elapsed = time.monotonic() - self.started
        remaining = 60 * self.max_minutes - elapsed
        return remaining >= estimated_epoch_seconds + margin_seconds


class GracefulStop:
    def __init__(self):
        self.requested = False
        self.signal_name: str | None = None
        self._previous: dict[int, Any] = {}

    def __enter__(self) -> "GracefulStop":
        for signum in (signal.SIGTERM, signal.SIGINT, getattr(signal, "SIGUSR1", signal.SIGTERM)):
            if signum in self._previous:
                continue
            self._previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle)
        return self

    def _handle(self, signum, _frame) -> None:
        self.requested = True
        self.signal_name = signal.Signals(signum).name

    def __exit__(self, *_args) -> None:
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)


class RunContext:
    def __init__(self, run_dir: Path, resume: bool):
        self.run_dir = run_dir
        self.checkpoints = run_dir / "checkpoints"
        self.logs = run_dir / "logs"
        self.completed = run_dir / "COMPLETED"
        if run_dir.exists() and any(run_dir.iterdir()) and not resume:
            raise FileExistsError(
                f"Run directory already exists at {run_dir}; use a new run identity or --resume"
            )
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **fields: Any) -> None:
        row = {"time": utc_now(), "event": event, **fields}
        with (self.logs / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")

    def mark_complete(self) -> None:
        self.completed.write_text(f"completed {utc_now()}\n", encoding="utf-8")
