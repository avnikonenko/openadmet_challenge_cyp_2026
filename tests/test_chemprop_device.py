from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import patch

from src.utils import load_config


ROOT = Path(__file__).resolve().parents[1]


def test_official_chemprop_cuda_device_index_is_a_list() -> None:
    with patch.object(sys, "path", [str(ROOT / "scripts"), *sys.path]):
        module = importlib.import_module("train_chemprop_official")
    config = load_config(ROOT / "configs/chemprop_official_multitask.yaml")
    with patch("torch.cuda.is_available", return_value=True):
        for device, expected in (("cuda:0", "0,"), ("cuda:1", "1,"), ("cuda:10", "10,")):
            command = module.chemprop_command(
                "chemprop", config, Path("input.csv"), Path("output"), 42, device
            )
            assert command[command.index("--accelerator") + 1] == "gpu"
            assert command[command.index("--devices") + 1] == expected


def test_official_chemprop_cpu_device_is_unchanged() -> None:
    with patch.object(sys, "path", [str(ROOT / "scripts"), *sys.path]):
        module = importlib.import_module("train_chemprop_official")
    config = load_config(ROOT / "configs/chemprop_official_multitask.yaml")
    command = module.chemprop_command(
        "chemprop", config, Path("input.csv"), Path("output"), 42, "cpu"
    )
    assert command[command.index("--accelerator") + 1] == "cpu"
    assert command[command.index("--devices") + 1] == "1"
