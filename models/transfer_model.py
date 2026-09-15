from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.constants import FEATURE_SCHEMA_VERSION, MODEL_CODE_VERSION
from src.utils import file_sha256


ARCHITECTURE_FIELDS = (
    "type", "message_hidden_dim", "message_passing_depth", "dropout", "aggregation"
)


def validate_encoder_checkpoint(
    checkpoint: dict[str, Any], expected_model_config: dict[str, Any] | None = None,
    partial_load: bool = False,
) -> dict[str, Any]:
    """Validate the native encoder checkpoint contract before examining tensors."""
    required_metadata = {
        "architecture_type", "model_config", "task_names", "model_code_version",
        "feature_schema_version", "pytorch_version", "chemprop_version",
    }
    missing_metadata = sorted(required_metadata - set(checkpoint))
    raw_source_config = checkpoint.get("model_config", {})
    source_config = raw_source_config if isinstance(raw_source_config, dict) else {}
    missing_config = sorted(field for field in ARCHITECTURE_FIELDS if field not in source_config)
    tasks = checkpoint.get("task_names")
    invalid_metadata = []
    if not isinstance(raw_source_config, dict):
        invalid_metadata.append("model_config must be a mapping")
    if (
        not isinstance(tasks, (list, tuple)) or not tasks
        or any(not isinstance(task, str) or not task for task in tasks)
        or len(set(tasks)) != len(tasks)
    ):
        invalid_metadata.append("task_names must be a nonempty unique string sequence")
    metadata_mismatches = {}
    expected_metadata = {
        "architecture_type": "dmpnn",
        "model_code_version": MODEL_CODE_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
    }
    for field, expected in expected_metadata.items():
        if checkpoint.get(field) != expected:
            metadata_mismatches[field] = {
                "source": checkpoint.get(field), "expected": expected,
            }
    config_mismatches = {
        field: {"source": source_config.get(field), "requested": expected_model_config.get(field)}
        for field in ARCHITECTURE_FIELDS
        if expected_model_config is not None
        and source_config.get(field) != expected_model_config.get(field)
    }
    problems = (
        missing_metadata or missing_config or invalid_metadata
        or metadata_mismatches or config_mismatches
    )
    if problems and not partial_load:
        raise ValueError(
            "Incompatible encoder checkpoint; expected a native OpenADMET D-MPNN checkpoint: "
            f"missing_metadata={missing_metadata}, missing_model_config={missing_config}, "
            f"invalid_metadata={invalid_metadata}, "
            f"metadata_mismatches={metadata_mismatches}, config_mismatches={config_mismatches}. "
            "Third-party Chemprop key layouts are not translated by this loader; use a tested "
            "adapter or --partial-load only when the native tensor layout is known to match."
        )
    return {
        "missing_metadata": missing_metadata,
        "missing_model_config": missing_config,
        "invalid_metadata": invalid_metadata,
        "metadata_mismatches": metadata_mismatches,
        "config_mismatches": config_mismatches,
    }


def load_encoder_weights(
    model, checkpoint_path: str | Path, expected_model_config: dict[str, Any] | None = None,
    partial_load: bool = False,
) -> dict[str, Any]:
    path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata_diagnostics = validate_encoder_checkpoint(
        checkpoint, expected_model_config, partial_load
    )
    state = checkpoint.get("encoder_state_dict")
    if state is None:
        full = checkpoint.get("model_state_dict", checkpoint)
        state = {
            key.removeprefix("encoder."): value
            for key, value in full.items()
            if key.startswith("encoder.")
        }
    if not state:
        raise ValueError(f"No encoder weights found in {path}")
    source_config = checkpoint.get("model_config", {})
    if not isinstance(source_config, dict):
        source_config = {}
    config_mismatches = metadata_diagnostics["config_mismatches"]
    destination = model.encoder.state_dict()
    compatible = {
        key: value for key, value in state.items()
        if key in destination and destination[key].shape == value.shape
    }
    skipped_shape_keys = sorted(
        key for key, value in state.items()
        if key in destination and destination[key].shape != value.shape
    )
    source_only_keys = sorted(set(state) - set(destination))
    missing_destination_keys = sorted(set(destination) - set(compatible))
    if not compatible:
        raise ValueError(f"Incompatible encoder checkpoint {path}: no shape-compatible encoder weights")
    if (config_mismatches or skipped_shape_keys or source_only_keys or missing_destination_keys) and not partial_load:
        raise ValueError(
            f"Incompatible encoder checkpoint {path}: config_mismatches={config_mismatches}, "
            f"shape_mismatches={skipped_shape_keys}, source_only={source_only_keys}, "
            f"missing_destination={missing_destination_keys}; use --partial-load only when intentional"
        )
    incompatible = model.encoder.load_state_dict(compatible, strict=False)
    return {
        "checkpoint": str(path),
        "sha256": file_sha256(path),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "source_only_keys": source_only_keys,
        "skipped_shape_keys": skipped_shape_keys,
        "loaded_key_count": len(compatible),
        "config_mismatches": config_mismatches,
        "partial_load": partial_load,
        "source_architecture_type": checkpoint.get("architecture_type", source_config.get("type")),
        "source_model_code_version": checkpoint.get("model_code_version"),
        "source_pytorch_version": checkpoint.get("pytorch_version"),
        "source_chemprop_version": checkpoint.get("chemprop_version"),
        **metadata_diagnostics,
    }
