from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.constants import FEATURE_SCHEMA_VERSION, MODEL_CODE_VERSION
from src.utils import file_sha256


ARCHITECTURE_FIELDS = (
    "type", "message_hidden_dim", "message_passing_depth", "dropout", "aggregation"
)
GINE_ARCHITECTURE_FIELDS = (
    "type", "hidden_dim", "num_layers", "dropout", "residual", "normalization", "pooling"
)
ARCHITECTURE_FIELDS_BY_TYPE = {
    "dmpnn": ARCHITECTURE_FIELDS,
    "gine": GINE_ARCHITECTURE_FIELDS,
}


def architecture_type(model_config: dict[str, Any] | None) -> str:
    """Encoder family recorded in checkpoints; the historical default is the D-MPNN."""
    return str((model_config or {}).get("type", "dmpnn")).lower()


def architecture_fields(model_config: dict[str, Any] | None) -> tuple[str, ...]:
    """Encoder hyperparameters that must match exactly for a transfer to be valid.

    Only encoder fields are compared. Prediction-head options and the
    `architecture` selector deliberately stay out of this set, because a pretrained
    encoder may legitimately be reused under a different head design.
    """
    return ARCHITECTURE_FIELDS_BY_TYPE.get(architecture_type(model_config), ARCHITECTURE_FIELDS)


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
    comparison_fields = architecture_fields(
        expected_model_config if expected_model_config is not None else source_config
    )
    missing_config = sorted(field for field in comparison_fields if field not in source_config)
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
        "architecture_type": architecture_type(
            expected_model_config if expected_model_config is not None else source_config
        ),
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
        for field in comparison_fields
        if expected_model_config is not None
        and source_config.get(field) != expected_model_config.get(field)
    }
    problems = (
        missing_metadata or missing_config or invalid_metadata
        or metadata_mismatches or config_mismatches
    )
    if problems and not partial_load:
        raise ValueError(
            "Incompatible encoder checkpoint; expected a native OpenADMET "
            f"{expected_metadata['architecture_type']} checkpoint: "
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
        "source_architecture": source_config.get("architecture", "cyp_specific_heads"),
        "source_model_code_version": checkpoint.get("model_code_version"),
        "source_pytorch_version": checkpoint.get("pytorch_version"),
        "source_chemprop_version": checkpoint.get("chemprop_version"),
        "source_stage": checkpoint.get("stage"),
        "source_task": _source_task(checkpoint),
        "source_loss_mode": checkpoint.get("loss_mode"),
        **metadata_diagnostics,
    }


def _source_task(checkpoint: dict[str, Any]) -> str | None:
    split_metadata = checkpoint.get("split_metadata")
    if isinstance(split_metadata, dict):
        return split_metadata.get("purpose")
    return None


def describe_checkpoint(path: str | Path) -> dict[str, Any]:
    """Provenance of one transfer-stage checkpoint, without loading it into a model."""
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Transfer stage checkpoint does not exist: {resolved}")
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
    source_config = checkpoint.get("model_config")
    source_config = source_config if isinstance(source_config, dict) else {}
    return {
        "checkpoint": str(resolved),
        "sha256": file_sha256(resolved),
        "source_task": _source_task(checkpoint),
        "architecture_type": checkpoint.get("architecture_type", source_config.get("type")),
        "architecture": source_config.get("architecture", "cyp_specific_heads"),
        "loss_mode": checkpoint.get("loss_mode"),
        "stage": checkpoint.get("stage"),
        "task_names": list(checkpoint.get("task_names", ())),
        "model_code_version": checkpoint.get("model_code_version"),
    }


def resolve_transfer_stages(
    config: dict[str, Any], cli_checkpoint: str | Path | None, current_stage: str,
) -> tuple[str | None, list[dict[str, Any]]]:
    """Resolve the encoder-initialization lineage for this run.

    Without a ``transfer.stages`` block the behaviour is exactly the historical one:
    the encoder is initialized from ``--checkpoint`` (or ``pretrained.checkpoint``) if
    one is given, and from scratch otherwise. With a ``stages`` block, the final entry
    is the stage being trained now and every earlier entry may supply a checkpoint;
    the last earlier stage carrying a checkpoint is the effective source, and
    ``--checkpoint`` overrides that stage's configured path. Either way the returned
    records are written to run metadata so each stage's path, hash, source task, and
    architecture are recorded.
    """
    configured = config.get("pretrained", {}).get("checkpoint")
    stages = (config.get("transfer") or {}).get("stages")
    if not stages:
        source = cli_checkpoint or configured
        records: list[dict[str, Any]] = []
        if source:
            records.append(
                {
                    "index": 0, "name": "pretrained_encoder", "role": "source",
                    **describe_checkpoint(source),
                }
            )
        records.append(
            {
                "index": len(records), "name": current_stage, "role": "current",
                "checkpoint": None,
            }
        )
        return (str(source) if source else None, records)
    if not isinstance(stages, list) or any(not isinstance(stage, dict) for stage in stages):
        raise ValueError("transfer.stages must be a list of mappings")
    if len(stages) < 2:
        raise ValueError("transfer.stages must declare at least one source stage and the current stage")
    names = [stage.get("name") for stage in stages]
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("every transfer.stages entry requires a name")
    if len(set(names)) != len(names):
        raise ValueError(f"transfer.stages names must be unique: {names}")
    preceding = stages[:-1]
    checkpoints = [stage.get("checkpoint") for stage in preceding]
    if cli_checkpoint:
        checkpoints[-1] = str(cli_checkpoint)
    source = next(
        (path for path in reversed(checkpoints) if path), None
    )
    records = []
    for index, (stage, path) in enumerate(zip(preceding, checkpoints)):
        record = {
            "index": index, "name": stage["name"], "role": "source",
            "checkpoint": str(path) if path else None,
        }
        if path:
            record.update(describe_checkpoint(path))
        record["used_for_initialization"] = bool(path) and str(path) == str(source)
        records.append(record)
    records.append(
        {
            "index": len(preceding), "name": stages[-1]["name"], "role": "current",
            "checkpoint": None, "requested_stage": current_stage,
        }
    )
    return (str(source) if source else None, records)
