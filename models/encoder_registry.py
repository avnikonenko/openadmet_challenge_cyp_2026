"""Encoder construction and pretrained-encoder loading.

`build_encoder` is the single place that turns a `model:` configuration block into a
molecular encoder, so new architectures become selectable through YAML instead of new
training scripts. `load_pretrained_encoder` dispatches on the *source format* of a
checkpoint rather than assuming every checkpoint is a native OpenADMET one.

Extension point for external/foundation encoders
------------------------------------------------
Only ``native`` is registered today. A third-party source (official Chemprop, a public
foundation encoder) must be added to `ENCODER_SOURCE_LOADERS` as an explicit adapter
that either calls that project's own API or applies a verified key/shape mapping, and
must return the same diagnostics dictionary. `partial_load` relaxes completeness checks
for native checkpoints only; it never translates key layouts and must not be used to
silence an incompatible third-party checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from torch import nn

from .multitask_gnn import DMPNNEncoder
from .transfer_model import load_encoder_weights


ENCODER_TYPE_ALIASES = {
    "dmpnn": "native_dmpnn",
    "native_dmpnn": "native_dmpnn",
    "gine": "gine",
}
SUPPORTED_ENCODERS = ("native_dmpnn", "gine")


def encoder_type(model_config: dict[str, Any]) -> str:
    """Resolve the canonical encoder name for a `model:` configuration block."""
    raw = str(model_config.get("type", "dmpnn")).lower()
    resolved = ENCODER_TYPE_ALIASES.get(raw)
    if resolved is None:
        raise ValueError(
            f"Unsupported neural architecture type: {model_config.get('type')!r}; "
            f"supported encoders are {SUPPORTED_ENCODERS}"
        )
    return resolved


def build_encoder(model_config: dict[str, Any]) -> nn.Module:
    """Construct the molecular encoder described by a `model:` configuration block."""
    resolved = encoder_type(model_config)
    if resolved == "native_dmpnn":
        return DMPNNEncoder(
            int(model_config.get("message_hidden_dim", 300)),
            int(model_config.get("message_passing_depth", 3)),
            float(model_config.get("dropout", 0.1)),
            model_config.get("aggregation", "mean"),
        )
    from .gine_model import GINEEncoder

    return GINEEncoder(
        hidden_dim=int(model_config.get("hidden_dim", 512)),
        num_layers=int(model_config.get("num_layers", 5)),
        dropout=float(model_config.get("dropout", 0.1)),
        residual=bool(model_config.get("residual", True)),
        normalization=str(model_config.get("normalization", "layernorm")),
        pooling=str(model_config.get("pooling", "mean")),
    )


def encoder_output_dim(model_config: dict[str, Any]) -> int:
    """Width of the molecular embedding produced by the configured encoder."""
    resolved = encoder_type(model_config)
    if resolved == "native_dmpnn":
        return int(model_config.get("message_hidden_dim", 300))
    return int(model_config.get("hidden_dim", 512))


def _load_native_encoder(
    model: nn.Module, checkpoint_path: str | Path,
    expected_model_config: dict[str, Any] | None, partial_load: bool,
) -> dict[str, Any]:
    return load_encoder_weights(model, checkpoint_path, expected_model_config, partial_load)


ENCODER_SOURCE_LOADERS: dict[str, Callable[..., dict[str, Any]]] = {
    "native": _load_native_encoder,
}


def load_pretrained_encoder(
    model: nn.Module, checkpoint_path: str | Path,
    expected_model_config: dict[str, Any] | None = None,
    partial_load: bool = False, source: str = "native",
) -> dict[str, Any]:
    """Load a pretrained encoder into `model.encoder` using the source's adapter."""
    loader = ENCODER_SOURCE_LOADERS.get(source)
    if loader is None:
        raise ValueError(
            f"No encoder adapter registered for source {source!r}; "
            f"available sources are {sorted(ENCODER_SOURCE_LOADERS)}. Add an explicit, "
            "tested adapter rather than forcing foreign weights through the native loader."
        )
    diagnostics = loader(model, checkpoint_path, expected_model_config, partial_load)
    return {**diagnostics, "encoder_source": source}


def load_model_state_for_tasks(
    model: nn.Module, checkpoint: dict[str, Any], task_names: tuple[str, ...],
    partial_load: bool = False,
) -> dict[str, Any]:
    """Initialize an endpoint-refinement model from a completed multitask checkpoint.

    The encoder must load completely. Head weights are loaded for the requested tasks
    only: per-CYP heads for the other endpoints are absent from the destination model
    and are reported as skipped, while target-conditioned predictors load whole because
    their embedding tables are sized from fixed vocabularies.
    """
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint does not contain a model_state_dict")
    destination = model.state_dict()
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
    missing_encoder_keys = [key for key in missing_destination_keys if key.startswith("encoder.")]
    encoder_shape_mismatches = [key for key in skipped_shape_keys if key.startswith("encoder.")]
    head_shape_mismatches = [key for key in skipped_shape_keys if not key.startswith("encoder.")]
    if missing_encoder_keys or encoder_shape_mismatches:
        raise ValueError(
            f"Refinement checkpoint is incompatible: missing_encoder_keys={missing_encoder_keys}, "
            f"encoder_shape_mismatches={encoder_shape_mismatches}"
        )
    missing_head_keys = [key for key in missing_destination_keys if not key.startswith("encoder.")]
    if (missing_head_keys or head_shape_mismatches) and not partial_load:
        raise ValueError(
            f"Refinement checkpoint lacks head weights for the requested tasks {list(task_names)}: "
            f"missing={missing_head_keys}, shape_mismatches={head_shape_mismatches}; "
            "use --partial-load only when reinitializing them is intended"
        )
    model.load_state_dict(compatible, strict=False)
    return {
        "loaded_key_count": len(compatible),
        "loaded_head_keys": sorted(key for key in compatible if not key.startswith("encoder.")),
        "missing_head_keys": missing_head_keys,
        "skipped_head_shape_keys": head_shape_mismatches,
        "skipped_source_keys": source_only_keys,
        "task_names": list(task_names),
        "partial_load": partial_load,
    }
