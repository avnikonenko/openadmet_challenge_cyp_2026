from __future__ import annotations

import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from models.encoder_registry import (
    build_encoder, encoder_output_dim, encoder_type, load_model_state_for_tasks,
)
from models.multitask_gnn import CYPDMPNN, batch_graphs
from models.target_conditioned_gnn import CYPTargetConditioned, TargetConditionedHeads
from .data import DataBundle, matrix, save_split_ids
from .constants import FEATURE_SCHEMA_VERSION, MODEL_CODE_VERSION
from .metrics import regression_metrics
from .resource_monitor import ResourceMonitor
from .utils import (
    GracefulStop, RuntimeGuard, RunContext, atomic_torch_save, file_sha256, json_dump,
    normalize_device,
)
from .validation import validate_predictions


@dataclass
class TargetScaling:
    mean: np.ndarray
    std: np.ndarray


def fit_target_scaling(bundle: DataBundle, mode: str = "zscore_per_target") -> TargetScaling:
    values = matrix(bundle.frame.loc[bundle.train_mask], bundle.target_columns)
    observed = np.isfinite(values).any(axis=0)
    if not observed.all():
        missing = [bundle.task_names[index] for index in np.flatnonzero(~observed)]
        raise ValueError(f"Training partition has no observed labels for targets: {missing}")
    if mode == "none":
        size = len(bundle.target_columns)
        return TargetScaling(np.zeros(size, dtype=np.float32), np.ones(size, dtype=np.float32))
    if mode != "zscore_per_target":
        raise ValueError("target_normalization must be none or zscore_per_target")
    means = np.nanmean(values, axis=0)
    stds = np.nanstd(values, axis=0)
    stds[~np.isfinite(stds) | (stds < 1e-8)] = 1.0
    return TargetScaling(means.astype(np.float32), stds.astype(np.float32))


def uncertainty_weights(values: np.ndarray, clip_min: float, clip_max: float) -> np.ndarray:
    result = np.ones_like(values, dtype=np.float32)
    for column in range(values.shape[1]):
        valid = np.isfinite(values[:, column]) & (values[:, column] > 0)
        if not valid.any():
            continue
        raw = 1.0 / np.square(values[valid, column])
        raw /= np.median(raw)
        result[valid, column] = np.clip(raw, clip_min, clip_max)
    return result


class MolecularDataset(Dataset):
    def __init__(
        self, frame: pd.DataFrame, bundle: DataBundle, scaling: TargetScaling,
        loss_config: dict[str, Any],
    ):
        self.frame = frame.reset_index(drop=True)
        self.smiles = self.frame["canonical_smiles"].astype(str).tolist()
        raw = matrix(self.frame, bundle.target_columns)
        self.targets = (raw - scaling.mean) / scaling.std
        self.raw_targets = raw
        self.lower = self._bounds(bundle.lower_columns, scaling)
        self.upper = self._bounds(bundle.upper_columns, scaling)
        uncertainty = self._values(bundle.uncertainty_columns)
        self.uncertainty = uncertainty
        self.weights = uncertainty_weights(
            uncertainty,
            float(loss_config.get("weight_clip_min", 0.2)),
            float(loss_config.get("weight_clip_max", 5.0)),
        )

    def _values(self, columns: tuple[str, ...]) -> np.ndarray:
        result = np.full((len(self.frame), len(columns)), np.nan, dtype=np.float32)
        for index, column in enumerate(columns):
            if column and column in self.frame:
                result[:, index] = self.frame[column].to_numpy(dtype=np.float32)
        return result

    def _bounds(self, columns: tuple[str, ...], scaling: TargetScaling) -> np.ndarray:
        values = self._values(columns)
        return (values - scaling.mean) / scaling.std

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> int:
        return index

    def collate(self, indices: list[int]) -> dict[str, Any]:
        selected = np.asarray(indices, dtype=int)
        return {
            "graph": batch_graphs([self.smiles[index] for index in selected]),
            "targets": torch.from_numpy(self.targets[selected]),
            "lower": torch.from_numpy(self.lower[selected]),
            "upper": torch.from_numpy(self.upper[selected]),
            "weights": torch.from_numpy(self.weights[selected]),
            "indices": selected,
        }


def masked_loss(
    prediction: torch.Tensor, target: torch.Tensor, lower: torch.Tensor,
    upper: torch.Tensor, weights: torch.Tensor, mode: str,
) -> torch.Tensor:
    task_losses = []
    for task in range(target.shape[1]):
        mask = torch.isfinite(target[:, task])
        if not mask.any():
            continue
        pred = prediction[mask, task]
        truth = target[mask, task]
        if mode == "standard":
            losses = torch.square(pred - truth)
        elif mode == "clipped_uncertainty_weighted":
            losses = torch.square(pred - truth) * weights[mask, task]
        elif mode == "interval_aware":
            low = lower[mask, task]
            high = upper[mask, task]
            valid_bounds = torch.isfinite(low) & torch.isfinite(high)
            point_loss = torch.square(pred - truth)
            safe_low = torch.where(valid_bounds, low, pred.detach())
            safe_high = torch.where(valid_bounds, high, pred.detach())
            distance = torch.relu(pred - safe_high) + torch.relu(safe_low - pred)
            losses = torch.where(valid_bounds, torch.square(distance), point_loss)
        else:
            raise ValueError(f"Unknown loss mode: {mode}")
        task_losses.append(losses.mean())
    if not task_losses:
        return prediction.sum() * 0
    return torch.stack(task_losses).mean()


ARCHITECTURES = ("cyp_specific_heads", "target_conditioned", "joint_multiassay")
DIRECT_SCORED_PURPOSES = ("direct_pic50", "joint_multiassay")
FINETUNING_STAGES = ("transfer", "endpoint_refinement")


def build_model(task_names: tuple[str, ...], config: dict[str, Any]):
    """Compose the configured encoder and prediction heads.

    ``model.type`` selects the encoder (``dmpnn`` or ``gine``) and
    ``model.architecture`` selects how targets are predicted from the molecular
    embedding. The default ``dmpnn`` + ``cyp_specific_heads`` combination builds the
    original `CYPDMPNN`, so existing configurations and checkpoints are unchanged.
    """
    model = config["model"]
    architecture = str(model.get("architecture", "cyp_specific_heads"))
    if architecture not in ARCHITECTURES:
        raise ValueError(f"model.architecture must be one of {ARCHITECTURES}, not {architecture!r}")
    resolved_encoder = encoder_type(model)
    dropout = float(model.get("dropout", 0.1))
    head_dropout = float(model.get("ffn_dropout", dropout))
    head_type = str(model.get("head_type", "mlp"))
    head_layer_norm = bool(model.get("head_layer_norm", False))
    if architecture == "cyp_specific_heads":
        if resolved_encoder == "native_dmpnn":
            return CYPDMPNN(
                task_names=task_names,
                message_hidden_dim=int(model.get("message_hidden_dim", 300)),
                message_passing_depth=int(model.get("message_passing_depth", 3)),
                dropout=dropout,
                aggregation=model.get("aggregation", "mean"),
                ffn_hidden_dim=int(model.get("ffn_hidden_dim", 300)),
                ffn_num_layers=int(model.get("ffn_num_layers", 2)),
                ffn_dropout=head_dropout,
                head_type=head_type,
                head_layer_norm=head_layer_norm,
            )
        from models.gine_model import CYPGINE

        return CYPGINE(
            task_names=task_names,
            hidden_dim=int(model.get("hidden_dim", 512)),
            num_layers=int(model.get("num_layers", 5)),
            dropout=dropout,
            residual=bool(model.get("residual", True)),
            normalization=str(model.get("normalization", "layernorm")),
            pooling=str(model.get("pooling", "mean")),
            ffn_hidden_dim=int(model.get("ffn_hidden_dim", 512)),
            ffn_num_layers=int(model.get("ffn_num_layers", 3)),
            ffn_dropout=head_dropout,
            head_type=head_type,
            head_layer_norm=head_layer_norm,
        )
    heads = TargetConditionedHeads(
        task_names=task_names,
        input_dim=encoder_output_dim(model),
        cyp_embedding_dim=int(model.get("cyp_embedding_dim", 32)),
        predictor_hidden_dim=int(model.get("predictor_hidden_dim", 600)),
        predictor_layers=int(model.get("predictor_layers", 3)),
        dropout=head_dropout,
        head_type=head_type,
        head_layer_norm=head_layer_norm,
        assay_conditioned=architecture == "joint_multiassay",
        assay_embedding_dim=model.get("assay_embedding_dim"),
    )
    return CYPTargetConditioned(build_encoder(model), heads, task_names)


def resolve_loss_config(config: dict[str, Any], stage: str | None = None) -> dict[str, Any]:
    """Resolve the loss for one training stage.

    A stage section (``pretraining``/``transfer``/``training``) may set ``loss_mode``
    directly or carry a nested ``loss:`` mapping. Anything it does not set falls back
    to the shared top-level ``loss:`` block, so configurations without stage-specific
    losses behave exactly as before.
    """
    resolved = dict(config.get("loss") or {})
    section = config.get(stage) if stage else None
    if isinstance(section, dict):
        nested = section.get("loss")
        if isinstance(nested, dict):
            resolved.update(nested)
        elif nested is not None:
            raise ValueError(f"{stage}.loss must be a mapping of loss options")
        if "loss_mode" in section:
            resolved["loss_mode"] = section["loss_mode"]
    resolved.setdefault("loss_mode", "standard")
    return resolved


def parameter_counts(model) -> dict[str, int]:
    """Capacity diagnostics used to tell genuine gains from extra parameters."""
    total = int(sum(parameter.numel() for parameter in model.parameters()))
    encoder = int(sum(parameter.numel() for parameter in model.encoder.parameters()))
    heads = int(sum(parameter.numel() for parameter in model.heads.parameters()))
    trainable = int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )
    return {
        "total_parameters": total,
        "encoder_parameters": encoder,
        "head_parameters": heads,
        "other_parameters": total - encoder - heads,
        "trainable_parameters": trainable,
    }


def prediction_frame(
    model, dataset: MolecularDataset, loader: DataLoader,
    scaling: TargetScaling, task_names: tuple[str, ...], device: str,
    fold: int, seed: int, model_name: str,
    scored_indices: tuple[int, ...] | None = None,
    scored_labels: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Out-of-fold predictions for the scored tasks, on the original target scale.

    ``scored_indices``/``scored_labels`` restrict the output to the tasks that are
    actually scored and give each one its reported CYP name; by default every task is
    scored under its own name, which is the behaviour of the single-assay models.
    """
    model.eval()
    predictions = np.full_like(dataset.raw_targets, np.nan)
    with torch.no_grad():
        for batch in loader:
            output = model(batch["graph"].to(device)).cpu().numpy()
            predictions[batch["indices"]] = output * scaling.std + scaling.mean
    scored = tuple(range(len(task_names))) if scored_indices is None else tuple(scored_indices)
    labels = tuple(task_names) if scored_labels is None else tuple(scored_labels)
    rows = []
    for task_index, cyp in zip(scored, labels):
        for row_index, row in dataset.frame.iterrows():
            if not np.isfinite(dataset.raw_targets[row_index, task_index]):
                continue
            rows.append(
                {
                    "molecule_id": row["Molecule_Name"],
                    "canonical_smiles": row["canonical_smiles"],
                    "fold": fold,
                    "CYP": cyp,
                    "y_true": dataset.raw_targets[row_index, task_index],
                    "y_pred": predictions[row_index, task_index],
                    "experimental_uncertainty": dataset.uncertainty[row_index, task_index],
                    "lower": (
                        dataset.lower[row_index, task_index] * scaling.std[task_index]
                        + scaling.mean[task_index]
                    ),
                    "upper": (
                        dataset.upper[row_index, task_index] * scaling.std[task_index]
                        + scaling.mean[task_index]
                    ),
                    "model": model_name,
                    "seed": seed,
                    "prediction_scale": "original",
                }
            )
    return pd.DataFrame(rows)


def train_dmpnn(
    bundle: DataBundle, config: dict[str, Any], run: RunContext, fold: int,
    seed: int, device: str, max_runtime_minutes: float | None,
    encoder_checkpoint: str | Path | None = None, transfer_mode: str = "random_init",
    partial_load: bool = False, runtime_started: float | None = None,
    stage: str | None = None, initial_checkpoint: str | Path | None = None,
    freeze_encoder: bool = False, transfer_stages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    normalized_device = normalize_device(device)
    if normalized_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {normalized_device}")
    torch_device = torch.device(normalized_device)
    training = config.get("training", config.get("pretraining", config.get("transfer", {})))
    if str(training.get("optimizer", "adamw")).lower() != "adamw":
        raise ValueError("The implemented neural optimizer is adamw")
    loss_config = resolve_loss_config(config, stage)
    scored_indices = bundle.scored_indices
    scored_labels = bundle.scored_labels
    selection_metric = training.get(
        "selection_metric",
        "macro_ST_RAE"
        if bundle.split_metadata["purpose"] in DIRECT_SCORED_PURPOSES
        else "macro_MAE",
    )
    monitored_metric = selection_metric
    metric_direction = str(training.get("metric_direction", "minimize"))
    if metric_direction not in {"minimize", "maximize"}:
        raise ValueError("training.metric_direction must be minimize or maximize")
    normalization_mode = config.get("data", {}).get("target_normalization", "zscore_per_target")
    scaling = fit_target_scaling(bundle, normalization_mode)
    train_data = MolecularDataset(bundle.frame.loc[bundle.train_mask], bundle, scaling, loss_config)
    validation_data = MolecularDataset(bundle.frame.loc[bundle.validation_mask], bundle, scaling, loss_config)
    validation_observed = np.isfinite(validation_data.raw_targets).any(axis=0)
    unobserved_scored = [
        bundle.task_names[index] for index in scored_indices if not validation_observed[index]
    ]
    if unobserved_scored:
        raise ValueError(
            f"Validation partition has no observed labels for targets: {unobserved_scored}"
        )
    generator = torch.Generator().manual_seed(seed)
    batch_size = int(training.get("batch_size", 64))
    max_epochs = int(training.get("max_epochs", 100))
    patience = int(training.get("early_stopping_patience", 15))
    if batch_size < 1 or max_epochs < 1 or patience < 1:
        raise ValueError("batch_size, max_epochs, and early_stopping_patience must be positive")
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, generator=generator,
        num_workers=int(training.get("num_workers", 0)), collate_fn=train_data.collate,
    )
    validation_loader = DataLoader(
        validation_data, batch_size=batch_size, shuffle=False,
        num_workers=int(training.get("num_workers", 0)), collate_fn=validation_data.collate,
    )
    model = build_model(bundle.task_names, config)
    preload_metadata = None
    refinement_metadata = None
    if encoder_checkpoint and initial_checkpoint:
        raise ValueError(
            "Provide either an encoder checkpoint or a full refinement checkpoint, not both"
        )
    if encoder_checkpoint:
        from models.encoder_registry import load_pretrained_encoder

        preload_metadata = load_pretrained_encoder(
            model, encoder_checkpoint, config.get("model", {}), partial_load
        )
        model.reset_heads()
    elif initial_checkpoint:
        from models.transfer_model import validate_encoder_checkpoint

        source = torch.load(initial_checkpoint, map_location="cpu", weights_only=False)
        validate_encoder_checkpoint(source, config.get("model", {}), partial_load=False)
        refinement_metadata = {
            **load_model_state_for_tasks(model, source, bundle.task_names, partial_load),
            "checkpoint": str(Path(initial_checkpoint).resolve()),
            "sha256": file_sha256(initial_checkpoint),
            "source_task_names": list(source.get("task_names", ())),
            "source_stage": source.get("stage"),
            "source_loss_mode": source.get("loss_mode"),
        }
    if normalized_device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch_device)
    model.to(torch_device)

    requested_precision = str(training.get("precision", "auto")).lower()
    if normalized_device.startswith("cuda"):
        if requested_precision == "auto":
            precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        else:
            precision = requested_precision
        if precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("training.precision must be auto, fp32, fp16, or bf16")
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 was requested but is unsupported by this GPU")
    else:
        if requested_precision not in {"auto", "fp32"}:
            raise RuntimeError("fp16/bf16 mixed precision is supported only on CUDA")
        precision = "fp32"
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")

    encoder_lr = float(training.get("encoder_learning_rate", training.get("learning_rate", 1e-3)))
    head_lr = float(training.get("head_learning_rate", training.get("learning_rate", 1e-3)))
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": encoder_lr, "name": "encoder"},
            {"params": model.heads.parameters(), "lr": head_lr, "name": "heads"},
        ],
        weight_decay=float(training.get("weight_decay", 1e-5)),
    )
    scheduler_name = str(training.get("scheduler", "none")).lower()
    if scheduler_name == "none":
        scheduler = None
    elif scheduler_name == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(training.get("metric_direction", "minimize")).replace("imize", ""),
            factor=float(training.get("scheduler_factor", 0.5)),
            patience=int(training.get("scheduler_patience", 5)),
        )
    else:
        raise ValueError("training.scheduler must be none or reduce_on_plateau")
    freeze_epochs = int(training.get("freeze_encoder_epochs", 0)) if transfer_mode == "freeze_then_unfreeze" else 0
    encoder_trainable = not freeze_encoder
    for parameter in model.encoder.parameters():
        parameter.requires_grad = encoder_trainable and freeze_epochs == 0

    latest = run.checkpoints / "latest.pt"
    start_epoch = 0
    best_epoch = -1
    best_metric = float("inf") if metric_direction == "minimize" else float("-inf")
    best_metrics: dict[str, float] = {}
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    if latest.exists():
        checkpoint = torch.load(latest, map_location=torch_device, weights_only=False)
        from models.transfer_model import validate_encoder_checkpoint

        validate_encoder_checkpoint(checkpoint, config.get("model", {}), partial_load=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = int(checkpoint["best_epoch"])
        best_metric = float(checkpoint["best_metric"])
        stale_epochs = int(checkpoint.get("stale_epochs", 0))
        best_metrics = dict(checkpoint.get("best_metrics", {}))
        history = list(checkpoint.get("history", []))
        monitored_metric = checkpoint.get("monitored_metric", selection_metric)
        if "data_loader_generator_state" in checkpoint:
            generator.set_state(checkpoint["data_loader_generator_state"].cpu())
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_states"):
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_rng_states"]]
            )
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])
        run.log("resumed", epoch=start_epoch)
    for parameter in model.encoder.parameters():
        parameter.requires_grad = encoder_trainable and start_epoch >= freeze_epochs

    guard = RuntimeGuard(max_runtime_minutes, started=runtime_started or time.monotonic())
    loss_mode = loss_config.get("loss_mode", "standard")
    architecture = str(config["model"].get("architecture", "cyp_specific_heads"))
    initial_capacity = parameter_counts(model)
    json_dump(
        {
            **initial_capacity,
            "initial_trainable_parameters": initial_capacity["trainable_parameters"],
            "architecture": architecture,
            "encoder_type": config["model"].get("type", "dmpnn"),
            "head_type": config["model"].get("head_type", "mlp"),
            "encoder_trainable": any(
                parameter.requires_grad for parameter in model.encoder.parameters()
            ),
            "task_names": list(bundle.task_names),
            "scored_tasks": list(bundle.scored_task_names),
        },
        run.run_dir / "model_capacity.json",
    )
    save_split_ids(bundle, run.run_dir)
    weight_stats = {
        task: {
            "min": float(np.nanmin(train_data.weights[:, index])),
            "median": float(np.nanmedian(train_data.weights[:, index])),
            "max": float(np.nanmax(train_data.weights[:, index])),
        }
        for index, task in enumerate(bundle.task_names)
    }
    json_dump(weight_stats, run.run_dir / "uncertainty_weight_summary.json")
    expected_validation_keys = {
        (row["Molecule_Name"], label)
        for task_index, label in zip(scored_indices, scored_labels)
        for row_index, row in validation_data.frame.iterrows()
        if np.isfinite(validation_data.raw_targets[row_index, task_index])
    }

    runtime_limit_reached = False
    stop_signal = None
    monitor = ResourceMonitor(normalized_device)
    monitor.start()
    estimated_epoch_seconds = (
        max(float(row["runtime_seconds"]) for row in history if row.get("runtime_seconds") is not None)
        if any(row.get("runtime_seconds") is not None for row in history)
        else training.get("initial_epoch_estimate_seconds")
    )
    with GracefulStop() as graceful_stop:
        for epoch in range(start_epoch, max_epochs):
            if graceful_stop.requested or not guard.can_start_epoch(
                estimated_epoch_seconds,
                float(training.get("runtime_safety_margin_seconds", 60)),
            ):
                runtime_limit_reached = True
                stop_signal = graceful_stop.signal_name
                run.log(
                    "training_stopped_before_epoch", epoch=epoch,
                    reason=f"signal:{stop_signal}" if stop_signal else "insufficient_runtime",
                )
                break
            epoch_started = time.monotonic()
            if epoch == freeze_epochs and freeze_epochs > 0:
                for parameter in model.encoder.parameters():
                    parameter.requires_grad = True
                run.log("encoder_unfrozen", epoch=epoch)
            model.train()
            losses = []
            gradient_norms = []
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                graph = batch["graph"].to(torch_device)
                with torch.autocast(
                    device_type="cuda", dtype=amp_dtype,
                    enabled=normalized_device.startswith("cuda") and precision != "fp32",
                ):
                    prediction = model(graph)
                    loss = masked_loss(
                        prediction, batch["targets"].to(torch_device), batch["lower"].to(torch_device),
                        batch["upper"].to(torch_device), batch["weights"].to(torch_device), loss_mode,
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training.get("gradient_clip_norm", 5.0))
                )
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach().cpu()))
                gradient_norms.append(float(gradient_norm.detach().cpu()))

            validation_predictions = prediction_frame(
                model, validation_data, validation_loader, scaling, bundle.task_names,
                normalized_device, fold, seed,
                config.get("model_id", config.get("experiment", "dmpnn")),
                scored_indices, scored_labels,
            )
            integrity = validate_predictions(
                validation_predictions, expected_validation_keys, scored_labels, True
            )
            json_dump(integrity, run.run_dir / "prediction_integrity.json")
            per_cyp, macro = regression_metrics(validation_predictions)
            current = float(macro.get(monitored_metric, np.nan))
            if not np.isfinite(current):
                if monitored_metric != selection_metric:
                    raise ValueError(f"Monitored fallback metric {monitored_metric} is non-finite")
                fallback_metric = "macro_MAE"
                current = float(macro.get(fallback_metric, np.nan))
                if not np.isfinite(current):
                    raise ValueError(
                        f"Monitored metric {selection_metric} and fallback {fallback_metric} are non-finite"
                    )
                if metric_direction != "minimize":
                    raise ValueError(
                        f"Cannot replace non-finite {selection_metric} with MAE while metric_direction="
                        f"{metric_direction}; configure a finite monitored metric or minimize direction"
                    )
                run.log("selection_metric_fallback", configured=selection_metric, used=fallback_metric)
                monitored_metric = fallback_metric
            epoch_seconds = time.monotonic() - epoch_started
            estimated_epoch_seconds = max(epoch_seconds, estimated_epoch_seconds or 0)
            per_cyp_columns = {
                f"ST_RAE_{row['CYP']}": float(row["ST_RAE"])
                for row in per_cyp.to_dict("records")
            }
            learning_rates = {
                f"learning_rate_{group.get('name', index)}": float(group["lr"])
                for index, group in enumerate(optimizer.param_groups)
            }
            epoch_row = {
                "epoch": epoch, "train_loss": float(np.mean(losses)),
                "gradient_norm_mean": float(np.mean(gradient_norms)) if gradient_norms else np.nan,
                "gradient_norm_max": float(np.max(gradient_norms)) if gradient_norms else np.nan,
                "runtime_seconds": epoch_seconds, "monitored_metric": monitored_metric,
                "monitored_value": current, **macro, **per_cyp_columns, **learning_rates,
                "encoder_trainable": bool(
                    next(model.encoder.parameters()).requires_grad
                ),
            }
            history.append(epoch_row)
            pd.DataFrame(history).to_csv(run.run_dir / "training_history.csv", index=False)
            improved = current < best_metric if metric_direction == "minimize" else current > best_metric
            if improved:
                best_metric, best_epoch, stale_epochs = current, epoch, 0
                best_metrics = {key: float(value) for key, value in macro.items()}
            else:
                stale_epochs += 1
            if scheduler is not None:
                scheduler.step(current)
            state = {
                "epoch": epoch, "best_epoch": best_epoch, "best_metric": best_metric,
                "stale_epochs": stale_epochs, "model_state_dict": model.state_dict(),
                "encoder_state_dict": model.encoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "history": history,
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "scaler_state_dict": scaler.state_dict(),
                "task_names": bundle.task_names, "target_mean": scaling.mean,
                "target_std": scaling.std, "model_config": config["model"],
                "transfer_mode": transfer_mode, "best_metrics": best_metrics,
                "target_normalization": {
                    "mode": normalization_mode, "mean": scaling.mean, "std": scaling.std,
                },
                "architecture_type": config["model"].get("type", "dmpnn"),
                "architecture": architecture,
                "stage": stage,
                "loss_mode": loss_mode,
                "loss_config": dict(loss_config),
                "transfer_stages": transfer_stages,
                "parameter_counts": parameter_counts(model),
                "scored_tasks": list(bundle.scored_task_names),
                "scored_labels": list(scored_labels),
                "model_code_version": MODEL_CODE_VERSION,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "pytorch_version": torch.__version__,
                "chemprop_version": "self-contained-chemprop-style",
                "monitored_metric": monitored_metric,
                "metric_direction": metric_direction,
                "data_loader_generator_state": generator.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "split_metadata": bundle.split_metadata,
            }
            if improved:
                atomic_torch_save(state, run.checkpoints / "best.pt")
                atomic_torch_save(
                    {
                        "encoder_state_dict": model.encoder.state_dict(),
                        "model_config": config["model"],
                        "source_epoch": epoch,
                        "task_names": bundle.task_names,
                        "target_normalization": state["target_normalization"],
                        "architecture_type": state["architecture_type"],
                        "architecture": architecture,
                        "stage": stage,
                        "loss_mode": loss_mode,
                        "loss_config": dict(loss_config),
                        "transfer_stages": transfer_stages,
                        "model_code_version": MODEL_CODE_VERSION,
                        "feature_schema_version": FEATURE_SCHEMA_VERSION,
                        "pytorch_version": torch.__version__,
                        "chemprop_version": "self-contained-chemprop-style",
                        "split_metadata": bundle.split_metadata,
                    },
                    run.checkpoints / "best_encoder.pt",
                )
                validation_predictions.to_csv(run.run_dir / "predictions.csv", index=False)
                per_cyp.to_csv(run.run_dir / "metrics_per_cyp.csv", index=False)
                json_dump(macro, run.run_dir / "metrics.json")
            atomic_torch_save(state, latest)
            run.log("epoch", **epoch_row, improved=improved)
            runtime_limit_reached = guard.should_stop() or graceful_stop.requested
            if stale_epochs >= patience or runtime_limit_reached:
                stop_signal = graceful_stop.signal_name
                reason = "patience" if stale_epochs >= patience else (
                    f"signal:{stop_signal}" if stop_signal else "runtime"
                )
                run.log("training_stopped", reason=reason)
                break

    if runtime_limit_reached and not latest.exists():
        atomic_torch_save(
            {
                "epoch": -1, "best_epoch": best_epoch, "best_metric": best_metric,
                "stale_epochs": stale_epochs, "model_state_dict": model.state_dict(),
                "encoder_state_dict": model.encoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "history": history,
                "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                "scaler_state_dict": scaler.state_dict(), "task_names": bundle.task_names,
                "target_mean": scaling.mean, "target_std": scaling.std,
                "model_config": config["model"], "transfer_mode": transfer_mode,
                "best_metrics": best_metrics,
                "target_normalization": {
                    "mode": normalization_mode, "mean": scaling.mean, "std": scaling.std,
                },
                "architecture_type": config["model"].get("type", "dmpnn"),
                "architecture": architecture, "stage": stage, "loss_mode": loss_mode,
                "loss_config": dict(loss_config), "transfer_stages": transfer_stages,
                "parameter_counts": parameter_counts(model),
                "scored_tasks": list(bundle.scored_task_names),
                "scored_labels": list(scored_labels),
                "model_code_version": MODEL_CODE_VERSION,
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "pytorch_version": torch.__version__,
                "chemprop_version": "self-contained-chemprop-style",
                "monitored_metric": monitored_metric, "metric_direction": metric_direction,
                "data_loader_generator_state": generator.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state(),
                "split_metadata": bundle.split_metadata,
            },
            latest,
        )

    resource_summary = monitor.finish()
    final_capacity = parameter_counts(model)
    capacity_report = {
        **initial_capacity,
        "initial_trainable_parameters": initial_capacity["trainable_parameters"],
        "final_trainable_parameters": final_capacity["trainable_parameters"],
        "trainable_parameters": final_capacity["trainable_parameters"],
        "architecture": architecture,
        "encoder_type": config["model"].get("type", "dmpnn"),
        "head_type": config["model"].get("head_type", "mlp"),
        "encoder_trainable": any(
            parameter.requires_grad for parameter in model.encoder.parameters()
        ),
        "task_names": list(bundle.task_names),
        "scored_tasks": list(bundle.scored_task_names),
    }
    json_dump(capacity_report, run.run_dir / "model_capacity.json")

    preloaded = preload_metadata or {}
    return {
        "best_epoch": best_epoch,
        "best_validation_metric": best_metric,
        "selection_metric": selection_metric,
        "monitored_metric": monitored_metric,
        "metric_direction": metric_direction,
        "target_normalization": normalization_mode,
        "precision_mode": precision,
        "final_metrics": best_metrics,
        "transfer_mode": transfer_mode,
        "pretrained_encoder": preload_metadata,
        "endpoint_refinement_source": refinement_metadata,
        "transfer_stages": transfer_stages,
        "training_stage": stage,
        "architecture": architecture,
        "encoder_type": config["model"].get("type", "dmpnn"),
        "head_type": config["model"].get("head_type", "mlp"),
        "loss_config": dict(loss_config),
        "loss_mode": loss_mode,
        # The loss actually optimized at each stage: this run's own loss, plus the loss
        # recorded in the checkpoint this run's encoder came from. A from-scratch run
        # has neither stage, and records only loss_mode.
        "finetuning_loss_mode": (
            loss_mode if stage in FINETUNING_STAGES else None
        ),
        "pretraining_loss_mode": (
            loss_mode
            if stage == "pretraining"
            else preloaded.get("source_loss_mode")
            or (refinement_metadata or {}).get("source_loss_mode")
        ),
        "encoder_trainable": capacity_report["encoder_trainable"],
        "scored_tasks": list(bundle.scored_task_names),
        **final_capacity,
        "epochs_completed": len(history),
        "runtime_limit_reached": runtime_limit_reached,
        "stop_signal": stop_signal,
        **resource_summary,
    }
