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

from models.multitask_gnn import CYPDMPNN, batch_graphs
from .data import DataBundle, matrix, save_split_ids
from .constants import FEATURE_SCHEMA_VERSION, MODEL_CODE_VERSION
from .metrics import regression_metrics
from .resource_monitor import ResourceMonitor
from .utils import (
    GracefulStop, RuntimeGuard, RunContext, atomic_torch_save, json_dump, normalize_device,
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


def build_model(task_names: tuple[str, ...], config: dict[str, Any]) -> CYPDMPNN:
    model = config["model"]
    if model.get("type", "dmpnn") != "dmpnn":
        raise ValueError(f"Unsupported neural architecture type: {model.get('type')}")
    return CYPDMPNN(
        task_names=task_names,
        message_hidden_dim=int(model.get("message_hidden_dim", 300)),
        message_passing_depth=int(model.get("message_passing_depth", 3)),
        dropout=float(model.get("dropout", 0.1)),
        aggregation=model.get("aggregation", "mean"),
        ffn_hidden_dim=int(model.get("ffn_hidden_dim", 300)),
        ffn_num_layers=int(model.get("ffn_num_layers", 2)),
        ffn_dropout=float(model.get("ffn_dropout", model.get("dropout", 0.1))),
    )


def prediction_frame(
    model: CYPDMPNN, dataset: MolecularDataset, loader: DataLoader,
    scaling: TargetScaling, task_names: tuple[str, ...], device: str,
    fold: int, seed: int, model_name: str,
) -> pd.DataFrame:
    model.eval()
    predictions = np.full_like(dataset.raw_targets, np.nan)
    with torch.no_grad():
        for batch in loader:
            output = model(batch["graph"].to(device)).cpu().numpy()
            predictions[batch["indices"]] = output * scaling.std + scaling.mean
    rows = []
    for task_index, cyp in enumerate(task_names):
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
) -> dict[str, Any]:
    normalized_device = normalize_device(device)
    if normalized_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {normalized_device}")
    torch_device = torch.device(normalized_device)
    training = config.get("training", config.get("pretraining", config.get("transfer", {})))
    if str(training.get("optimizer", "adamw")).lower() != "adamw":
        raise ValueError("The implemented neural optimizer is adamw")
    loss_config = config.get("loss", {})
    selection_metric = training.get(
        "selection_metric",
        "macro_ST_RAE" if bundle.split_metadata["purpose"] == "direct_pic50" else "macro_MAE",
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
    if not validation_observed.all():
        missing = [bundle.task_names[index] for index in np.flatnonzero(~validation_observed)]
        raise ValueError(f"Validation partition has no observed labels for targets: {missing}")
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
    if encoder_checkpoint:
        from models.transfer_model import load_encoder_weights

        preload_metadata = load_encoder_weights(
            model, encoder_checkpoint, config.get("model", {}), partial_load
        )
        model.reset_heads()
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
    for parameter in model.encoder.parameters():
        parameter.requires_grad = freeze_epochs == 0

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
        parameter.requires_grad = start_epoch >= freeze_epochs

    guard = RuntimeGuard(max_runtime_minutes, started=runtime_started or time.monotonic())
    loss_mode = loss_config.get("loss_mode", "standard")
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
        (row["Molecule_Name"], task)
        for task_index, task in enumerate(bundle.task_names)
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("gradient_clip_norm", 5.0)))
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach().cpu()))
    
            validation_predictions = prediction_frame(
                model, validation_data, validation_loader, scaling, bundle.task_names,
                normalized_device, fold, seed, config.get("experiment", "dmpnn"),
            )
            integrity = validate_predictions(
                validation_predictions, expected_validation_keys, bundle.task_names, True
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
            epoch_row = {
                "epoch": epoch, "train_loss": float(np.mean(losses)),
                "runtime_seconds": epoch_seconds, "monitored_metric": monitored_metric,
                "monitored_value": current, **macro,
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
        "epochs_completed": len(history),
        "runtime_limit_reached": runtime_limit_reached,
        "stop_signal": stop_signal,
        **resource_summary,
    }
