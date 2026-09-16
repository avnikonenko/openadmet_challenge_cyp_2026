from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from models.transfer_model import resolve_transfer_stages
from src.data import (
    load_direct_data, load_joint_multiassay_data, load_single_concentration_data, smoke_subset,
)
from src.run_management import begin_run, finish_run
from src.training import train_dmpnn
from src.utils import add_common_args, load_config, seed_everything
from src.validation import audit_featurization, leakage_checks


STAGE_NAMES = {
    "single_concentration": "pretraining",
    "direct": "transfer",
    "joint_multiassay": "training",
}
CURRENT_STAGE_LABELS = {
    "single_concentration": "single_concentration",
    "direct": "direct_pic50",
    "joint_multiassay": "joint_multiassay",
}


def neural_parser(description: str, default_config: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    add_common_args(parser, default_config)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--cyp", action="append", default=[])
    parser.add_argument("--partial-load", action="store_true")
    parser.add_argument(
        "--transfer-mode",
        choices=("random_init", "full", "freeze_then_unfreeze"),
        default=None,
    )
    return parser


def run_neural(args: argparse.Namespace, purpose: str) -> int:
    config = load_config(args.config, args.overrides)
    task_set = str(config.get("data", {}).get("task_set", "direct_pic50"))
    if task_set not in {"direct_pic50", "joint_multiassay"}:
        raise ValueError("data.task_set must be direct_pic50 or joint_multiassay")
    if purpose == "direct" and task_set == "joint_multiassay":
        purpose = "joint_multiassay"
    checkpoint, transfer_stages = resolve_transfer_stages(
        config, args.checkpoint, CURRENT_STAGE_LABELS.get(purpose, purpose)
    )
    configured_transfer_mode = config.get("transfer", {}).get("mode", "random_init")
    transfer_mode = args.transfer_mode or configured_transfer_mode
    if purpose == "single_concentration":
        transfer_mode = "random_init"
    elif checkpoint is None:
        if args.transfer_mode and args.transfer_mode != "random_init":
            raise ValueError(
                f"--transfer-mode {args.transfer_mode} requires an encoder checkpoint"
            )
        transfer_mode = "random_init"
    stage = STAGE_NAMES.get(purpose, "training")
    if purpose == "single_concentration":
        config["training"] = dict(config.get("pretraining", config.get("training", {})))
        config["experiment"] = args.experiment or config.get("pretraining_experiment", config.get("experiment", "singleconc_pretrain"))
    elif checkpoint:
        config["training"] = dict(config.get("transfer", config.get("training", {})))
        prefix = config.get("finetune_experiment", "transfer_pic50")
        config["experiment"] = args.experiment or f"{prefix}_{transfer_mode}"
    else:
        # Transfer-oriented configs do not necessarily declare a separate `training`
        # section.  A no-checkpoint invocation is the random-initialization control and
        # should use the same direct-pIC50 optimization settings as fine-tuning.
        config["training"] = dict(config.get("training") or config.get("transfer") or {})
        if not config["training"]:
            raise ValueError("Direct neural training requires a nonempty training or transfer section")
        if args.experiment:
            config["experiment"] = args.experiment
    if not checkpoint and purpose != "single_concentration":
        stage = "training"
    if args.smoke_test:
        args.run_name = args.run_name or "smoke"
        config["training"]["max_epochs"] = min(2, int(config["training"].get("max_epochs", 2)))
        config["training"]["early_stopping_patience"] = 2
    seed_everything(args.seed, bool(config.get("reproducibility", {}).get("deterministic", True)))
    run, metadata, started = begin_run(args, config)
    if metadata.get("already_completed"):
        print(f"Run already completed: {run.run_dir}")
        return 0
    scheme = config.get("data", {}).get("split_scheme", "ecfp_cluster")
    configured = config.get("data", {}).get("cyps")
    cyps = args.cyp or configured
    if purpose == "single_concentration":
        bundle = load_single_concentration_data(
            scheme, args.fold, config.get("data", {}).get("inner_validation_fold")
        )
    elif purpose == "joint_multiassay":
        bundle = load_joint_multiassay_data(scheme, args.fold, cyps=cyps)
    else:
        bundle = load_direct_data(scheme, args.fold, cyps=cyps)
    if args.smoke_test:
        bundle = smoke_subset(bundle)
    audit_auxiliary_checkpoint = (
        checkpoint
        if purpose != "single_concentration"
        and config.get("transfer", {}).get("require_fold_safe_auxiliary", False)
        else None
    )
    leakage_checks(bundle, run.run_dir, audit_auxiliary_checkpoint)
    audit_featurization(
        bundle, run.run_dir, config.get("featurization", {}).get("policy", "fail")
    )
    result = train_dmpnn(
        bundle=bundle, config=config, run=run, fold=args.fold, seed=args.seed,
        device=args.device, max_runtime_minutes=args.max_runtime_minutes,
        encoder_checkpoint=checkpoint, transfer_mode=transfer_mode,
        partial_load=args.partial_load,
        runtime_started=started,
        stage=stage, transfer_stages=transfer_stages,
    )
    result["split_metadata"] = bundle.split_metadata
    runtime_exit = bool(result.get("runtime_limit_reached"))
    finish_run(run, metadata, started, result, completed=not runtime_exit)
    print(f"Run {'checkpointed' if runtime_exit else 'completed'}: {run.run_dir}")
    return 0
