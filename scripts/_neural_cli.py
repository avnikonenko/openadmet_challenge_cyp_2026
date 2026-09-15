from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from src.data import load_direct_data, load_single_concentration_data, smoke_subset
from src.run_management import begin_run, finish_run
from src.training import train_dmpnn
from src.utils import add_common_args, load_config, seed_everything
from src.validation import audit_featurization, leakage_checks


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
    checkpoint = args.checkpoint or config.get("pretrained", {}).get("checkpoint")
    transfer_mode = args.transfer_mode or config.get("transfer", {}).get("mode", "random_init")
    if purpose == "single_concentration":
        config["training"] = dict(config.get("pretraining", config.get("training", {})))
        config["experiment"] = args.experiment or config.get("pretraining_experiment", config.get("experiment", "singleconc_pretrain"))
    elif checkpoint:
        config["training"] = dict(config.get("transfer", config.get("training", {})))
        prefix = config.get("finetune_experiment", "transfer_pic50")
        config["experiment"] = args.experiment or f"{prefix}_{transfer_mode}"
    elif args.experiment:
        config["experiment"] = args.experiment
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
    if purpose == "single_concentration":
        bundle = load_single_concentration_data(
            scheme, args.fold, config.get("data", {}).get("inner_validation_fold")
        )
    else:
        configured = config.get("data", {}).get("cyps")
        cyps = args.cyp or configured
        bundle = load_direct_data(scheme, args.fold, cyps=cyps)
    if args.smoke_test:
        bundle = smoke_subset(bundle)
    audit_auxiliary_checkpoint = (
        checkpoint
        if purpose == "direct"
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
    )
    result["split_metadata"] = bundle.split_metadata
    runtime_exit = bool(result.get("runtime_limit_reached"))
    finish_run(run, metadata, started, result, completed=not runtime_exit)
    print(f"Run {'checkpointed' if runtime_exit else 'completed'}: {run.run_dir}")
    return 0
