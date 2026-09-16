#!/usr/bin/env python3
"""Refine one CYP endpoint from a completed multitask run.

The shared encoder and that endpoint's head are loaded from a completed multitask
(or transfer) run and trained further on a single CYP, with the encoder either frozen
or fully trainable. This is the second stage after multitask transfer: CYP2D6 is the
intended first target because it is the least correlated endpoint with the others.

The refinement is an ordinary run with its own identity, fold, seed, leakage check,
and checkpoints, so it is evaluated, predicted, and ensembled by the existing tools.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from src.constants import CYPS
from src.data import load_direct_data, smoke_subset
from src.run_management import begin_run, finish_run
from src.training import train_dmpnn
from src.utils import add_common_args, load_config, seed_everything
from src.validation import audit_featurization, leakage_checks


ENCODER_MODES = ("frozen", "full")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fine-tune one CYP endpoint from a completed multitask checkpoint"
    )
    add_common_args(parser, "configs/transfer_singleconc.yaml")
    # Default to the source run's own resolved configuration rather than a repo config.
    parser.set_defaults(config=None)
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed multitask run")
    parser.add_argument("--cyp", required=True, choices=CYPS)
    parser.add_argument("--encoder-mode", choices=ENCODER_MODES, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--partial-load", action="store_true")
    args = parser.parse_args()

    source_run = args.run_dir.resolve()
    source_metadata_path = source_run / "metadata.json"
    checkpoint = source_run / "checkpoints" / "best.pt"
    if not source_metadata_path.exists():
        raise FileNotFoundError(f"Source run metadata is missing: {source_metadata_path}")
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    if source_metadata.get("status") != "completed" or not (source_run / "COMPLETED").exists():
        raise ValueError(
            f"Refusing to refine from run status={source_metadata.get('status')!r}; "
            "a completed multitask run is required"
        )
    if not checkpoint.exists():
        raise FileNotFoundError(f"Source checkpoint is missing: {checkpoint}")
    if int(source_metadata.get("fold", -1)) != args.fold:
        raise ValueError(
            f"Source run is fold {source_metadata.get('fold')} but --fold is {args.fold}; "
            "endpoint refinement must stay inside the same outer fold"
        )

    # The source run's own resolved configuration is the default, so the refinement
    # inherits the exact architecture it is initialized from.
    config = load_config(args.config or source_run / "config.yaml", args.overrides)
    refinement = dict((config.get("transfer") or {}).get("endpoint_refinement") or {})
    encoder_mode = args.encoder_mode or str(refinement.get("encoder_mode", "frozen"))
    if encoder_mode not in ENCODER_MODES:
        raise ValueError(f"transfer.endpoint_refinement.encoder_mode must be one of {ENCODER_MODES}")
    epochs = int(args.epochs or refinement.get("epochs", 20))
    learning_rate = float(args.learning_rate or refinement.get("learning_rate", 1e-4))

    base_training = dict(
        config.get("transfer") or config.get("training") or config.get("pretraining") or {}
    )
    base_training.pop("stages", None)
    base_training.pop("endpoint_refinement", None)
    base_training.update(
        {
            "max_epochs": epochs,
            "learning_rate": learning_rate,
            "encoder_learning_rate": learning_rate,
            "head_learning_rate": learning_rate,
            "early_stopping_patience": int(
                refinement.get("early_stopping_patience", min(epochs, base_training.get("early_stopping_patience", 15)))
            ),
        }
    )
    config["training"] = base_training
    config["experiment"] = args.experiment or (
        f"{source_metadata.get('experiment', 'multitask')}_refine_{args.cyp}_{encoder_mode}"
    )
    config.setdefault("transfer", {})["endpoint_refinement"] = {
        **refinement,
        "enabled": True,
        "cyp": args.cyp,
        "encoder_mode": encoder_mode,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "source_run": str(source_run),
    }
    if args.smoke_test:
        args.run_name = args.run_name or "smoke"
        config["training"]["max_epochs"] = min(2, int(config["training"]["max_epochs"]))
        config["training"]["early_stopping_patience"] = 2

    seed_everything(args.seed, bool(config.get("reproducibility", {}).get("deterministic", True)))
    run, metadata, started = begin_run(args, config)
    if metadata.get("already_completed"):
        print(f"Run already completed: {run.run_dir}")
        return 0
    scheme = config.get("data", {}).get("split_scheme", "ecfp_cluster")
    bundle = load_direct_data(scheme, args.fold, cyps=[args.cyp])
    if config.get("model", {}).get("architecture") == "joint_multiassay":
        # Joint checkpoints condition on both CYP and assay type.  The direct-data
        # loader intentionally exposes plain CYP names for ordinary models, so qualify
        # this one refinement task to match the checkpoint's conditioning vocabulary.
        direct_task = f"{args.cyp}|direct_pic50"
        bundle.task_names = (direct_task,)
        bundle.scored_tasks = (direct_task,)
        bundle.task_labels = (args.cyp,)
    if args.smoke_test:
        bundle = smoke_subset(bundle)
    leakage_checks(bundle, run.run_dir)
    audit_featurization(bundle, run.run_dir, config.get("featurization", {}).get("policy", "fail"))
    result = train_dmpnn(
        bundle=bundle, config=config, run=run, fold=args.fold, seed=args.seed,
        device=args.device, max_runtime_minutes=args.max_runtime_minutes,
        transfer_mode=f"endpoint_refinement_{encoder_mode}",
        partial_load=args.partial_load, runtime_started=started,
        stage="endpoint_refinement", initial_checkpoint=checkpoint,
        freeze_encoder=encoder_mode == "frozen",
        transfer_stages=[
            {
                "index": 0, "name": "multitask_source", "role": "source",
                "checkpoint": str(checkpoint),
                "source_experiment": source_metadata.get("experiment"),
            },
            {
                "index": 1, "name": f"endpoint_refinement_{args.cyp}", "role": "current",
                "checkpoint": None, "encoder_mode": encoder_mode,
            },
        ],
    )
    result["split_metadata"] = bundle.split_metadata
    result["refined_endpoint"] = args.cyp
    runtime_exit = bool(result.get("runtime_limit_reached"))
    finish_run(run, metadata, started, result, completed=not runtime_exit)
    print(f"Run {'checkpointed' if runtime_exit else 'completed'}: {run.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
