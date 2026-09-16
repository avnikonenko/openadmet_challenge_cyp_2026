#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Queue independent fold/seed jobs across GPUs or CPUs")
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--folds", nargs="+", type=int, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--devices", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--grid", action="store_true", help="Expand the config search_grid")
    parser.add_argument(
        "--experiment-matrix", action="store_true",
        help="Run only the explicitly named configurations in the config's experiments list",
    )
    parser.add_argument(
        "--experiment-matrix-section", default="experiments",
        help="YAML list used by --experiment-matrix (default: experiments)",
    )
    parser.add_argument("--max-runtime-minutes", type=float, default=50)
    parser.add_argument(
        "--checkpoint-template",
        help="Optional fold/seed-specific path, e.g. outputs/pretrain/fold{fold}/seed{seed}/checkpoints/best.pt",
    )
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.extra[:1] == ["--"]:
        args.extra = args.extra[1:]
    gpu_devices = [device for device in args.devices if device.lower() != "cpu"]
    if len(set(gpu_devices)) != len(gpu_devices):
        parser.error("--devices contains duplicate GPU IDs; each GPU worker slot must be unique")
    if len(set(args.folds)) != len(args.folds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("--folds and --seeds must not contain duplicates")
    if args.grid and args.experiment_matrix:
        parser.error("--grid and --experiment-matrix are mutually exclusive")
    # Each variant is one (run_name, checkpoint_run_name, dotted overrides) job shape,
    # crossed with folds/seeds.  Distinct fine-tuning variants may intentionally reuse
    # one architecture-matched pretraining run.
    variants: list[tuple[str | None, str | None, list[str]]] = [(None, None, [])]
    if args.grid:
        import yaml

        with args.config.open(encoding="utf-8") as handle:
            search_grid = (yaml.safe_load(handle) or {}).get("search_grid", {})
        if not search_grid or any(not isinstance(values, list) or not values for values in search_grid.values()):
            parser.error("--grid requires a nonempty search_grid whose values are nonempty lists")
        if "--run-name" in args.extra:
            parser.error("Do not combine --grid with an explicit --run-name")
        grid_keys = list(search_grid)
        variants = [
            (
                f"grid{index:03d}",
                f"grid{index:03d}",
                [f"{key}={json.dumps(value)}" for key, value in zip(grid_keys, combination)],
            )
            for index, combination in enumerate(
                itertools.product(*(search_grid[key] for key in grid_keys))
            )
        ]
    elif args.experiment_matrix:
        import yaml

        with args.config.open(encoding="utf-8") as handle:
            experiments = (yaml.safe_load(handle) or {}).get(args.experiment_matrix_section)
        if not isinstance(experiments, list) or not experiments:
            parser.error(
                "--experiment-matrix requires a nonempty "
                f"{args.experiment_matrix_section!r} list in the config"
            )
        if "--run-name" in args.extra:
            parser.error("Do not combine --experiment-matrix with an explicit --run-name")
        variants = []
        for entry in experiments:
            if not isinstance(entry, dict) or not entry.get("name"):
                parser.error("each experiments entry must be a mapping with a name")
            overrides = entry.get("overrides") or {}
            if not isinstance(overrides, dict):
                parser.error(f"experiments entry {entry['name']!r} has non-mapping overrides")
            variants.append(
                (
                    str(entry["name"]),
                    str(entry.get("checkpoint_run_name", entry["name"])),
                    [f"{key}={json.dumps(value)}" for key, value in overrides.items()],
                )
            )
        names = [name for name, _checkpoint_name, _overrides in variants]
        if len(set(names)) != len(names):
            parser.error(f"experiments names must be unique: {names}")
    launcher = args.output_dir / "launcher_logs" / f"{args.script.stem}_{time.time_ns()}"
    launcher.mkdir(parents=True, exist_ok=True)
    queue = deque(
        (fold, seed, variant_index, run_name, checkpoint_run_name, overrides)
        for fold in args.folds
        for seed in args.seeds
        for variant_index, (run_name, checkpoint_run_name, overrides) in enumerate(variants)
    )
    available = deque(args.devices)
    running = {}
    results = []
    stop_requested = False

    def request_stop(signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True
        queue.clear()
        for process, *_rest in list(running.values()):
            if process.poll() is None:
                process.send_signal(signum)

    previous_handlers = {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, request_stop)
    while queue or running:
        while queue and available and not stop_requested:
            fold, seed, variant_index, run_name, checkpoint_run_name, variant_overrides = queue.popleft()
            device_id = available.popleft()
            device = "cpu" if device_id.lower() == "cpu" else f"cuda:{device_id}"
            command = [
                sys.executable, str(args.script), "--config", str(args.config),
                "--fold", str(fold), "--seed", str(seed), "--device", device,
                "--output-dir", str(args.output_dir),
                "--max-runtime-minutes", str(args.max_runtime_minutes),
            ]
            if args.resume:
                command.append("--resume")
            if args.checkpoint_template:
                checkpoint = args.checkpoint_template.format(
                    fold=fold, seed=seed, grid=variant_index, run_name=run_name or "",
                    checkpoint_run_name=checkpoint_run_name or run_name or "",
                )
                command.extend(["--checkpoint", checkpoint])
            if run_name:
                command.extend(["--run-name", run_name])
            for override in variant_overrides:
                command.extend(["--set", override])
            command.extend(args.extra)
            variant_suffix = f"_{run_name}" if run_name else ""
            log_path = launcher / f"fold{fold}_seed{seed}{variant_suffix}_{device_id}.log"
            handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
            running[process.pid] = (
                process, handle, device_id, fold, seed, variant_index, run_name,
                checkpoint_run_name,
                variant_overrides, command, log_path,
            )
        time.sleep(1)
        for pid, item in list(running.items()):
            (
                process, handle, device_id, fold, seed, variant_index, run_name,
                checkpoint_run_name,
                variant_overrides, command, log_path,
            ) = item
            return_code = process.poll()
            if return_code is None:
                continue
            handle.close()
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            status = (
                "failed" if return_code != 0
                else "checkpointed" if "Run checkpointed:" in log_text
                else "completed"
            )
            results.append(
                {
                    "fold": fold, "seed": seed,
                    "grid_index": variant_index if args.grid else "",
                    "run_name": run_name or "",
                    "checkpoint_run_name": checkpoint_run_name or "",
                    "grid_overrides": " ".join(variant_overrides),
                    "device": device_id, "return_code": return_code,
                    "status": status,
                    "log": str(log_path), "command": " ".join(command),
                }
            )
            available.append(device_id)
            del running[pid]
    summary_path = launcher / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]) if results else ["status"])
        writer.writeheader()
        writer.writerows(results)
    for signum, handler in previous_handlers.items():
        signal.signal(signum, handler)
    failed = sum(row["return_code"] != 0 for row in results)
    print(f"Jobs={len(results)} failed={failed} interrupted={stop_requested}; summary={summary_path}")
    return 130 if stop_requested else 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
