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
    grid_combinations = [()]
    grid_keys: list[str] = []
    if args.grid:
        import yaml

        with args.config.open(encoding="utf-8") as handle:
            search_grid = (yaml.safe_load(handle) or {}).get("search_grid", {})
        if not search_grid or any(not isinstance(values, list) or not values for values in search_grid.values()):
            parser.error("--grid requires a nonempty search_grid whose values are nonempty lists")
        if "--run-name" in args.extra:
            parser.error("Do not combine --grid with an explicit --run-name")
        grid_keys = list(search_grid)
        grid_combinations = list(itertools.product(*(search_grid[key] for key in grid_keys)))
    launcher = args.output_dir / "launcher_logs" / f"{args.script.stem}_{time.time_ns()}"
    launcher.mkdir(parents=True, exist_ok=True)
    queue = deque(
        (fold, seed, grid_index, combination)
        for fold in args.folds
        for seed in args.seeds
        for grid_index, combination in enumerate(grid_combinations)
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
            fold, seed, grid_index, combination = queue.popleft()
            run_name = f"grid{grid_index:03d}" if args.grid else None
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
                    fold=fold, seed=seed, grid=grid_index, run_name=run_name or ""
                )
                command.extend(["--checkpoint", checkpoint])
            if run_name:
                command.extend(["--run-name", run_name])
            grid_overrides = [
                f"{key}={json.dumps(value)}"
                for key, value in zip(grid_keys, combination)
            ]
            for override in grid_overrides:
                command.extend(["--set", override])
            command.extend(args.extra)
            grid_suffix = f"_{run_name}" if run_name else ""
            log_path = launcher / f"fold{fold}_seed{seed}{grid_suffix}_{device_id}.log"
            handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
            running[process.pid] = (
                process, handle, device_id, fold, seed, grid_index, grid_overrides,
                command, log_path,
            )
        time.sleep(1)
        for pid, item in list(running.items()):
            (
                process, handle, device_id, fold, seed, grid_index, grid_overrides,
                command, log_path,
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
                    "grid_index": grid_index if args.grid else "",
                    "grid_overrides": " ".join(grid_overrides),
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
