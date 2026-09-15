#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from src.model_registry import update_model_statistics


def main() -> int:
    parser = argparse.ArgumentParser(description="Update the timestamped model-statistics registry")
    parser.add_argument("--input", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/experiment_leaderboard.csv"))
    args = parser.parse_args()
    statistics = update_model_statistics(args.input, args.output)
    runs = statistics["run_id"].nunique() if len(statistics) else 0
    completed = int(statistics["completion_status"].eq("completed").sum()) if len(statistics) else 0
    print(f"Runs={runs}; rows={len(statistics)}; completed_rows={completed}; output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
