#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import _bootstrap  # noqa: F401
from src.submission import build_submission, official_test_frame, validate_submission


def main() -> int:
    parser = argparse.ArgumentParser(description="Write and validate an OpenADMET regression submission")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Submission already exists: {args.output}; use --force to replace it")
    test = official_test_frame()
    submission = build_submission(pd.read_csv(args.predictions), test)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    reloaded = pd.read_csv(args.output)
    summary = validate_submission(reloaded, test)
    print(
        f"Validated submission: rows={summary['rows']} targets={summary['targets']} "
        f"finite_predictions={summary['finite_values']} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
