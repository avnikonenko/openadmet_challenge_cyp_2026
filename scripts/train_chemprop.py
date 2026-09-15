#!/usr/bin/env python3
from _neural_cli import neural_parser, run_neural


def main() -> int:
    parser = neural_parser(
        "Train a single-task Chemprop-style CYP D-MPNN.",
        "configs/chemprop_multitask.yaml",
    )
    args = parser.parse_args()
    if not args.cyp:
        args.cyp = ["CYP2D6"]
    if not args.experiment:
        args.experiment = "chemprop_single_" + "_".join(args.cyp)
    return run_neural(args, "direct")


if __name__ == "__main__":
    raise SystemExit(main())
