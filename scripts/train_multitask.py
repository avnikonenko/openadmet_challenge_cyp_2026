#!/usr/bin/env python3
from _neural_cli import neural_parser, run_neural


def main() -> int:
    parser = neural_parser(
        "Train a masked multitask CYP D-MPNN with CYP-specific heads.",
        "configs/chemprop_multitask.yaml",
    )
    return run_neural(parser.parse_args(), "direct")


if __name__ == "__main__":
    raise SystemExit(main())
