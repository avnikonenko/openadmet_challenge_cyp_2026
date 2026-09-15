#!/usr/bin/env python3
from _neural_cli import neural_parser, run_neural


def main() -> int:
    parser = neural_parser(
        "Fold-safe D-MPNN pretraining on single-concentration inhibition.",
        "configs/transfer_singleconc.yaml",
    )
    return run_neural(parser.parse_args(), "single_concentration")


if __name__ == "__main__":
    raise SystemExit(main())
