#!/usr/bin/env python3
from _neural_cli import neural_parser, run_neural


def main() -> int:
    parser = neural_parser(
        "Fine-tune a pretrained D-MPNN encoder on direct pIC50.",
        "configs/transfer_singleconc.yaml",
    )
    args = parser.parse_args()
    if args.checkpoint is None:
        parser.error("--checkpoint is required for fine-tuning")
    if args.transfer_mode is None:
        args.transfer_mode = "full"
    return run_neural(args, "direct")


if __name__ == "__main__":
    raise SystemExit(main())
