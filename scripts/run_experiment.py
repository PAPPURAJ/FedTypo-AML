#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import runpy
from pathlib import Path


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the FedTypo submission experiment."
    )
    parser.add_argument("--dataset", choices=("ibm", "samld"), required=True)
    parser.add_argument(
        "--partition",
        choices=("account_hash", "typology_skew"),
        default="account_hash",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--seeds", type=positive_integer, default=10)
    parser.add_argument("--methods", help="comma-separated method identifiers")
    parser.add_argument(
        "--conditions", default="control,drift", help="control, drift, or both"
    )
    parser.add_argument("--run-name", default="tifs_revision_v1")
    parser.add_argument(
        "--optimizer-policy",
        choices=("per_round", "broadcast_only"),
        default="per_round",
    )
    parser.add_argument("--fast-dev", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not data_root.is_dir():
        parser.error(f"data directory does not exist: {data_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    os.environ["FT_DATASET"] = args.dataset
    os.environ["FT_PARTITION"] = args.partition
    os.environ["FT_OUT"] = str(output_root)
    os.environ["FT_N_SEEDS"] = str(args.seeds)
    os.environ["FT_CONDITIONS"] = args.conditions
    os.environ["FT_RUN_NAME"] = args.run_name
    os.environ["FT_OPTIMIZER_POLICY"] = args.optimizer_policy
    if args.methods:
        os.environ["FT_METHODS"] = args.methods
    else:
        os.environ.pop("FT_METHODS", None)
    os.environ["FT_FAST_DEV"] = "1" if args.fast_dev else "0"
    os.environ["FT_SAVE_PREDICTIONS"] = "1" if args.save_predictions else "0"
    if args.dataset == "ibm":
        os.environ["FT_DATA_ROOT"] = str(data_root)
    else:
        os.environ["FT_SAMLD_ROOT"] = str(data_root)

    experiment = Path(__file__).resolve().parents[1] / "experiments" / "run_submission.py"
    runpy.run_path(str(experiment), run_name="__main__")


if __name__ == "__main__":
    main()
