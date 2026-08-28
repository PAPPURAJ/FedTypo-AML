"""Repair mutually exclusive win/tie/loss counts in completed result roots.

The training outputs, metric values, p-values, confidence intervals, and effect
sizes are not modified. Only the three categorical outcome-count columns are
recomputed from stored paired differences or stored inoculation diagnostics.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def counts(values: np.ndarray) -> tuple[int, int, int]:
    values = np.asarray(values, dtype=float)
    ties = np.isclose(values, 0.0)
    wins = (values > 0) & ~ties
    losses = (values < 0) & ~ties
    return int(wins.sum()), int(ties.sum()), int(losses.sum())


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def repair_test_family(root: Path, tests_name: str, differences_name: str) -> bool:
    tests_path = root / tests_name
    differences_path = root / differences_name
    try:
        tests = pd.read_csv(tests_path)
        differences = pd.read_csv(differences_path)
    except EmptyDataError:
        return False
    if tests.empty:
        return False
    for index, row in tests.iterrows():
        selected = differences[
            (differences["condition"] == row["condition"])
            & (differences["reference"] == row["reference"])
            & (differences["comparison"] == row["comparison"])
            & (differences["metric"] == row["metric"])
        ]
        if len(selected) != int(row["n_seeds"]):
            raise RuntimeError(
                f"{tests_path}: {row['condition']}/{row['comparison']} has "
                f"{len(selected)} differences, expected {int(row['n_seeds'])}"
            )
        tests.loc[index, ["wins", "ties", "losses"]] = counts(
            selected["difference"].to_numpy(dtype=float)
        )
    atomic_csv(tests, tests_path)
    return True


def repair_mechanism(root: Path, seeds: list[int]) -> bool:
    path = root / "mechanism_tests.csv"
    try:
        tests = pd.read_csv(path)
    except EmptyDataError:
        return False
    if tests.empty:
        return False
    for index, row in tests.iterrows():
        horizon = int(row["horizon_windows"])
        differences = []
        for seed in seeds:
            directory = root / f"drift_s{seed}"
            registry = pd.read_csv(directory / "inoculation_fedtypo.csv")
            no_registry = pd.read_csv(directory / "inoculation_fedtypo_noreg.csv")
            registry = registry[registry["horizon_windows"] == horizon]
            no_registry = no_registry[no_registry["horizon_windows"] == horizon]
            difference = registry["capture_rate"].mean() - no_registry["capture_rate"].mean()
            if np.isfinite(difference):
                differences.append(float(difference))
        if len(differences) != int(row["n_seeds"]):
            raise RuntimeError(
                f"{path}: horizon {horizon} has {len(differences)} finite differences, "
                f"expected {int(row['n_seeds'])}"
            )
        tests.loc[index, ["wins", "ties", "losses"]] = counts(
            np.asarray(differences, dtype=float)
        )
    atomic_csv(tests, path)
    return True


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: fix_inference_tie_counts.py RESULT_ROOT [...]")
    for root_argument in sys.argv[1:]:
        root = Path(root_argument).resolve()
        environment = json.loads((root / "environment.json").read_text(encoding="utf-8"))
        targets = [
            root / "seed_level_tests.csv",
            root / "ablation_tests.csv",
            root / "mechanism_tests.csv",
        ]
        before = {path.name: sha256(path) for path in targets if path.exists()}
        changed = {
            "seed_level_tests.csv": repair_test_family(
                root, "seed_level_tests.csv", "paired_differences.csv"
            ),
            "ablation_tests.csv": repair_test_family(
                root, "ablation_tests.csv", "ablation_paired_differences.csv"
            ),
            "mechanism_tests.csv": repair_mechanism(root, environment["seeds"]),
        }
        after = {path.name: sha256(path) for path in targets if path.exists()}
        for path in targets:
            try:
                frame = pd.read_csv(path)
            except EmptyDataError:
                continue
            if not frame.empty:
                totals = frame["wins"] + frame["ties"] + frame["losses"]
                if not (totals == frame["n_seeds"]).all():
                    raise RuntimeError(f"{path}: outcome counts remain non-exclusive")
        manifest = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "reason": "make win/tie/loss categories mutually exclusive under np.isclose",
            "training_script_sha256": environment["script_sha256"],
            "postprocessor_sha256": sha256(Path(__file__)),
            "changed": changed,
            "before_sha256": before,
            "after_sha256": after,
            "unchanged_fields": [
                "model outputs",
                "seed metrics",
                "paired differences",
                "p-values",
                "confidence intervals",
                "effect sizes",
            ],
        }
        (root / "postprocess_inference_counts.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(root, changed)


if __name__ == "__main__":
    main()
