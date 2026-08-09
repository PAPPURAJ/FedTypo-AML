#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "run_submission.py"
RESULT_ROOTS = {
    "ibm": ROOT / "results" / "tifs_submission_v2_ibm",
    "samld": ROOT / "results" / "tifs_submission_v2_samld",
}
CONDITIONS = ("control", "drift")
SEEDS = tuple(range(42, 52))
METHODS = (
    "local_only",
    "fedavg",
    "fedprox",
    "fedproto",
    "cda_fedavg",
    "fedtypo_noreg",
    "fedtypo",
)
METHOD_FILES = (
    "client_metrics_{method}.csv",
    "window_metrics_{method}.csv",
    "budget_metrics_{method}.csv",
    "typology_metrics_{method}.csv",
    "inoculation_{method}.csv",
)
AGGREGATES = (
    "environment.json",
    "seed_summary.csv",
    "method_summary.csv",
    "seed_level_tests.csv",
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def require(path: Path, missing: list[str]) -> None:
    if not path.is_file():
        missing.append(str(path.relative_to(ROOT)))


def main() -> None:
    missing: list[str] = []
    source_hash = file_hash(EXPERIMENT)

    for dataset, result_root in RESULT_ROOTS.items():
        for name in AGGREGATES:
            require(result_root / name, missing)
        if missing:
            continue

        environment = json.loads(
            (result_root / "environment.json").read_text(encoding="utf-8")
        )
        if environment.get("dataset") != dataset:
            raise RuntimeError(f"dataset mismatch in {result_root / 'environment.json'}")
        if environment.get("script_sha256") != source_hash:
            raise RuntimeError(f"source hash mismatch for {dataset}")
        if tuple(environment.get("seeds", ())) != SEEDS:
            raise RuntimeError(f"seed record mismatch for {dataset}")

        seed_summary = csv_rows(result_root / "seed_summary.csv")
        expected_seed_rows = len(CONDITIONS) * len(SEEDS) * len(METHODS)
        if len(seed_summary) != expected_seed_rows:
            raise RuntimeError(
                f"expected {expected_seed_rows} seed rows for {dataset}, "
                f"found {len(seed_summary)}"
            )

        for condition in CONDITIONS:
            for seed in SEEDS:
                run_root = result_root / f"{condition}_s{seed}"
                require(run_root / "client_profile.csv", missing)
                require(run_root / "client_typology_counts.csv", missing)
                require(run_root / "drift_events.csv", missing)
                for method in METHODS:
                    for pattern in METHOD_FILES:
                        require(run_root / pattern.format(method=method), missing)
                for method in ("fedtypo_noreg", "fedtypo"):
                    require(run_root / f"prototype_{method}.csv", missing)

    predictions = tuple((ROOT / "results").rglob("preds_*.csv*"))
    if predictions:
        joined = ", ".join(str(path.relative_to(ROOT)) for path in predictions)
        raise RuntimeError(f"raw prediction files are present: {joined}")
    if missing:
        joined = "\n".join(f"- {path}" for path in sorted(set(missing)))
        raise RuntimeError(f"required files are missing:\n{joined}")

    result_files = sum(1 for path in (ROOT / "results").rglob("*") if path.is_file())
    print(f"Release validation passed: {result_files} result files, source {source_hash}")


if __name__ == "__main__":
    main()
