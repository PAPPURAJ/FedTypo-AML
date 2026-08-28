#!/usr/bin/env python3
"""Validate the compact FedTypo revision artifact using the standard library."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXECUTED_SOURCE = ROOT / "experiments" / "run_submission_executed_201bf0f.py"
CURRENT_SOURCE = ROOT / "experiments" / "run_submission.py"
POSTPROCESSOR = ROOT / "scripts" / "fix_inference_tie_counts.py"
SEEDS = tuple(range(42, 52))
CONDITIONS = ("control", "drift")
PRIMARY_METHODS = (
    "local_only", "fedavg", "fedprox", "fedproto", "fedtypo_noreg",
    "fedtypo", "cda_fedavg", "ablate_g1", "ablate_g3", "ablate_random",
    "ablate_nommd", "ablate_samplewt", "ablate_rho0",
)
SECONDARY_METHODS = ("fedavg", "fedtypo_noreg", "fedtypo")
ROOTS = {
    "tifs_revision_v1_ibm_account_hash": ("ibm", "account_hash", PRIMARY_METHODS),
    "tifs_revision_v1_samld_account_hash": ("samld", "account_hash", PRIMARY_METHODS),
    "tifs_revision_partition_sensitivity_ibm_typology_skew": (
        "ibm", "typology_skew", SECONDARY_METHODS,
    ),
    "tifs_revision_partition_sensitivity_samld_typology_skew": (
        "samld", "typology_skew", SECONDARY_METHODS,
    ),
}
AGGREGATES = (
    "environment.json", "seed_summary.csv", "method_summary.csv",
    "seed_level_tests.csv", "paired_differences.csv", "ablation_tests.csv",
    "ablation_paired_differences.csv", "mechanism_tests.csv",
    "raw_window_counts.csv", "postprocess_inference_counts.json",
)
METHOD_FILES = (
    "client_metrics_{method}.csv", "window_metrics_{method}.csv",
    "stream_metrics_{method}.csv", "budget_metrics_{method}.csv",
    "typology_metrics_{method}.csv", "inoculation_{method}.csv",
)
PROTOTYPE_METHODS = {
    "fedtypo_noreg", "fedtypo", "ablate_g1", "ablate_g3", "ablate_random",
    "ablate_nommd", "ablate_samplewt", "ablate_rho0",
}
ASSETS = (
    "artifacts/manuscript.pdf", "artifacts/results_auto.tex",
    "artifacts/figures/architecture.pdf", "artifacts/figures/results_main.pdf",
    "artifacts/figures/temporal_support.pdf", "artifacts/figures/ablations.pdf",
    "artifacts/figures/budget_sensitivity.pdf",
    "artifacts/figures/prototype_fidelity.pdf",
)
NONFINITE = {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity",
             "+infinity", "-infinity"}


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


def assert_finite_csv(path: Path) -> None:
    for line_number, row in enumerate(csv_rows(path), start=2):
        for key, value in row.items():
            if value is not None and value.strip().lower() in NONFINITE:
                raise RuntimeError(
                    f"non-finite token in {path.relative_to(ROOT)} "
                    f"line {line_number}, column {key}"
                )


def validate_postprocess(result_root: Path, executed_hash: str, post_hash: str) -> None:
    manifest = json.loads(
        (result_root / "postprocess_inference_counts.json").read_text(encoding="utf-8")
    )
    if manifest.get("training_script_sha256") != executed_hash:
        raise RuntimeError(f"training hash mismatch in {result_root.name} manifest")
    if manifest.get("postprocessor_sha256") != post_hash:
        raise RuntimeError(f"postprocessor hash mismatch in {result_root.name}")
    for name, expected in manifest.get("after_sha256", {}).items():
        if file_hash(result_root / name) != expected:
            raise RuntimeError(f"postprocessed-file hash mismatch: {result_root.name}/{name}")


def main() -> None:
    missing: list[str] = []
    for path in (EXECUTED_SOURCE, CURRENT_SOURCE, POSTPROCESSOR):
        require(path, missing)
    for relative in ASSETS:
        require(ROOT / relative, missing)
    if missing:
        raise RuntimeError("required files are missing:\n" + "\n".join(missing))

    executed_hash = file_hash(EXECUTED_SOURCE)
    post_hash = file_hash(POSTPROCESSOR)
    total_result_files = 0

    for name, (dataset, partition, methods) in ROOTS.items():
        result_root = ROOT / "results" / name
        for aggregate in AGGREGATES:
            require(result_root / aggregate, missing)
        if missing:
            continue

        environment = json.loads(
            (result_root / "environment.json").read_text(encoding="utf-8")
        )
        expected_support = 10 if dataset == "ibm" else 19
        checks = {
            "dataset": environment.get("dataset") == dataset,
            "partition": environment.get("config", {}).get("partition_mode") == partition,
            "optimizer": environment.get("config", {}).get("optimizer_policy") == "per_round",
            "source hash": environment.get("script_sha256") == executed_hash,
            "seeds": tuple(environment.get("seeds", ())) == SEEDS,
            "conditions": tuple(environment.get("conditions", ())) == CONDITIONS,
            "methods": tuple(environment.get("methods", ())) == tuple(methods),
            "support": environment.get("config", {}).get("supported_windows") == expected_support,
        }
        failed = [label for label, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError(f"{name} metadata mismatch: {', '.join(failed)}")

        seed_summary = csv_rows(result_root / "seed_summary.csv")
        expected_rows = len(CONDITIONS) * len(SEEDS) * len(methods)
        if len(seed_summary) != expected_rows:
            raise RuntimeError(
                f"{name}: expected {expected_rows} seed rows, found {len(seed_summary)}"
            )

        for condition in CONDITIONS:
            for seed in SEEDS:
                run_root = result_root / f"{condition}_s{seed}"
                for common in (
                    "client_profile.csv", "client_typology_counts.csv",
                    "drift_events.csv", "window_support.csv",
                    "window_typology_counts.csv",
                ):
                    require(run_root / common, missing)
                if condition == "drift" and (run_root / "drift_events.csv").is_file():
                    if len(csv_rows(run_root / "drift_events.csv")) != 4:
                        raise RuntimeError(f"expected four drift events in {run_root}")
                for method in methods:
                    require(run_root / f"DONE_{method}", missing)
                    for pattern in METHOD_FILES:
                        require(run_root / pattern.format(method=method), missing)
                    if method in PROTOTYPE_METHODS:
                        require(run_root / f"prototype_{method}.csv", missing)
                        # Registry files are emitted only when at least one entry
                        # is admitted; absence therefore represents an empty
                        # registry rather than an incomplete run.

        validate_postprocess(result_root, executed_hash, post_hash)
        for csv_path in result_root.rglob("*.csv"):
            assert_finite_csv(csv_path)
        total_result_files += sum(1 for path in result_root.rglob("*") if path.is_file())

    if missing:
        raise RuntimeError(
            "required files are missing:\n" +
            "\n".join(f"- {path}" for path in sorted(set(missing)))
        )

    old_roots = tuple((ROOT / "results").glob("tifs_submission_v2_*"))
    if old_roots:
        raise RuntimeError("old result roots remain in the release")
    predictions = tuple((ROOT / "results").rglob("preds_*.csv*"))
    if predictions:
        raise RuntimeError("raw prediction files are present")

    print(
        "Release validation passed: "
        f"{len(ROOTS)} roots, {total_result_files} result files, "
        f"executed source {executed_hash}, postprocessor {post_hash}"
    )


if __name__ == "__main__":
    main()
