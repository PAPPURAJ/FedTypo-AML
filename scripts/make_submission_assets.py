#!/usr/bin/env python3
"""Build the figures and LaTeX result macros for the TIFS submission.

The script reads only the consolidated outputs produced by
experiments/run_submission.py.  It deliberately performs inference across
independent seeds (not pooled client-windows) and leaves all reported digits
traceable to CSV files.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


METHODS = [
    "local_only",
    "fedavg",
    "fedprox",
    "fedproto",
    "cda_fedavg",
    "fedtypo_noreg",
    "fedtypo",
]
ABLATION_METHODS = [
    "ablate_g1",
    "ablate_g3",
    "ablate_random",
    "ablate_nommd",
    "ablate_samplewt",
    "ablate_rho0",
]
SECONDARY_METHODS = ["fedavg", "fedtypo_noreg", "fedtypo"]
EVAL_START_W = 1
LABELS = {
    "local_only": "Local",
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "fedproto": "FedProto",
    "cda_fedavg": "CDA-FedAvg",
    "fedtypo_noreg": "FedTypo-NoReg",
    "fedtypo": "FedTypo",
    "ablate_g1": "One group ($G=1$)",
    "ablate_g3": "Three groups ($G=3$)",
    "ablate_random": "Random groups",
    "ablate_nommd": "No MMD stability",
    "ablate_samplewt": "Sample weighting",
    "ablate_rho0": "No global coupling ($\\rho=0$)",
}
COLORS = {
    "local_only": "#666666",
    "fedavg": "#2878B5",
    "fedprox": "#55A868",
    "fedproto": "#E17C05",
    "cda_fedavg": "#CCB974",
    "fedtypo_noreg": "#C44E52",
    "fedtypo": "#7A3E9D",
    "ablate_g1": "#4C78A8",
    "ablate_g3": "#F58518",
    "ablate_random": "#54A24B",
    "ablate_nommd": "#E45756",
    "ablate_samplewt": "#72B7B2",
    "ablate_rho0": "#B279A2",
}
MARKERS = {
    "local_only": "o",
    "fedavg": "s",
    "fedprox": "^",
    "fedproto": "X",
    "cda_fedavg": "D",
    "fedtypo_noreg": "v",
    "fedtypo": "P",
    "ablate_g1": "o",
    "ablate_g3": "s",
    "ablate_random": "^",
    "ablate_nommd": "D",
    "ablate_samplewt": "v",
    "ablate_rho0": "P",
}

METHOD_MACROS = {
    "local_only": "Local",
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "fedproto": "FedProto",
    "cda_fedavg": "CDA",
    "fedtypo_noreg": "NoReg",
    "fedtypo": "FedTypo",
    "ablate_g1": "GOne",
    "ablate_g3": "GThree",
    "ablate_random": "RandomGroups",
    "ablate_nommd": "NoMMD",
    "ablate_samplewt": "SampleWeight",
    "ablate_rho0": "RhoZero",
}


def fmt(x: float, digits: int = 4) -> str:
    if not np.isfinite(x):
        return "--"
    return f"{x:.{digits}f}"


def pct(x: float, digits: int = 1) -> str:
    if not np.isfinite(x):
        return "--"
    return f"{100*x:.{digits}f}\\%"


def sci(x: float) -> str:
    if not np.isfinite(x):
        return "--"
    if x <= 0:
        return "0"
    if x >= 0.001:
        return f"{x:.3f}"
    exponent = int(math.floor(math.log10(x)))
    mantissa = x / (10**exponent)
    return f"{mantissa:.2f}\\!\\times\\!10^{{{exponent}}}"


def seed_dirs(root: Path, condition: str) -> list[Path]:
    return sorted(
        root.glob(f"{condition}_s*"),
        key=lambda p: int(p.name.rsplit("_s", 1)[1]),
    )


def seed_id(path: Path) -> int:
    return int(path.name.rsplit("_s", 1)[1])


def read_csv_columns(
    path: Path, required: set[str], *, allow_empty: bool = False
) -> pd.DataFrame:
    """Read a CSV and fail with a useful schema error."""
    if not path.exists():
        raise RuntimeError(f"Missing result file: {path}")
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        if allow_empty:
            return pd.DataFrame(columns=sorted(required))
        raise RuntimeError(f"Empty result file: {path}") from exc
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{path}: missing columns {sorted(missing)}")
    if not allow_empty and frame.empty:
        raise RuntimeError(f"Empty result file: {path}")
    return frame


def load_raw_window_counts(root: Path) -> pd.DataFrame:
    """Normalize the loader's native time_window name for plotting/auditing."""
    path = root / "raw_window_counts.csv"
    frame = read_csv_columns(path, {"transactions"})
    if "window" not in frame.columns:
        if "time_window" not in frame.columns:
            raise RuntimeError(f"{path}: missing window/time_window column")
        frame = frame.rename(columns={"time_window": "window"})
    return frame


def validate_finite(frame: pd.DataFrame, columns: list[str], context: Path) -> None:
    values = frame[columns].to_numpy(dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise RuntimeError(f"{context}: non-finite values in {columns}")


def validate_paired_tests(
    frame: pd.DataFrame,
    path: Path,
    *,
    reference: str,
    comparisons: set[str],
    n_seeds: int,
) -> None:
    validate_finite(
        frame,
        [
            "n_seeds",
            "statistic",
            "p_raw",
            "p_holm",
            "mean_difference",
            "difference_ci_lo",
            "difference_ci_hi",
            "median_difference",
            "rank_biserial",
            "wins",
            "ties",
            "losses",
            "relative_gain",
        ],
        path,
    )
    if set(frame["reference"]) != {reference} or set(frame["metric"]) != {"auprc"}:
        raise RuntimeError(f"{path}: unexpected reference or metric")
    for condition in ("control", "drift"):
        present = set(frame.loc[frame.condition == condition, "comparison"])
        if present != comparisons:
            raise RuntimeError(
                f"{path}: {condition} comparisons {sorted(present)}; "
                f"expected {sorted(comparisons)}"
            )
    if (
        not frame["p_raw"].between(0, 1).all()
        or not frame["p_holm"].between(0, 1).all()
        or not frame["rank_biserial"].between(-1, 1).all()
        or not (frame["n_seeds"] == n_seeds).all()
        or not (
            frame["wins"] + frame["ties"] + frame["losses"] == frame["n_seeds"]
        ).all()
        or not (frame["difference_ci_lo"] <= frame["difference_ci_hi"]).all()
    ):
        raise RuntimeError(f"{path}: invalid paired two-sided inference values")


def validate_root(
    root: Path,
    *,
    require_ablations: bool = True,
    primary_methods: list[str] | tuple[str, ...] = METHODS,
    expected_partition: str | None = None,
) -> None:
    """Fail before asset generation if a consolidated run is incomplete.

    The revised protocol makes stream-level client AUPRC the primary metric.
    Validation therefore checks the per-client stream files themselves as
    well as the temporal-support audit used to determine the evaluated tail.
    """
    expected_seeds = set(range(42, 52))
    primary_methods = list(primary_methods)
    required_methods = primary_methods + (ABLATION_METHODS if require_ablations else [])

    environment_path = root / "environment.json"
    if not environment_path.exists():
        raise RuntimeError(f"Missing result file: {environment_path}")
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    configured_seeds = {int(seed) for seed in environment.get("seeds", [])}
    if configured_seeds and configured_seeds != expected_seeds:
        raise RuntimeError(
            f"{environment_path}: seeds {sorted(configured_seeds)}; "
            f"expected {sorted(expected_seeds)}"
        )
    configured_methods = set(environment.get("methods", []))
    missing_methods = set(required_methods) - configured_methods
    if configured_methods and missing_methods:
        raise RuntimeError(
            f"{environment_path}: missing configured methods "
            f"{sorted(missing_methods)}"
        )
    partition = environment.get("config", {}).get("partition_mode")
    if expected_partition is not None and partition != expected_partition:
        raise RuntimeError(
            f"{environment_path}: partition_mode={partition!r}; "
            f"expected {expected_partition!r}"
        )

    raw_support = load_raw_window_counts(root)
    validate_finite(raw_support, ["window", "transactions"], root / "raw_window_counts.csv")
    if raw_support["window"].duplicated().any() or (raw_support["transactions"] <= 0).any():
        raise RuntimeError(f"{root / 'raw_window_counts.csv'}: invalid window support")

    for condition in ("control", "drift"):
        directories = seed_dirs(root, condition)
        observed = {seed_id(path) for path in directories}
        if observed != expected_seeds:
            raise RuntimeError(
                f"{root}: {condition} seeds {sorted(observed)}; "
                f"expected {sorted(expected_seeds)}"
            )
        for directory in directories:
            extra_required = [
                directory / "client_profile.csv",
                directory / "client_typology_counts.csv",
                directory / "prototype_fedtypo.csv",
                directory / "window_support.csv",
                directory / "window_typology_counts.csv",
            ]
            if condition == "drift":
                extra_required.extend(
                    [
                        directory / "inoculation_fedtypo.csv",
                        directory / "inoculation_fedtypo_noreg.csv",
                    ]
                )
            extra_missing = [str(path) for path in extra_required if not path.exists()]
            if extra_missing:
                raise RuntimeError(
                    "Missing diagnostic result files:\n" + "\n".join(extra_missing)
                )
            profile = read_csv_columns(
                directory / "client_profile.csv",
                {"client", "transactions", "positives", "median_confirmation"},
            )
            n_clients = profile["client"].nunique()
            if n_clients < 2 or (profile[["transactions", "positives"]].to_numpy() < 0).any():
                raise RuntimeError(f"{directory}: invalid client_profile.csv")

            support = read_csv_columns(
                directory / "window_support.csv",
                {
                    "window",
                    "transactions",
                    "positives",
                    "prevalence",
                    "positive_clients",
                },
            )
            validate_finite(
                support,
                ["window", "transactions", "positives", "prevalence", "positive_clients"],
                directory / "window_support.csv",
            )
            expected_prevalence = support["positives"] / support["transactions"]
            if (
                support["window"].duplicated().any()
                or (support["transactions"] <= 0).any()
                or (support["positives"] < 0).any()
                or (support["positives"] > support["transactions"]).any()
                or (support["positive_clients"] < 0).any()
                or (support["positive_clients"] > n_clients).any()
                or not np.allclose(support["prevalence"], expected_prevalence)
                or not set(support["window"]).issubset(set(raw_support["window"]))
            ):
                raise RuntimeError(f"{directory}: invalid temporal-support diagnostics")

            read_csv_columns(
                directory / "window_typology_counts.csv",
                {"window", "typology", "positives"},
            )
            for method in required_methods:
                required = [
                    directory / f"DONE_{method}",
                    directory / f"client_metrics_{method}.csv",
                    directory / f"stream_metrics_{method}.csv",
                    directory / f"window_metrics_{method}.csv",
                    directory / f"budget_metrics_{method}.csv",
                ]
                missing = [str(path) for path in required if not path.exists()]
                if missing:
                    raise RuntimeError("Missing result files:\n" + "\n".join(missing))

                stream_path = directory / f"stream_metrics_{method}.csv"
                stream = read_csv_columns(
                    stream_path,
                    {"client", "auprc", "ap_lift", "positives", "transactions", "prevalence"},
                )
                if stream["client"].duplicated().any() or stream["client"].nunique() != n_clients:
                    raise RuntimeError(
                        f"{stream_path}: expected one stream-level row for each of "
                        f"{n_clients} clients"
                    )
                validate_finite(
                    stream,
                    ["auprc", "ap_lift", "positives", "transactions", "prevalence"],
                    stream_path,
                )
                stream_prevalence = stream["positives"] / stream["transactions"]
                if (
                    (stream["positives"] <= 0).any()
                    or (stream["transactions"] <= 0).any()
                    or not np.allclose(stream["prevalence"], stream_prevalence)
                    or not np.allclose(stream["ap_lift"], stream["auprc"] / stream["prevalence"])
                ):
                    raise RuntimeError(f"{stream_path}: inconsistent stream metrics")

                window_path = directory / f"window_metrics_{method}.csv"
                windows = read_csv_columns(
                    window_path,
                    {
                        "window",
                        "auprc",
                        "ap_lift",
                        "p_at_budget",
                        "positive_clients",
                        "total_clients",
                        "positives",
                        "transactions",
                        "prevalence",
                    },
                )
                if not (windows["window"] >= EVAL_START_W).any():
                    raise RuntimeError(
                        f"{directory}/{method}: no evaluated window after warm-up"
                    )
            if condition == "drift":
                events = read_csv_columns(
                    directory / "drift_events.csv",
                    {"kind", "client", "typology", "window", "moved"},
                )
                if len(events) != 4 or (events["moved"] <= 0).any():
                    raise RuntimeError(
                        f"{directory}: expected four nonzero drift events, got\n"
                        f"{events.to_string(index=False)}"
                    )
                if "ibm" in root.name.lower() and (events["typology"] == 8).any():
                    raise RuntimeError(
                        f"{directory}: IBM UNSTRUCTURED category cannot be a "
                        "controlled drift target"
                    )
            proto = read_csv_columns(
                directory / "prototype_fedtypo.csv",
                {
                    "window",
                    "client",
                    "purity",
                    "nmi",
                    "ari",
                    "named_purity",
                    "named_nmi",
                    "named_ari",
                    "cosine_gap",
                    "confirmed_positives",
                },
            )
            proto_values = proto[["purity", "nmi", "ari"]].to_numpy(dtype=float)
            if not np.isfinite(proto_values).any(axis=0).all():
                raise RuntimeError(
                    f"{directory}: missing or non-finite prototype-fidelity values"
                )

    seed_summary = read_csv_columns(
        root / "seed_summary.csv",
        {
            "condition",
            "seed",
            "method",
            "auprc",
            "ap_lift",
            "p_at_budget",
            "window_macro_auprc",
            "transaction_weighted_window_auprc",
        },
    )
    selected_seed_summary = seed_summary[seed_summary["method"].isin(required_methods)]
    if len(selected_seed_summary) != 2 * len(expected_seeds) * len(required_methods):
        raise RuntimeError(f"{root}: incomplete seed_summary.csv")
    validate_finite(
        selected_seed_summary,
        [
            "auprc",
            "ap_lift",
            "p_at_budget",
            "window_macro_auprc",
            "transaction_weighted_window_auprc",
        ],
        root / "seed_summary.csv",
    )

    for row in selected_seed_summary.itertuples(index=False):
        stream_path = root / f"{row.condition}_s{row.seed}" / f"stream_metrics_{row.method}.csv"
        stream = pd.read_csv(stream_path)
        if not np.isclose(row.auprc, stream["auprc"].mean(), rtol=1e-9, atol=1e-12):
            raise RuntimeError(
                f"{stream_path}: stream AUPRC does not match seed_summary.csv"
            )

    summary_columns = {"condition", "method", "n_seeds"}
    for metric in (
        "auprc",
        "ap_lift",
        "p_at_budget",
        "window_macro_auprc",
        "transaction_weighted_window_auprc",
    ):
        summary_columns.update(
            {f"{metric}_mean", f"{metric}_std", f"{metric}_ci_lo", f"{metric}_ci_hi"}
        )
    summary = read_csv_columns(root / "method_summary.csv", summary_columns)
    selected_summary = summary[summary["method"].isin(required_methods)]
    if len(selected_summary) != 2 * len(required_methods):
        raise RuntimeError(f"{root}: incomplete method_summary.csv")

    test_columns = {
        "condition",
        "reference",
        "comparison",
        "metric",
        "n_seeds",
        "statistic",
        "p_raw",
        "p_holm",
        "mean_difference",
        "difference_ci_lo",
        "difference_ci_hi",
        "median_difference",
        "cohen_dz",
        "rank_biserial",
        "wins",
        "ties",
        "losses",
        "relative_gain",
    }
    primary_tests = read_csv_columns(root / "seed_level_tests.csv", test_columns)
    validate_paired_tests(
        primary_tests,
        root / "seed_level_tests.csv",
        reference="fedtypo",
        comparisons=set(primary_methods) - {"fedtypo"},
        n_seeds=len(expected_seeds),
    )

    if require_ablations:
        ablation_tests = read_csv_columns(root / "ablation_tests.csv", test_columns)
        validate_paired_tests(
            ablation_tests,
            root / "ablation_tests.csv",
            reference="fedtypo_noreg",
            comparisons=set(ABLATION_METHODS),
            n_seeds=len(expected_seeds),
        )

    mechanism = read_csv_columns(
        root / "mechanism_tests.csv",
        {
            "horizon_windows",
            "n_seeds",
            "statistic",
            "p_raw",
            "p_holm",
            "mean_difference",
            "difference_ci_lo",
            "difference_ci_hi",
            "rank_biserial",
            "wins",
            "ties",
            "losses",
        },
    )
    validate_finite(
        mechanism,
        [
            "horizon_windows",
            "n_seeds",
            "statistic",
            "p_raw",
            "p_holm",
            "mean_difference",
            "difference_ci_lo",
            "difference_ci_hi",
            "rank_biserial",
            "wins",
            "ties",
            "losses",
        ],
        root / "mechanism_tests.csv",
    )
    if (
        set(mechanism["horizon_windows"].astype(int)) != {1, 3, -1}
        or not mechanism["p_raw"].between(0, 1).all()
        or not mechanism["p_holm"].between(0, 1).all()
        or not mechanism["rank_biserial"].between(-1, 1).all()
        or not (mechanism["n_seeds"] == len(expected_seeds)).all()
        or not (
            mechanism["wins"] + mechanism["ties"] + mechanism["losses"]
            == mechanism["n_seeds"]
        ).all()
    ):
        raise RuntimeError(f"{root / 'mechanism_tests.csv'}: invalid mechanism tests")


def load_window_seed_means(
    root: Path, condition: str, method: str
) -> pd.DataFrame:
    frames = []
    for directory in seed_dirs(root, condition):
        path = directory / f"window_metrics_{method}.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame = frame[frame["window"] >= EVAL_START_W]
            frame["seed"] = seed_id(directory)
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def line_summary(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    grouped = frame.groupby("window")[metric]
    out = grouped.agg(["mean", "std", "count"]).reset_index()
    out["half"] = 1.96 * out["std"].fillna(0) / np.sqrt(out["count"])
    return out


def build_architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.15, 3.55))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec="#333333", size=6.7, weight="normal"):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.06,rounding_size=0.08",
            linewidth=0.9,
            edgecolor=ec,
            facecolor=fc,
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=size,
            weight=weight,
        )

    def arrow(x1, y1, x2, y2, color="#444444", style="-|>", lw=1.0):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle=style,
                mutation_scale=9,
                linewidth=lw,
                color=color,
            )
        )

    ax.text(0.25, 5.68, "Institution $k$ (repeated for $K$ silos)", fontsize=9.2, weight="bold")
    client = FancyBboxPatch(
        (0.18, 0.45),
        7.2,
        5.0,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        linewidth=1.2,
        edgecolor="#2878B5",
        facecolor="#F4F8FC",
    )
    ax.add_patch(client)
    box(0.48, 4.28, 1.35, 0.75, "Window graph\n$G_k^w$", "#DCEAF7")
    box(2.13, 4.28, 1.45, 0.75, "Online local\nstatistics", "#DCEAF7")
    box(3.90, 4.13, 1.66, 1.05, "Shared backbone\nattention + GRU", "#BFD8EF", size=5.6, weight="bold")
    box(5.92, 4.28, 1.10, 0.75, "Local head\n$r_k(e)$", "#E8DDF1")
    arrow(1.83, 4.65, 2.13, 4.65)
    arrow(3.58, 4.65, 3.90, 4.65)
    arrow(5.56, 4.65, 5.92, 4.65)

    box(0.48, 2.68, 1.45, 0.85, "Confirmed labels\n+ current alerts", "#FDE7C6", size=5.7)
    box(2.26, 2.68, 1.48, 0.85, "Focal-loss\ntraining", "#FDE7C6")
    arrow(1.93, 3.10, 2.26, 3.10)
    arrow(3.00, 3.53, 4.38, 4.13)

    box(4.02, 2.55, 1.38, 1.10, "Positive\nprototypes\n$\\mathcal{P}_k,\\tau_k$", "#DDF0DF", size=6.1)
    box(5.62, 2.55, 1.35, 1.10, "MMD stability\n$s_k$ + novelty", "#DDF0DF", size=6.1)
    arrow(4.70, 4.13, 4.70, 3.65)
    arrow(5.40, 3.10, 5.62, 3.10)

    box(1.05, 1.12, 2.12, 0.75, "Upload backbone $\\theta_k$", "#FFFFFF", size=5.8)
    box(4.05, 1.12, 2.15, 0.75, "Upload $\\mathcal{P}_k$, $s_k$, candidate", "#FFFFFF", size=5.6)
    arrow(3.00, 2.68, 2.15, 1.87)
    arrow(5.30, 2.55, 5.15, 1.87)

    ax.text(8.05, 5.68, "Coordination server", fontsize=9.2, weight="bold")
    server = FancyBboxPatch(
        (7.82, 0.45),
        4.0,
        5.0,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        linewidth=1.2,
        edgecolor="#7A3E9D",
        facecolor="#FAF7FC",
    )
    ax.add_patch(server)
    box(8.22, 4.28, 3.18, 0.76, "Chamfer similarity $\\rightarrow$ groups", "#E8DDF1")
    box(8.22, 2.96, 3.18, 0.86, "Weighted group mean\n+ $\\rho$ global coupling", "#E8DDF1", weight="bold")
    box(8.22, 1.54, 3.18, 0.86, "De-duplicated registry $\\mathcal{R}$", "#E8DDF1")
    arrow(9.81, 4.28, 9.81, 3.82)
    arrow(7.38, 1.97, 8.22, 1.97, color="#7A3E9D")
    arrow(7.38, 3.39, 8.22, 3.39, color="#7A3E9D")
    arrow(8.22, 3.39, 6.85, 4.25, color="#2878B5")
    arrow(8.22, 1.97, 6.90, 4.28, color="#2878B5")
    ax.text(
        8.12,
        0.74,
        "Raw data, account IDs, local head,\nand running statistics stay local.",
        fontsize=5.25,
        ha="left",
        va="center",
        color="#333333",
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build_results_figure(roots: dict[str, Path], path: Path) -> None:
    panels = [
        ("IBM AMLworld", roots["ibm"], "control"),
        ("IBM AMLworld", roots["ibm"], "drift"),
        ("SAML-D", roots["samld"], "control"),
        ("SAML-D", roots["samld"], "drift"),
    ]
    # IBM retains windows 1-9 while SAML-D retains 1-18. Independent x-axes
    # prevent the shorter IBM traces from being compressed into half a panel.
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.0), sharex=False)
    keep = ["local_only", "fedavg", "fedtypo_noreg", "fedtypo"]
    for ax, (dataset, root, condition) in zip(axes.flat, panels):
        final_window = EVAL_START_W
        for method in keep:
            frame = load_window_seed_means(root, condition, method)
            if frame.empty:
                continue
            final_window = max(final_window, int(frame["window"].max()))
            summary = line_summary(frame, "auprc")
            x = summary["window"].to_numpy()
            y = summary["mean"].to_numpy()
            half = summary["half"].to_numpy()
            ax.plot(
                x,
                y,
                label=LABELS[method],
                color=COLORS[method],
                marker=MARKERS[method],
                markersize=2.7,
                linewidth=1.05,
                markevery=2,
            )
            ax.fill_between(
                x,
                np.maximum(0, y - half),
                y + half,
                color=COLORS[method],
                alpha=0.10,
                linewidth=0,
            )
        if condition == "drift":
            event_path = seed_dirs(root, "drift")[0] / "drift_events.csv"
            onsets = sorted(pd.read_csv(event_path)["window"].astype(int).unique())
            for onset in onsets:
                ax.axvline(onset, color="#BBBBBB", linewidth=0.55, linestyle="--")
        ax.set_title(f"{dataset}: {condition}", fontsize=9)
        ax.grid(True, linewidth=0.35, alpha=0.45)
        ax.tick_params(labelsize=7.5)
        ax.set_xlim(EVAL_START_W, final_window)
        ax.set_ylim(bottom=0)
    axes[1, 0].set_xlabel("Prequential window", fontsize=8)
    axes[1, 1].set_xlabel("Prequential window", fontsize=8)
    axes[0, 0].set_ylabel("Client-macro AUPRC", fontsize=8)
    axes[1, 0].set_ylabel("Client-macro AUPRC", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=7.5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def temporal_support_summary(root: Path) -> pd.DataFrame:
    frames = []
    for directory in seed_dirs(root, "control"):
        frame = pd.read_csv(directory / "window_support.csv")
        frame["seed"] = seed_id(directory)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    support = pd.concat(frames, ignore_index=True)
    return (
        support.groupby("window", as_index=False)
        .agg(
            transactions=("transactions", "mean"),
            prevalence=("prevalence", "mean"),
            positive_clients=("positive_clients", "mean"),
        )
        .sort_values("window")
    )


def build_temporal_support_figure(roots: dict[str, Path], path: Path) -> None:
    """Show the label-free support rule at its final manuscript size."""
    datasets = [("IBM AMLworld", roots["ibm"]), ("SAML-D", roots["samld"])]
    fig, ax = plt.subplots(figsize=(3.65, 1.28))
    colors = {"IBM AMLworld": "#2878B5", "SAML-D": "#C44E52"}
    markers = {"IBM AMLworld": "o", "SAML-D": "D"}
    for dataset, root in datasets:
        raw = load_raw_window_counts(root).sort_values("window")
        support = temporal_support_summary(root)
        if support.empty:
            continue
        last_retained = int(support["window"].max())
        kept = raw[raw["window"] <= last_retained]
        ax.plot(
            kept["window"],
            kept["transactions"],
            color=colors[dataset],
            marker=markers[dataset],
            markersize=2.5,
            linewidth=1.0,
            label="IBM supported" if dataset == "IBM AMLworld" else "SAML-D supported",
            zorder=3,
        )
        if int(raw["window"].max()) > last_retained:
            tail = raw[raw["window"] >= last_retained]
            ax.plot(
                tail["window"],
                tail["transactions"],
                color="#9A4444",
                marker="x",
                markersize=2.8,
                linewidth=0.9,
                linestyle="--",
                label="IBM tail",
                zorder=2,
            )
            ax.axvspan(
                last_retained + 0.5,
                float(raw["window"].max()) + 0.5,
                color="#F4B6B6",
                alpha=0.18,
                linewidth=0,
            )
    ax.axvspan(-0.5, 0.5, color="#F6D58F", alpha=0.18, linewidth=0)
    ax.set_yscale("log")
    ax.set_xlabel("Natural window", fontsize=7)
    ax.set_ylabel("Transactions (log)", fontsize=7)
    ax.set_xlim(-0.6, 18.6)
    ax.set_xticks([0, 5, 10, 15])
    ax.grid(True, axis="y", linewidth=0.35, alpha=0.4)
    ax.tick_params(labelsize=6.8)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        fontsize=6.2,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86), pad=0.35)
    fig.savefig(path)
    plt.close(fig)


def build_ablation_figure(roots: dict[str, Path], path: Path) -> None:
    """Forest plot of paired NoReg-minus-ablation AUPRC differences."""
    datasets = [("IBM AMLworld", roots["ibm"]), ("SAML-D", roots["samld"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.15), sharey=True)
    y = np.arange(len(ABLATION_METHODS), dtype=float)
    offsets = {"control": -0.12, "drift": 0.12}
    condition_labels = {"control": "C0 control", "drift": "C1 drift"}
    condition_colors = {"control": "#2878B5", "drift": "#C44E52"}
    for ax, (dataset, root) in zip(axes, datasets):
        tests = pd.read_csv(root / "ablation_tests.csv")
        for condition in ("control", "drift"):
            block = (
                tests[tests.condition == condition]
                .set_index("comparison")
                .reindex(ABLATION_METHODS)
            )
            mean = block["mean_difference"].to_numpy(dtype=float)
            lo = block["difference_ci_lo"].to_numpy(dtype=float)
            hi = block["difference_ci_hi"].to_numpy(dtype=float)
            errors = np.vstack([np.maximum(0, mean - lo), np.maximum(0, hi - mean)])
            ax.errorbar(
                mean,
                y + offsets[condition],
                xerr=errors,
                fmt="o",
                markersize=3.8,
                capsize=2,
                elinewidth=0.9,
                color=condition_colors[condition],
                label=condition_labels[condition],
            )
        ax.axvline(0, color="#555555", linewidth=0.65, linestyle="--")
        ax.set_title(dataset, fontsize=9)
        ax.set_xlabel("Paired AUPRC difference\n(NoReg $-$ ablation)", fontsize=7.5)
        ax.grid(True, axis="x", linewidth=0.35, alpha=0.45)
        ax.tick_params(labelsize=7.2)
    axes[0].set_yticks(y, [LABELS[method] for method in ABLATION_METHODS])
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=7.2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=1.4)
    fig.savefig(path)
    plt.close(fig)


def budget_seed_frame(root: Path, condition: str, method: str) -> pd.DataFrame:
    rows = []
    for directory in seed_dirs(root, condition):
        path = directory / f"budget_metrics_{method}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame = frame[frame["window"] >= EVAL_START_W]
        for budget, group in frame.groupby("budget"):
            rows.append(
                {
                    "seed": seed_id(directory),
                    "budget": int(budget),
                    "precision": group["precision"].mean(),
                    "recall": group["recall"].mean(),
                }
            )
    return pd.DataFrame(rows)


def build_budget_figure(roots: dict[str, Path], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.65))
    keep = ["fedavg", "fedprox", "fedproto", "fedtypo_noreg", "fedtypo"]
    for ax, (dataset, root) in zip(
        axes, [("IBM AMLworld", roots["ibm"]), ("SAML-D", roots["samld"])]
    ):
        for method in keep:
            frame = budget_seed_frame(root, "drift", method)
            if frame.empty:
                continue
            summary = (
                frame.groupby("budget")["precision"]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            half = 1.96 * summary["std"].fillna(0) / np.sqrt(summary["count"])
            ax.errorbar(
                summary["budget"],
                summary["mean"],
                yerr=half,
                label=LABELS[method],
                color=COLORS[method],
                marker=MARKERS[method],
                markersize=3.5,
                linewidth=1.1,
                capsize=2,
            )
        ax.set_title(dataset, fontsize=9)
        ax.set_xlabel("Investigation budget per client-window", fontsize=8)
        ax.set_ylabel("Precision at budget", fontsize=8)
        ax.grid(True, linewidth=0.35, alpha=0.45)
        ax.tick_params(labelsize=7.5)
        ax.set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=7.0, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def final_prototype_rows(root: Path, condition: str) -> pd.DataFrame:
    rows = []
    for directory in seed_dirs(root, condition):
        path = directory / "prototype_fedtypo.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path).sort_values("window")
        frame = frame.groupby("client", as_index=False).tail(1)
        frame["seed"] = seed_id(directory)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_prototype_figure(roots: dict[str, Path], path: Path) -> None:
    groups = [
        ("IBM C0", final_prototype_rows(roots["ibm"], "control")),
        ("IBM C1", final_prototype_rows(roots["ibm"], "drift")),
        ("SAML C0", final_prototype_rows(roots["samld"], "control")),
        ("SAML C1", final_prototype_rows(roots["samld"], "drift")),
    ]
    metrics = [("purity", "Purity"), ("nmi", "NMI"), ("ari", "ARI")]
    x = np.arange(len(groups))
    width = 0.24
    # The manuscript places this panel at roughly half text width.  Matching
    # that final size preserves publication-readable typography.
    fig, ax = plt.subplots(figsize=(3.35, 1.28))
    for idx, (metric, label) in enumerate(metrics):
        means = [g[metric].mean() if not g.empty else np.nan for _, g in groups]
        errors = [
            1.96 * g[metric].std(ddof=1) / math.sqrt(len(g)) if len(g) > 1 else 0
            for _, g in groups
        ]
        ax.bar(
            x + (idx - 1) * width,
            means,
            width,
            yerr=errors,
            label=label,
            color=["#4C78A8", "#F58518", "#54A24B"][idx],
            capsize=2,
        )
    ax.axhline(0, color="#555555", linewidth=0.5)
    ax.set_xticks(x, [name for name, _ in groups])
    ax.set_ylabel("Typology agreement", fontsize=7)
    ax.set_ylim(-0.2, 1.0)
    ax.grid(True, axis="y", linewidth=0.35, alpha=0.45)
    ax.tick_params(labelsize=6.8)
    ax.legend(ncol=3, loc="upper center", fontsize=6.8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def js_divergence(probabilities: np.ndarray) -> float:
    values = []
    for i in range(len(probabilities)):
        for j in range(i + 1, len(probabilities)):
            p = probabilities[i]
            q = probabilities[j]
            m = 0.5 * (p + q)
            mask_p = p > 0
            mask_q = q > 0
            kl_pm = np.sum(p[mask_p] * np.log2(p[mask_p] / m[mask_p]))
            kl_qm = np.sum(q[mask_q] * np.log2(q[mask_q] / m[mask_q]))
            values.append(0.5 * (kl_pm + kl_qm))
    return float(np.mean(values)) if values else np.nan


def typology_js(root: Path, condition: str) -> tuple[float, float]:
    per_seed = []
    for directory in seed_dirs(root, condition):
        path = directory / "client_typology_counts.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, index_col=0).to_numpy(dtype=float)
        denom = frame.sum(axis=1, keepdims=True)
        probabilities = np.divide(
            frame, denom, out=np.zeros_like(frame), where=denom > 0
        )
        per_seed.append(js_divergence(probabilities))
    return float(np.mean(per_seed)), float(np.std(per_seed, ddof=1))


def registry_count(root: Path, condition: str) -> tuple[float, float]:
    counts = []
    for directory in seed_dirs(root, condition):
        path = directory / "registry_fedtypo.csv"
        counts.append(len(pd.read_csv(path)) if path.exists() else 0)
    return float(np.mean(counts)), float(np.std(counts, ddof=1))


def inoculation(root: Path, method: str, horizon: int = -1) -> tuple[float, float]:
    values = []
    for directory in seed_dirs(root, "drift"):
        path = directory / f"inoculation_{method}.csv"
        if path.exists():
            frame = pd.read_csv(path)
            if "horizon_windows" in frame.columns:
                frame = frame[frame.horizon_windows == horizon]
            values.append(frame["capture_rate"].mean())
    return float(np.nanmean(values)), float(np.nanstd(values, ddof=1))


def tex_macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def append_registry_beta_macros(chunks: list[str], root: Path) -> None:
    """Validate and emit the SAML-D registry-score sensitivity table."""
    expected_conditions = {"control", "drift"}
    expected_seeds = set(range(42, 52))
    expected_betas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]

    environment_path = root / "environment.json"
    if not environment_path.exists():
        raise RuntimeError(f"Missing result file: {environment_path}")
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    config = environment.get("config", {})
    recorded_betas = [float(value) for value in config.get("registry_beta_grid", [])]
    if recorded_betas != expected_betas:
        raise RuntimeError(
            f"{environment_path}: expected registry beta grid {expected_betas}, "
            f"found {recorded_betas}"
        )

    seed_table = read_csv_columns(
        root / "registry_beta_seed_summary.csv",
        {"condition", "seed", "beta", "auprc", "p_at_budget"},
    )
    if set(seed_table.condition) != expected_conditions:
        raise RuntimeError(f"{root}: incomplete registry-beta conditions")
    if set(seed_table.seed.astype(int)) != expected_seeds:
        raise RuntimeError(f"{root}: registry-beta seeds must be 42--51")
    observed_cells = set(
        zip(
            seed_table.condition,
            seed_table.seed.astype(int),
            seed_table.beta.astype(float),
        )
    )
    expected_cells = {
        (condition, seed, beta)
        for condition in expected_conditions
        for seed in expected_seeds
        for beta in expected_betas
    }
    if observed_cells != expected_cells or len(seed_table) != len(expected_cells):
        raise RuntimeError(f"{root}: registry-beta seed matrix is incomplete")
    validate_finite(seed_table, ["auprc", "p_at_budget"], root)

    summary = read_csv_columns(
        root / "registry_beta_summary.csv",
        {
            "condition",
            "beta",
            "n_seeds",
            "auprc_mean",
            "p_at_budget_mean",
        },
    )
    tests = read_csv_columns(
        root / "registry_beta_tests.csv",
        {
            "condition",
            "metric",
            "reference_beta",
            "beta",
            "n_seeds",
            "mean_difference",
            "p_holm",
        },
    )
    validate_finite(summary, ["beta", "auprc_mean", "p_at_budget_mean"], root)
    validate_finite(tests, ["beta", "mean_difference", "p_holm"], root)

    indexed = summary.set_index(["condition", "beta"])
    rows = []
    for beta in expected_betas:
        control = indexed.loc[("control", beta)]
        drift = indexed.loc[("drift", beta)]
        label = "$0.15^{\\dagger}$" if beta == 0.15 else f"{beta:.2f}"
        rows.append(
            "{} & {} & {} & {} & {} \\\\".format(
                label,
                fmt(control.auprc_mean, 6),
                fmt(control.p_at_budget_mean, 4),
                fmt(drift.auprc_mean, 6),
                fmt(drift.p_at_budget_mean, 4),
            )
        )
    chunks.append(
        "\\newcommand{\\SAMLRegistryBetaRows}{%%\n%s\n}\n" % "\n".join(rows)
    )

    reference = indexed.xs(0.0, level="beta")
    selected = indexed.xs(0.15, level="beta")
    selected_tests = tests[np.isclose(tests.beta, 0.15)].set_index(
        ["condition", "metric"]
    )
    for condition, suffix in [("control", "Control"), ("drift", "Drift")]:
        gain = selected.loc[condition, "auprc_mean"] / reference.loc[
            condition, "auprc_mean"
        ] - 1
        chunks.append(tex_macro(f"SAMLRegistryBeta{suffix}AUPRCGain", pct(gain)))
        for metric, metric_suffix in [
            ("auprc", "AUPRC"),
            ("p_at_budget", "PatFifty"),
        ]:
            row = selected_tests.loc[(condition, metric)]
            chunks.append(
                tex_macro(
                    f"SAMLRegistryBeta{suffix}{metric_suffix}PHolm",
                    sci(float(row.p_holm)),
                )
            )


def table_macro(
    name: str, frame: pd.DataFrame, methods: list[str] | tuple[str, ...] = METHODS
) -> str:
    by_method = frame.set_index("method")
    rows = []
    for method in methods:
        row = by_method.loc[method]
        rows.append(
            "{} & {} [{}, {}] & {} [{}, {}] \\\\".format(
                LABELS[method],
                fmt(row.auprc_mean),
                fmt(row.auprc_ci_lo),
                fmt(row.auprc_ci_hi),
                fmt(row.p_at_budget_mean),
                fmt(row.p_at_budget_ci_lo),
                fmt(row.p_at_budget_ci_hi),
            )
        )
    return "\\newcommand{\\%s}{%%\n%s\n}\n" % (name, "\n".join(rows))


def ablation_table_macro(
    name: str, summary: pd.DataFrame, tests: pd.DataFrame
) -> str:
    """Six component rows: performance, paired CI, effect size, and p-value."""
    by_method = summary.set_index("method")
    by_comparison = tests.set_index("comparison")
    rows = []
    for method in ABLATION_METHODS:
        result = by_method.loc[method]
        test = by_comparison.loc[method]
        rows.append(
            "{} & {} [{}, {}] & {} [{}, {}] & {} & ${}$ \\\\".format(
                LABELS[method],
                fmt(result.auprc_mean),
                fmt(result.auprc_ci_lo),
                fmt(result.auprc_ci_hi),
                fmt(test.mean_difference),
                fmt(test.difference_ci_lo),
                fmt(test.difference_ci_hi),
                fmt(test.rank_biserial, 3),
                sci(test.p_holm),
            )
        )
    return "\\newcommand{\\%s}{%%\n%s\n}\n" % (name, "\n".join(rows))


def append_metric_macros(
    chunks: list[str], prefix: str, suffix: str, block: pd.DataFrame
) -> None:
    metrics = [
        ("auprc", "AUPRC", 4),
        ("ap_lift", "APLift", 3),
        ("p_at_budget", "PatFifty", 4),
        ("window_macro_auprc", "WindowAUPRC", 4),
        ("transaction_weighted_window_auprc", "TxnWeightedWindowAUPRC", 4),
    ]
    for method in METHODS + ABLATION_METHODS:
        match = block[block.method == method]
        if match.empty:
            continue
        row = match.iloc[0]
        stem = f"{prefix}{suffix}{METHOD_MACROS[method]}"
        for column, macro, digits in metrics:
            chunks.append(tex_macro(f"{stem}{macro}", fmt(row[f"{column}_mean"], digits)))
            chunks.append(
                tex_macro(f"{stem}{macro}CILo", fmt(row[f"{column}_ci_lo"], digits))
            )
            chunks.append(
                tex_macro(f"{stem}{macro}CIHi", fmt(row[f"{column}_ci_hi"], digits))
            )


def append_test_macros(chunks: list[str], stem: str, test: pd.Series) -> None:
    """Emit every quantity from the paired, two-sided seed-level test."""
    chunks.extend(
        [
            tex_macro(f"{stem}Gain", pct(test.relative_gain)),
            tex_macro(f"{stem}PRaw", sci(test.p_raw)),
            tex_macro(f"{stem}PHolm", sci(test.p_holm)),
            tex_macro(f"{stem}Difference", fmt(test.mean_difference)),
            tex_macro(f"{stem}DifferenceCILo", fmt(test.difference_ci_lo)),
            tex_macro(f"{stem}DifferenceCIHi", fmt(test.difference_ci_hi)),
            tex_macro(f"{stem}MedianDifference", fmt(test.median_difference)),
            tex_macro(f"{stem}CohenDz", fmt(test.cohen_dz, 3)),
            tex_macro(f"{stem}RankBiserial", fmt(test.rank_biserial, 3)),
            tex_macro(f"{stem}Wins", str(int(test.wins))),
            tex_macro(f"{stem}Ties", str(int(test.ties))),
            tex_macro(f"{stem}Losses", str(int(test.losses))),
            tex_macro(f"{stem}NSeeds", str(int(test.n_seeds))),
        ]
    )
    # Short aliases are useful in narrow IEEE tables.
    chunks.extend(
        [
            tex_macro(f"{stem}Diff", f"\\{stem}Difference"),
            tex_macro(f"{stem}DiffCILo", f"\\{stem}DifferenceCILo"),
            tex_macro(f"{stem}DiffCIHi", f"\\{stem}DifferenceCIHi"),
        ]
    )


def append_support_macros(chunks: list[str], prefix: str, root: Path) -> None:
    environment = json.loads((root / "environment.json").read_text(encoding="utf-8"))
    audit = environment.get("config", {}).get("window_support_rule", {})
    fields = [
        ("FullWindows", "full_windows", 0),
        ("SupportedWindows", "supported_windows", 0),
        ("MinimumWindowTransactions", "minimum_supported_transactions", 0),
        ("FullTransactions", "full_transactions", 0),
        ("RetainedTransactions", "retained_transactions", 0),
        ("TrimmedTransactions", "trimmed_transactions", 0),
        ("RetainedFraction", "retained_fraction", 3),
    ]
    for macro, key, digits in fields:
        value = audit.get(key, np.nan)
        if digits == 0 and np.isfinite(value):
            rendered = f"{int(value):,}"
        else:
            rendered = fmt(value, digits)
        chunks.append(tex_macro(f"{prefix}{macro}", rendered))
    support = pd.read_csv(seed_dirs(root, "control")[0] / "window_support.csv")
    prevalence = support["positives"].sum() / support["transactions"].sum()
    chunks.append(tex_macro(f"{prefix}StreamPrevalence", pct(prevalence, 3)))


def append_mechanism_macros(chunks: list[str], prefix: str, root: Path) -> None:
    path = root / "mechanism_tests.csv"
    if not path.exists():
        return
    tests = pd.read_csv(path).set_index("horizon_windows")
    for horizon, horizon_macro in [(1, "One"), (3, "Three"), (-1, "All")]:
        if horizon not in tests.index:
            continue
        row = tests.loc[horizon]
        stem = f"{prefix}Inoc{horizon_macro}"
        chunks.extend(
            [
                tex_macro(f"{stem}Difference", fmt(row.mean_difference)),
                tex_macro(f"{stem}DifferenceCILo", fmt(row.difference_ci_lo)),
                tex_macro(f"{stem}DifferenceCIHi", fmt(row.difference_ci_hi)),
                tex_macro(f"{stem}RankBiserial", fmt(row.rank_biserial, 3)),
                tex_macro(f"{stem}PRaw", sci(row.p_raw)),
                tex_macro(f"{stem}PHolm", sci(row.p_holm)),
                tex_macro(f"{stem}Wins", str(int(row.wins))),
                tex_macro(f"{stem}Ties", str(int(row.ties))),
                tex_macro(f"{stem}Losses", str(int(row.losses))),
            ]
        )


def append_root_macros(
    chunks: list[str],
    root: Path,
    prefix: str,
    *,
    include_ablations: bool,
    include_diagnostics: bool,
    table_methods: list[str] | tuple[str, ...] = METHODS,
) -> dict[str, pd.DataFrame]:
    summary = pd.read_csv(root / "method_summary.csv")
    tests = pd.read_csv(root / "seed_level_tests.csv")
    ablation_tests = (
        pd.read_csv(root / "ablation_tests.csv") if include_ablations else pd.DataFrame()
    )
    for condition, suffix in [("control", "Control"), ("drift", "Drift")]:
        block = summary[summary.condition == condition]
        chunks.append(table_macro(f"{prefix}{suffix}Rows", block, table_methods))
        append_metric_macros(chunks, prefix, suffix, block)
        condition_tests = tests[tests.condition == condition].set_index("comparison")
        for comparison in METHODS:
            if comparison == "fedtypo" or comparison not in condition_tests.index:
                continue
            append_test_macros(
                chunks,
                f"{prefix}{suffix}{METHOD_MACROS[comparison]}",
                condition_tests.loc[comparison],
            )
        if include_ablations:
            condition_ablation_tests = ablation_tests[
                ablation_tests.condition == condition
            ]
            chunks.append(
                ablation_table_macro(
                    f"{prefix}{suffix}AblationRows", block, condition_ablation_tests
                )
            )
            indexed_ablation_tests = condition_ablation_tests.set_index("comparison")
            for comparison in ABLATION_METHODS:
                append_test_macros(
                    chunks,
                    f"{prefix}{suffix}{METHOD_MACROS[comparison]}",
                    indexed_ablation_tests.loc[comparison],
                )

    if not include_diagnostics:
        return {"tests": tests, "ablation_tests": ablation_tests}

    append_support_macros(chunks, prefix, root)
    js_mean, js_std = typology_js(root, "control")
    chunks.append(
        tex_macro(f"{prefix}TypologyJS", f"{fmt(js_mean, 3)}$\\pm${fmt(js_std, 3)}")
    )
    for condition, suffix in [("control", "Control"), ("drift", "Drift")]:
        reg_mean, reg_std = registry_count(root, condition)
        chunks.append(
            tex_macro(
                f"{prefix}{suffix}RegistryCount",
                f"{fmt(reg_mean, 1)}$\\pm${fmt(reg_std, 1)}",
            )
        )
        proto = final_prototype_rows(root, condition)
        for metric, metric_macro in [
            ("purity", "Purity"),
            ("nmi", "NMI"),
            ("ari", "ARI"),
            ("named_purity", "NamedPurity"),
            ("named_nmi", "NamedNMI"),
            ("named_ari", "NamedARI"),
            ("cosine_gap", "CosineGap"),
        ]:
            chunks.append(
                tex_macro(
                    f"{prefix}{suffix}{metric_macro}", fmt(proto[metric].mean(), 3)
                )
            )
    fedtypo_mean, _ = inoculation(root, "fedtypo", horizon=-1)
    noreg_mean, _ = inoculation(root, "fedtypo_noreg", horizon=-1)
    chunks.append(tex_macro(f"{prefix}InocFedTypo", fmt(fedtypo_mean, 3)))
    chunks.append(tex_macro(f"{prefix}InocNoReg", fmt(noreg_mean, 3)))
    gain = fedtypo_mean / noreg_mean - 1 if noreg_mean > 0 else np.nan
    chunks.append(tex_macro(f"{prefix}InocGain", pct(gain)))
    append_mechanism_macros(chunks, prefix, root)
    return {"tests": tests, "ablation_tests": ablation_tests}


def write_results_tex(
    roots: dict[str, Path],
    path: Path,
    secondary_roots: dict[str, Path] | None = None,
    registry_beta_root: Path | None = None,
) -> None:
    chunks = [
        "% Auto-generated by make_submission_assets.py; do not edit by hand.\n"
    ]
    drift_tests = {}
    for root, prefix in [(roots["ibm"], "IBM"), (roots["samld"], "SAML")]:
        environment = json.loads((root / "environment.json").read_text(encoding="utf-8"))
        partition = environment.get("config", {}).get("partition_mode", "unknown")
        partition_label = {
            "account_random": "account-random (label-independent)",
            "typology_skew": "typology-skew",
        }.get(partition, str(partition).replace("_", "-"))
        chunks.append(tex_macro(f"{prefix}PrimaryPartition", partition_label))
        tables = append_root_macros(
            chunks,
            root,
            prefix,
            include_ablations=True,
            include_diagnostics=True,
        )
        drift_tests[prefix] = tables["tests"][
            tables["tests"].condition == "drift"
        ].set_index("comparison")

    if secondary_roots:
        for dataset, prefix in [("ibm", "IBM"), ("samld", "SAML")]:
            root = secondary_roots.get(dataset)
            if root is None:
                continue
            environment = json.loads(
                (root / "environment.json").read_text(encoding="utf-8")
            )
            partition = environment.get("config", {}).get("partition_mode", "unknown")
            partition_label = {
                "account_random": "account-random (label-independent)",
                "typology_skew": "typology-skew",
            }.get(partition, str(partition).replace("_", "-"))
            chunks.append(tex_macro(f"{prefix}SecondaryPartition", partition_label))
            append_root_macros(
                chunks,
                root,
                f"{prefix}Secondary",
                include_ablations=False,
                include_diagnostics=False,
                table_methods=SECONDARY_METHODS,
            )

    test_rows = []
    effect_rows = []
    for baseline in [
        "local_only",
        "fedavg",
        "fedprox",
        "fedproto",
        "cda_fedavg",
        "fedtypo_noreg",
    ]:
        ibm = drift_tests["IBM"].loc[baseline]
        saml = drift_tests["SAML"].loc[baseline]
        test_rows.append(
            "{} & {} & ${}$ & {} & ${}$ \\\\".format(
                LABELS[baseline],
                pct(ibm.relative_gain),
                sci(ibm.p_holm),
                pct(saml.relative_gain),
                sci(saml.p_holm),
            )
        )
        effect_rows.append(
            "{} & {} [{}, {}] & {} & ${}$ & {} [{}, {}] & {} & ${}$ \\\\".format(
                LABELS[baseline],
                fmt(ibm.mean_difference),
                fmt(ibm.difference_ci_lo),
                fmt(ibm.difference_ci_hi),
                fmt(ibm.rank_biserial, 3),
                sci(ibm.p_holm),
                fmt(saml.mean_difference),
                fmt(saml.difference_ci_lo),
                fmt(saml.difference_ci_hi),
                fmt(saml.rank_biserial, 3),
                sci(saml.p_holm),
            )
        )
    chunks.append(
        "\\newcommand{\\DriftTestRows}{%%\n%s\n}\n" % "\n".join(test_rows)
    )
    chunks.append(
        "\\newcommand{\\DriftEffectRows}{%%\n%s\n}\n" % "\n".join(effect_rows)
    )
    chunks.append(tex_macro("DriftTwoSidedRows", "\\DriftEffectRows"))
    if registry_beta_root is not None:
        append_registry_beta_macros(chunks, registry_beta_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(chunks), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ibm", type=Path)
    parser.add_argument("--samld", type=Path)
    parser.add_argument(
        "--ibm-secondary",
        "--ibm-label-free",
        dest="ibm_secondary",
        type=Path,
        help=(
            "optional IBM partition-sensitivity root; --ibm-label-free is a "
            "backward-compatible alias"
        ),
    )
    parser.add_argument(
        "--samld-secondary",
        "--samld-label-free",
        dest="samld_secondary",
        type=Path,
        help=(
            "optional SAML-D partition-sensitivity root; --samld-label-free is a "
            "backward-compatible alias"
        ),
    )
    parser.add_argument(
        "--samld-registry-beta",
        type=Path,
        help="SAML-D registry-score beta-sensitivity result root",
    )
    parser.add_argument("--figdir", type=Path, required=True)
    parser.add_argument("--tex", type=Path)
    parser.add_argument("--architecture-only", action="store_true")
    args = parser.parse_args()
    args.figdir.mkdir(parents=True, exist_ok=True)
    if args.architecture_only:
        build_architecture(args.figdir / "architecture.pdf")
        print(f"Wrote {args.figdir / 'architecture.pdf'}")
        return
    if (
        args.ibm is None
        or args.samld is None
        or args.samld_registry_beta is None
        or args.tex is None
    ):
        parser.error(
            "--ibm, --samld, --samld-registry-beta, and --tex are required "
            "for result assets"
        )
    roots = {"ibm": args.ibm, "samld": args.samld}
    for root in roots.values():
        validate_root(root)
    secondary_roots = {
        dataset: root
        for dataset, root in [
            ("ibm", args.ibm_secondary),
            ("samld", args.samld_secondary),
        ]
        if root is not None
    }
    for dataset, root in secondary_roots.items():
        validate_root(
            root, require_ablations=False, primary_methods=SECONDARY_METHODS
        )
        primary_environment = json.loads(
            (roots[dataset] / "environment.json").read_text(encoding="utf-8")
        )
        secondary_environment = json.loads(
            (root / "environment.json").read_text(encoding="utf-8")
        )
        primary_partition = primary_environment.get("config", {}).get("partition_mode")
        secondary_partition = secondary_environment.get("config", {}).get(
            "partition_mode"
        )
        if primary_partition == secondary_partition:
            raise RuntimeError(
                f"{dataset}: primary and secondary roots both use "
                f"partition_mode={primary_partition!r}"
            )
    build_architecture(args.figdir / "architecture.pdf")
    build_results_figure(roots, args.figdir / "results_main.pdf")
    build_temporal_support_figure(roots, args.figdir / "temporal_support.pdf")
    build_ablation_figure(roots, args.figdir / "ablations.pdf")
    build_budget_figure(roots, args.figdir / "budget_sensitivity.pdf")
    build_prototype_figure(roots, args.figdir / "prototype_fidelity.pdf")
    write_results_tex(
        roots,
        args.tex,
        secondary_roots=secondary_roots,
        registry_beta_root=args.samld_registry_beta,
    )
    print(f"Wrote figures to {args.figdir} and macros to {args.tex}")


if __name__ == "__main__":
    main()
