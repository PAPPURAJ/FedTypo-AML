#!/usr/bin/env python3
"""Build the figures and LaTeX result macros for the TIFS submission.

The script reads only the consolidated outputs produced by
experiments/run_submission.py.  It deliberately performs inference across
independent seeds (not pooled client-windows) and leaves all reported digits
traceable to CSV files.
"""

from __future__ import annotations

import argparse
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
EVAL_START_W = 1
LABELS = {
    "local_only": "Local",
    "fedavg": "FedAvg",
    "fedprox": "FedProx",
    "fedproto": "FedProto",
    "cda_fedavg": "CDA-FedAvg",
    "fedtypo_noreg": "FedTypo-NoReg",
    "fedtypo": "FedTypo",
}
COLORS = {
    "local_only": "#666666",
    "fedavg": "#2878B5",
    "fedprox": "#55A868",
    "fedproto": "#E17C05",
    "cda_fedavg": "#CCB974",
    "fedtypo_noreg": "#C44E52",
    "fedtypo": "#7A3E9D",
}
MARKERS = {
    "local_only": "o",
    "fedavg": "s",
    "fedprox": "^",
    "fedproto": "X",
    "cda_fedavg": "D",
    "fedtypo_noreg": "v",
    "fedtypo": "P",
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


def validate_root(root: Path) -> None:
    """Fail before asset generation if a consolidated run is incomplete."""
    expected_seeds = set(range(42, 52))
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
            for method in METHODS:
                required = [
                    directory / f"DONE_{method}",
                    directory / f"client_metrics_{method}.csv",
                    directory / f"window_metrics_{method}.csv",
                    directory / f"budget_metrics_{method}.csv",
                ]
                missing = [str(path) for path in required if not path.exists()]
                if missing:
                    raise RuntimeError("Missing result files:\n" + "\n".join(missing))
            if condition == "drift":
                events = pd.read_csv(directory / "drift_events.csv")
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
            for method in METHODS:
                windows = pd.read_csv(directory / f"window_metrics_{method}.csv")
                if not (windows["window"] >= EVAL_START_W).any():
                    raise RuntimeError(
                        f"{directory}/{method}: no evaluated window after warm-up"
                    )
            proto = pd.read_csv(directory / "prototype_fedtypo.csv")
            proto_values = proto[["purity", "nmi", "ari"]].to_numpy(dtype=float)
            if len(proto_values) == 0 or not np.isfinite(proto_values).all():
                raise RuntimeError(
                    f"{directory}: missing or non-finite prototype-fidelity values"
                )
    seed_summary = pd.read_csv(root / "seed_summary.csv")
    if len(seed_summary) != 2 * len(expected_seeds) * len(METHODS):
        raise RuntimeError(f"{root}: incomplete seed_summary.csv")
    numeric = seed_summary[["auprc", "p_at_budget"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise RuntimeError(f"{root}: non-finite primary metrics")


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
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.0), sharex=True)
    keep = ["local_only", "fedavg", "fedtypo_noreg", "fedtypo"]
    for ax, (dataset, root, condition) in zip(axes.flat, panels):
        for method in keep:
            frame = load_window_seed_means(root, condition, method)
            if frame.empty:
                continue
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
            for onset in (6, 9, 10, 12):
                ax.axvline(onset, color="#BBBBBB", linewidth=0.55, linestyle="--")
        ax.set_title(f"{dataset}: {condition}", fontsize=9)
        ax.grid(True, linewidth=0.35, alpha=0.45)
        ax.tick_params(labelsize=7.5)
        ax.set_xlim(0, 19)
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
    fig, ax = plt.subplots(figsize=(7.15, 2.6))
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
    ax.set_ylabel("Agreement with annotated typology", fontsize=8)
    ax.set_ylim(-0.2, 1.0)
    ax.grid(True, axis="y", linewidth=0.35, alpha=0.45)
    ax.tick_params(labelsize=7.5)
    ax.legend(ncol=3, loc="upper center", fontsize=7.5, frameon=False)
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


def inoculation(root: Path, method: str) -> tuple[float, float]:
    values = []
    for directory in seed_dirs(root, "drift"):
        path = directory / f"inoculation_{method}.csv"
        if path.exists():
            values.append(pd.read_csv(path)["capture_rate"].mean())
    return float(np.nanmean(values)), float(np.nanstd(values, ddof=1))


def table_macro(name: str, frame: pd.DataFrame) -> str:
    by_method = frame.set_index("method")
    rows = []
    for method in METHODS:
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


def write_results_tex(roots: dict[str, Path], path: Path) -> None:
    chunks = [
        "% Auto-generated by make_submission_assets.py; do not edit by hand.\n"
    ]
    drift_tests = {}
    for dataset, root, prefix in [
        ("ibm", roots["ibm"], "IBM"),
        ("samld", roots["samld"], "SAML"),
    ]:
        summary = pd.read_csv(root / "method_summary.csv")
        tests = pd.read_csv(root / "seed_level_tests.csv")
        drift_tests[prefix] = tests[tests.condition == "drift"].set_index("baseline")
        seeds = pd.read_csv(root / "seed_summary.csv")
        for condition, suffix in [("control", "Control"), ("drift", "Drift")]:
            block = summary[summary.condition == condition]
            chunks.append(table_macro(f"{prefix}{suffix}Rows", block))
            for method, method_macro in [
                ("local_only", "Local"),
                ("fedavg", "FedAvg"),
                ("fedprox", "FedProx"),
                ("fedproto", "FedProto"),
                ("cda_fedavg", "CDA"),
                ("fedtypo_noreg", "NoReg"),
                ("fedtypo", "FedTypo"),
            ]:
                result = block[block.method == method].iloc[0]
                chunks.append(
                    f"\\newcommand{{\\{prefix}{suffix}{method_macro}AUPRC}}"
                    f"{{{fmt(result.auprc_mean)}}}\n"
                )
                chunks.append(
                    f"\\newcommand{{\\{prefix}{suffix}{method_macro}PatFifty}}"
                    f"{{{fmt(result.p_at_budget_mean)}}}\n"
                )
            ours = seeds[
                (seeds.condition == condition) & (seeds.method == "fedtypo")
            ].auprc.mean()
            for baseline, baseline_macro in [
                ("fedavg", "FedAvg"),
                ("fedtypo_noreg", "NoReg"),
                ("local_only", "Local"),
            ]:
                base = seeds[
                    (seeds.condition == condition) & (seeds.method == baseline)
                ].auprc.mean()
                test = tests[
                    (tests.condition == condition) & (tests.baseline == baseline)
                ].iloc[0]
                stem = f"{prefix}{suffix}{baseline_macro}"
                chunks.append(f"\\newcommand{{\\{stem}Gain}}{{{pct(ours/base-1)}}}\n")
                chunks.append(
                    f"\\newcommand{{\\{stem}PHolm}}{{{sci(test.p_holm)}}}\n"
                )
        js_mean, js_std = typology_js(root, "control")
        chunks.append(
            f"\\newcommand{{\\{prefix}TypologyJS}}{{{fmt(js_mean, 3)}"
            f"$\\pm${fmt(js_std, 3)}}}\n"
        )
        for condition, suffix in [("control", "Control"), ("drift", "Drift")]:
            reg_mean, reg_std = registry_count(root, condition)
            chunks.append(
                f"\\newcommand{{\\{prefix}{suffix}RegistryCount}}"
                f"{{{fmt(reg_mean, 1)}$\\pm${fmt(reg_std, 1)}}}\n"
            )
            proto = final_prototype_rows(root, condition)
            for metric, metric_macro in [
                ("purity", "Purity"),
                ("nmi", "NMI"),
                ("ari", "ARI"),
            ]:
                chunks.append(
                    f"\\newcommand{{\\{prefix}{suffix}{metric_macro}}}"
                    f"{{{fmt(proto[metric].mean(), 3)}}}\n"
                )
        fedtypo_mean, _ = inoculation(root, "fedtypo")
        noreg_mean, _ = inoculation(root, "fedtypo_noreg")
        chunks.append(
            f"\\newcommand{{\\{prefix}InocFedTypo}}{{{fmt(fedtypo_mean, 3)}}}\n"
        )
        chunks.append(
            f"\\newcommand{{\\{prefix}InocNoReg}}{{{fmt(noreg_mean, 3)}}}\n"
        )
        gain = fedtypo_mean / noreg_mean - 1 if noreg_mean > 0 else np.nan
        chunks.append(
            f"\\newcommand{{\\{prefix}InocGain}}{{{pct(gain)}}}\n"
        )
    test_rows = []
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
    chunks.append(
        "\\newcommand{\\DriftTestRows}{%%\n%s\n}\n" % "\n".join(test_rows)
    )
    path.write_text("".join(chunks), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ibm", type=Path)
    parser.add_argument("--samld", type=Path)
    parser.add_argument("--figdir", type=Path, required=True)
    parser.add_argument("--tex", type=Path)
    parser.add_argument("--architecture-only", action="store_true")
    args = parser.parse_args()
    args.figdir.mkdir(parents=True, exist_ok=True)
    if args.architecture_only:
        build_architecture(args.figdir / "architecture.pdf")
        print(f"Wrote {args.figdir / 'architecture.pdf'}")
        return
    if args.ibm is None or args.samld is None or args.tex is None:
        parser.error("--ibm, --samld, and --tex are required for result assets")
    roots = {"ibm": args.ibm, "samld": args.samld}
    for root in roots.values():
        validate_root(root)
    build_architecture(args.figdir / "architecture.pdf")
    build_results_figure(roots, args.figdir / "results_main.pdf")
    build_budget_figure(roots, args.figdir / "budget_sensitivity.pdf")
    build_prototype_figure(roots, args.figdir / "prototype_fidelity.pdf")
    write_results_tex(roots, args.tex)
    print(f"Wrote figures to {args.figdir} and macros to {args.tex}")


if __name__ == "__main__":
    main()
