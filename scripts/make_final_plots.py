from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METHOD_ORDER = [
    "source_only",
    "bn_adapt",
    "tent",
    "ug_tent",
    "eg_mix_batch",
    "eg_mix_sample",
    "disagree_aware_mix",
    "probe_commit_tent",
    "probe_commit_tent_loose",
    "source_mix_tent_075",
]

METHOD_LABELS = {
    "source_only": "Source-only",
    "bn_adapt": "BN Adapt",
    "tent": "Tent",
    "ug_tent": "UG-Tent",
    "eg_mix_batch": "EG Mix Batch",
    "eg_mix_sample": "EG Mix Sample",
    "disagree_aware_mix": "Disagree-Aware Mix",
    "probe_commit_tent": "Probe-Commit",
    "probe_commit_tent_loose": "Probe-Commit Loose",
    "source_mix_tent_075": "Source Mix 0.75",
}


def ordered_methods(df: pd.DataFrame) -> list[str]:
    present = set(df["method"].unique())
    ordered = [method for method in METHOD_ORDER if method in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def save_bar(
    values: pd.DataFrame,
    x_col: str,
    y_col: str,
    out_path: Path,
    title: str,
    ylabel: str,
    color: str = "#356f95",
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    labels = values[x_col].tolist()
    heights = values[y_col].tolist()
    ax.bar(labels, heights, color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(heights):
        ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_mean_accuracy(summary: pd.DataFrame, out_dir: Path) -> None:
    summary = summary.copy()
    summary["mean_accuracy_pct"] = summary["mean_accuracy"].astype(float) * 100
    methods = ordered_methods(summary)
    values = summary.set_index("method").loc[methods].reset_index()
    values["label"] = values["method"].map(METHOD_LABELS).fillna(values["method"])
    save_bar(
        values,
        "label",
        "mean_accuracy_pct",
        out_dir / "mean_accuracy_by_method.png",
        "Mean Accuracy Across 25 CIFAR-10-C Settings",
        "Mean Accuracy (%)",
    )


def plot_negative_transfer(summary: pd.DataFrame, out_dir: Path) -> None:
    summary = summary.copy()
    summary["negative_transfer_cases"] = summary["negative_transfer_cases"].astype(int)
    methods = ordered_methods(summary)
    values = summary.set_index("method").loc[methods].reset_index()
    values["label"] = values["method"].map(METHOD_LABELS).fillna(values["method"])
    save_bar(
        values,
        "label",
        "negative_transfer_cases",
        out_dir / "negative_transfer_cases_by_method.png",
        "Negative-Transfer Cases vs Source-only",
        "Count out of 25",
        color="#9b4a4a",
    )


def plot_accuracy_by_severity(by_severity: pd.DataFrame, out_dir: Path) -> None:
    by_severity = by_severity.copy()
    by_severity["severity"] = by_severity["severity"].astype(int)
    by_severity["mean_accuracy_pct"] = by_severity["mean_accuracy"].astype(float) * 100

    selected = [
        "source_only",
        "bn_adapt",
        "tent",
        "ug_tent",
        "eg_mix_batch",
        "source_mix_tent_075",
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    for method in selected:
        sub = by_severity[by_severity["method"] == method].sort_values("severity")
        if sub.empty:
            continue
        label = METHOD_LABELS.get(method, method)
        ax.plot(sub["severity"], sub["mean_accuracy_pct"], marker="o", linewidth=2, label=label)
    ax.set_title("Mean Accuracy By Corruption Severity")
    ax.set_xlabel("Severity")
    ax.set_ylabel("Mean Accuracy (%)")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_by_severity.png", dpi=180)
    plt.close(fig)


def plot_delta_vs_tent_by_corruption(results: pd.DataFrame, out_dir: Path) -> None:
    results = results.copy()
    results["accuracy"] = results["accuracy"].astype(float)
    tent = (
        results[results["method"] == "tent"]
        .groupby("corruption")["accuracy"]
        .mean()
        .rename("tent_accuracy")
    )
    method = "source_mix_tent_075"
    values = (
        results[results["method"] == method]
        .groupby("corruption")["accuracy"]
        .mean()
        .to_frame("method_accuracy")
        .join(tent)
        .reset_index()
    )
    values["delta_pp"] = (values["method_accuracy"] - values["tent_accuracy"]) * 100
    values = values.sort_values("delta_pp", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#317a50" if value >= 0 else "#9b4a4a" for value in values["delta_pp"]]
    ax.bar(values["corruption"], values["delta_pp"], color=colors)
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_title("Source Mix 0.75 Delta vs Tent By Corruption")
    ax.set_ylabel("Accuracy Delta (percentage points)")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(values["delta_pp"]):
        offset = 0.04 if value >= 0 else -0.16
        ax.text(idx, value + offset, f"{value:+.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "source_mix_delta_vs_tent_by_corruption.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/adaptive_results.csv")
    parser.add_argument("--summary", default="results/variant_summary.csv")
    parser.add_argument("--by-severity", default="results/variant_by_severity.csv")
    parser.add_argument("--out-dir", default="results/figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(args.results)
    summary = pd.read_csv(args.summary)
    by_severity = pd.read_csv(args.by_severity)

    plot_mean_accuracy(summary, out_dir)
    plot_negative_transfer(summary, out_dir)
    plot_accuracy_by_severity(by_severity, out_dir)
    plot_delta_vs_tent_by_corruption(results, out_dir)

    print(f"wrote figures to {out_dir}")


if __name__ == "__main__":
    main()

