from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/first_pass_results.csv")
    parser.add_argument("--out-dir", default="results/figures")
    args = parser.parse_args()

    results_path = Path(args.results)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_path)
    print("\nMean accuracy by method")
    print(df.groupby("method")["accuracy"].mean().sort_values(ascending=False))

    pivot = df.pivot_table(
        index="severity",
        columns="method",
        values="accuracy",
        aggfunc="mean",
    )
    print("\nAccuracy by severity")
    print(pivot)
    pivot.to_csv(out_dir.parent / "accuracy_by_severity.csv")

    ax = pivot.plot(marker="o", figsize=(7, 4))
    ax.set_title("Mean CIFAR-10-C Accuracy by Severity")
    ax.set_xlabel("Corruption Severity")
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_by_severity.png", dpi=180)

    ug = df[df["method"] == "ug_tent"]
    if not ug.empty:
        pivot_update = ug.pivot_table(
            index="severity",
            values="update_ratio",
            aggfunc="mean",
        )
        ax = pivot_update.plot(marker="o", legend=False, figsize=(7, 4))
        ax.set_title("UG-Tent Update Ratio by Severity")
        ax.set_xlabel("Corruption Severity")
        ax.set_ylabel("Selected Fraction")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "ug_tent_update_ratio.png", dpi=180)

    print(f"\nwrote figures to {out_dir}")


if __name__ == "__main__":
    main()

