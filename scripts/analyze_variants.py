from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/extended_results.csv")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    results_path = Path(args.results)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_path)
    df["accuracy"] = df["accuracy"].astype(float)
    df["severity"] = df["severity"].astype(int)
    df["mean_entropy"] = df["mean_entropy"].astype(float)
    df["update_ratio"] = df["update_ratio"].astype(float)

    keys = ["corruption", "severity"]
    source = (
        df[df["method"] == "source_only"][keys + ["accuracy"]]
        .rename(columns={"accuracy": "source_accuracy"})
    )
    tent = (
        df[df["method"] == "tent"][keys + ["accuracy"]]
        .rename(columns={"accuracy": "tent_accuracy"})
    )
    merged = df.merge(source, on=keys, how="left").merge(tent, on=keys, how="left")
    merged["delta_vs_source"] = merged["accuracy"] - merged["source_accuracy"]
    merged["delta_vs_tent"] = merged["accuracy"] - merged["tent_accuracy"]
    merged["negative_transfer_vs_source"] = merged["delta_vs_source"] < 0
    merged["worse_than_tent"] = merged["delta_vs_tent"] < 0

    summary = (
        merged.groupby("method")
        .agg(
            mean_accuracy=("accuracy", "mean"),
            mean_entropy=("mean_entropy", "mean"),
            mean_update_ratio=("update_ratio", "mean"),
            mean_delta_vs_source=("delta_vs_source", "mean"),
            mean_delta_vs_tent=("delta_vs_tent", "mean"),
            negative_transfer_cases=("negative_transfer_vs_source", "sum"),
            worse_than_tent_cases=("worse_than_tent", "sum"),
        )
        .reset_index()
        .sort_values(
            ["negative_transfer_cases", "mean_accuracy"],
            ascending=[True, False],
        )
    )

    by_severity = (
        merged.groupby(["method", "severity"])
        .agg(
            mean_accuracy=("accuracy", "mean"),
            mean_delta_vs_source=("delta_vs_source", "mean"),
            mean_delta_vs_tent=("delta_vs_tent", "mean"),
            mean_update_ratio=("update_ratio", "mean"),
        )
        .reset_index()
        .sort_values(["method", "severity"])
    )

    failures = (
        merged[
            (merged["method"] != "source_only")
            & (merged["negative_transfer_vs_source"])
        ][
            [
                "method",
                "corruption",
                "severity",
                "accuracy",
                "source_accuracy",
                "delta_vs_source",
                "update_ratio",
            ]
        ]
        .sort_values(["delta_vs_source", "method"])
    )

    summary_path = out_dir / "variant_summary.csv"
    severity_path = out_dir / "variant_by_severity.csv"
    failures_path = out_dir / "negative_transfer_cases.csv"

    summary.to_csv(summary_path, index=False)
    by_severity.to_csv(severity_path, index=False)
    failures.to_csv(failures_path, index=False)

    print("\n=== Method summary ===")
    print(summary.to_string(index=False))
    print(f"\nwrote {summary_path}")
    print(f"wrote {severity_path}")
    print(f"wrote {failures_path}")


if __name__ == "__main__":
    main()

