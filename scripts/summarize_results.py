#!/usr/bin/env python3
"""Collect pair-level metrics.csv files into one summary table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume les resultats RETFound glaucome.")
    parser.add_argument("--results-dir", default="results/glaucoma_matrix")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    metric_files = sorted(results_dir.glob("train-*__eval-*/metrics.csv"))
    if not metric_files:
        raise SystemExit(f"Aucun fichier metrics.csv trouve dans {results_dir}")

    frames = [pd.read_csv(path) for path in metric_files]
    summary = pd.concat(frames, ignore_index=True)
    summary = summary.sort_values(["train_dataset", "eval_dataset"])

    output = Path(args.output) if args.output else results_dir / "summary_metrics.csv"
    summary.to_csv(output, index=False)
    print(f"Resume ecrit: {output}")

    columns = [
        "train_dataset",
        "eval_dataset",
        "n_images",
        "accuracy",
        "auroc_macro_ovr",
        "f1_macro",
        "cohen_kappa",
        "composite_f1_auc_kappa",
    ]
    print(summary[columns].to_string(index=False))


if __name__ == "__main__":
    main()
