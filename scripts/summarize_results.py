#!/usr/bin/env python3
"""Collect pair-level metrics.csv files into one summary table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retfound_eval.config import default_results_dir, resolve_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume les resultats RETFound glaucome.")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument(
        "--external-only",
        action="store_true",
        help="Resume uniquement une campagne sans diagonales.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Utilise le dossier de resultats correspondant au mode train+val+test.",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Utilise le dossier de resultats correspondant au mode test uniquement.",
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, split_mode = resolve_splits(use_all=args.all, test_only=args.test_only)
    results_dir = Path(
        args.results_dir or default_results_dir(args.external_only, split_mode)
    )
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
