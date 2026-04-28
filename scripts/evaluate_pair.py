#!/usr/bin/env python3
"""Evaluate one RETFound checkpoint on one glaucoma dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retfound_eval.config import SPLITS, dataset_names
from retfound_eval.evaluate import run_evaluation


def fmt(value) -> str:
    return "nan" if value is None else f"{value:.4f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluation RETFound glaucome: checkpoint source -> dataset cible."
    )
    parser.add_argument("--train-dataset", required=True, choices=dataset_names())
    parser.add_argument("--eval-dataset", required=True, choices=dataset_names())
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--check-dir", default="check")
    parser.add_argument("--output-dir", default="results/glaucoma_matrix")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_evaluation(
        train_dataset=args.train_dataset,
        eval_dataset=args.eval_dataset,
        repo_root=args.repo_root,
        check_dir=args.check_dir,
        output_dir=args.output_dir,
        splits=tuple(args.splits),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_name=args.device,
        save_plots=not args.no_plots,
    )
    metrics = result["metrics"]
    print("\nEvaluation terminee")
    print(f"Sortie: {result['output_dir']}")
    print(
        "Metrics: "
        f"accuracy={fmt(metrics['accuracy'])}, "
        f"auroc={fmt(metrics['auroc_macro_ovr'])}, "
        f"f1={fmt(metrics['f1_macro'])}, "
        f"kappa={fmt(metrics['cohen_kappa'])}, "
        f"composite={fmt(metrics['composite_f1_auc_kappa'])}"
    )


if __name__ == "__main__":
    main()
