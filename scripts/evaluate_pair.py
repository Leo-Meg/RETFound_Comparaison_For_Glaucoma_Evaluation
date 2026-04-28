#!/usr/bin/env python3
"""Evaluate one RETFound checkpoint on one glaucoma dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from retfound_eval.config import default_results_dir, dataset_names, resolve_splits
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
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Fusionne train+val+test pour l'evaluation.",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Utilise uniquement le split test pour l'evaluation.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--progress",
        choices=("plain", "bar", "none"),
        default="plain",
        help="Affichage de progression: texte simple, barre tqdm, ou aucun.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits, split_mode = resolve_splits(
        use_all=args.all,
        test_only=args.test_only,
        explicit_splits=args.splits,
    )
    external_only = args.train_dataset != args.eval_dataset
    result = run_evaluation(
        train_dataset=args.train_dataset,
        eval_dataset=args.eval_dataset,
        repo_root=args.repo_root,
        check_dir=args.check_dir,
        output_dir=args.output_dir or default_results_dir(external_only, split_mode),
        splits=splits,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_name=args.device,
        save_plots=not args.no_plots,
        split_mode=split_mode,
        external_only=external_only,
        progress=args.progress,
    )
    metrics = result["metrics"]
    print("\nEvaluation terminee")
    print(f"Sortie: {result['output_dir']}")
    print(f"Mode splits: {split_mode}")
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
