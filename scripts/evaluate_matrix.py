#!/usr/bin/env python3
"""Run the full source-checkpoint x target-dataset evaluation matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from retfound_eval.config import default_results_dir, dataset_names, resolve_splits
from retfound_eval.evaluate import run_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Matrice d'evaluation RETFound sur Glaucoma_fundus et PAPILA."
    )
    parser.add_argument("--train-datasets", nargs="+", default=dataset_names())
    parser.add_argument("--eval-datasets", nargs="+", default=dataset_names())
    parser.add_argument(
        "--external-only",
        action="store_true",
        help="Ignore les diagonales train_dataset == eval_dataset.",
    )
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits, split_mode = resolve_splits(
        use_all=args.all,
        test_only=args.test_only,
        explicit_splits=args.splits,
    )
    output_dir = Path(
        args.output_dir or default_results_dir(args.external_only, split_mode)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = []

    for train_dataset in args.train_datasets:
        for eval_dataset in args.eval_datasets:
            if args.external_only and train_dataset == eval_dataset:
                continue

            print(f"\n=== {train_dataset} -> {eval_dataset} ===")
            result = run_evaluation(
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                repo_root=args.repo_root,
                check_dir=args.check_dir,
                output_dir=output_dir,
                splits=splits,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device_name=args.device,
                save_plots=not args.no_plots,
                split_mode=split_mode,
                external_only=args.external_only,
            )
            all_metrics.append(result["metrics"])

    summary = pd.DataFrame(all_metrics)
    summary_path = output_dir / "summary_metrics.csv"
    summary.to_csv(summary_path, index=False)

    print("\nMatrice terminee")
    print(f"Resume: {summary_path}")
    print(f"Mode splits: {split_mode}")
    print(
        "Portee: "
        + ("comparaison externe uniquement" if args.external_only else "comparaison interne + externe")
    )
    if not summary.empty:
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
