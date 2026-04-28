#!/usr/bin/env python3
"""Create a compact Markdown report from summary_metrics.csv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retfound_eval.config import default_results_dir, resolve_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genere un rapport Markdown glaucome.")
    parser.add_argument("--summary", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--external-only",
        action="store_true",
        help="Construit le rapport d'une campagne externe uniquement.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Rapport pour une evaluation sur train+val+test fusionnes.",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Rapport pour une evaluation sur le split test uniquement.",
    )
    return parser.parse_args()


def fmt(value) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.4f}"


def main() -> None:
    args = parse_args()
    _, split_mode = resolve_splits(use_all=args.all, test_only=args.test_only)
    results_dir = Path(
        default_results_dir(args.external_only, split_mode)
    )
    summary_path = Path(args.summary) if args.summary else results_dir / "summary_metrics.csv"
    output_path = Path(args.output) if args.output else results_dir / "report.md"
    df = pd.read_csv(summary_path)
    comparison_scope = (
        "comparaison externe uniquement"
        if args.external_only
        else "comparaison externe et interne"
    )
    split_label = (
        "train + val + test fusionnes"
        if split_mode == "all"
        else "split test uniquement"
        if split_mode == "test_only"
        else "splits personnalises"
    )

    lines = [
        "# Rapport d'evaluation RETFound glaucome",
        "",
        f"Portee: **{comparison_scope}**.",
        "",
        f"Jeu de donnees evalue: **{split_label}**.",
        "",
        "Chaque ligne correspond a un checkpoint fine-tune sur le dataset source, puis evalue sur le dataset cible.",
        "",
        "| Source checkpoint | Dataset cible | N images | Accuracy | AUROC | F1 macro | Kappa | Composite |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in df.sort_values(["train_dataset", "eval_dataset"]).iterrows():
        lines.append(
            "| "
            f"{row['train_dataset']} | "
            f"{row['eval_dataset']} | "
            f"{int(row['n_images'])} | "
            f"{fmt(row['accuracy'])} | "
            f"{fmt(row['auroc_macro_ovr'])} | "
            f"{fmt(row['f1_macro'])} | "
            f"{fmt(row['cohen_kappa'])} | "
            f"{fmt(row['composite_f1_auc_kappa'])} |"
        )

    external = df[df["train_dataset"] != df["eval_dataset"]].copy()
    if not external.empty:
        best = external.sort_values("composite_f1_auc_kappa", ascending=False).iloc[0]
        lines.extend(
            [
                "",
                "## Meilleure validation externe",
                "",
                (
                    f"- Source: `{best['train_dataset']}`; cible: `{best['eval_dataset']}`; "
                    f"composite: `{fmt(best['composite_f1_auc_kappa'])}`; "
                    f"AUROC: `{fmt(best['auroc_macro_ovr'])}`."
                ),
            ]
        )

    internal = df[df["train_dataset"] == df["eval_dataset"]].copy()
    if not args.external_only and not internal.empty:
        best_internal = internal.sort_values("composite_f1_auc_kappa", ascending=False).iloc[0]
        lines.extend(
            [
                "",
                "## Meilleure validation interne",
                "",
                (
                    f"- Source et cible: `{best_internal['train_dataset']}`; "
                    f"composite: `{fmt(best_internal['composite_f1_auc_kappa'])}`; "
                    f"AUROC: `{fmt(best_internal['auroc_macro_ovr'])}`."
                ),
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Rapport ecrit: {output_path}")


if __name__ == "__main__":
    main()
