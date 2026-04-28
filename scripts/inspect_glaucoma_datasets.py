#!/usr/bin/env python3
"""Inspect the local glaucoma datasets and their shared class mapping."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retfound_eval.config import CLASS_DESCRIPTIONS, GLAUCOMA_DATASETS, SPLITS


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def count_images(path: Path) -> int:
    return sum(
        1
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        and not any(part.startswith(".") for part in item.relative_to(path).parts)
    )


def main() -> None:
    rows = []
    for dataset_name, cfg in GLAUCOMA_DATASETS.items():
        root = cfg.root
        for split in SPLITS:
            for folder, label in cfg.folder_to_label.items():
                folder_path = root / split / folder
                rows.append(
                    {
                        "dataset": dataset_name,
                        "split": split,
                        "folder": folder,
                        "label": label,
                        "description": CLASS_DESCRIPTIONS[label],
                        "n_images": count_images(folder_path) if folder_path.exists() else 0,
                    }
                )

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("\nTotaux par dataset:")
    print(df.groupby("dataset")["n_images"].sum().to_string())


if __name__ == "__main__":
    main()
