#!/usr/bin/env python3
"""Populate the local project dataset/check folders from existing local assets."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
from pathlib import Path


DATASET_NAMES = ("Glaucoma_fundus", "PAPILA")
CHECKPOINT_NAMES = (
    "checkpoint-best-Glaucoma_fundus.pth",
    "checkpoint-best-PAPILA.pth",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare local glaucoma assets inside this project."
    )
    parser.add_argument("--source-dataset-dir", default="../dataset-glauc")
    parser.add_argument("--source-check-dir", default="../check-glauc")
    parser.add_argument("--target-dataset-dir", default="dataset")
    parser.add_argument("--target-check-dir", default="check")
    parser.add_argument(
        "--mode",
        choices=("auto", "copy", "symlink"),
        default="auto",
        help="copy recommande pour Windows; auto tente symlink puis fallback copy.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Supprime la cible existante avant recreation.",
    )
    return parser.parse_args()


def choose_mode(requested: str) -> str:
    if requested != "auto":
        return requested
    if platform.system().lower().startswith("win"):
        return "copy"
    return "symlink"


def remove_existing(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def link_or_copy(src: Path, dst: Path, mode: str, force: bool) -> str:
    if not src.exists():
        raise FileNotFoundError(f"Source introuvable: {src}")

    if dst.exists() or dst.is_symlink():
        if not force:
            return "skip"
        remove_existing(dst)

    dst.parent.mkdir(parents=True, exist_ok=True)

    if mode == "symlink":
        try:
            os.symlink(src.resolve(), dst, target_is_directory=src.is_dir())
            return "symlink"
        except OSError:
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
            return "copy-fallback"

    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    return "copy"


def main() -> None:
    args = parse_args()
    mode = choose_mode(args.mode)

    source_dataset_dir = Path(args.source_dataset_dir)
    source_check_dir = Path(args.source_check_dir)
    target_dataset_dir = Path(args.target_dataset_dir)
    target_check_dir = Path(args.target_check_dir)

    print(f"[mode] {mode}")

    for name in DATASET_NAMES:
        src = source_dataset_dir / name
        dst = target_dataset_dir / name
        action = link_or_copy(src, dst, mode=mode, force=args.force)
        print(f"[dataset] {name}: {action}")

    for name in CHECKPOINT_NAMES:
        src = source_check_dir / name
        dst = target_check_dir / name
        action = link_or_copy(src, dst, mode=mode, force=args.force)
        print(f"[checkpoint] {name}: {action}")

    print("[done] Assets locaux prets.")


if __name__ == "__main__":
    main()
