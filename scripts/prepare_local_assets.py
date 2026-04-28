#!/usr/bin/env python3
"""Download glaucoma datasets and checkpoints into the local project."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path


ASSETS = {
    "datasets": {
        "Glaucoma_fundus": {
            "file_id": "18vSazOYDsUGdZ64gGkTg3E6jiNtcrUrI",
            "kind": "zip",
        },
        "PAPILA": {
            "file_id": "1JltYs7WRWEU0yyki1CQw5-10HEbqCMBE",
            "kind": "zip",
        },
    },
    "checkpoints": {
        "Glaucoma_fundus": {
            "file_id": "1CvHRhXsN3IZ3xOQcfg4rd3KcbWCyyKeU",
            "filename": "checkpoint-best-Glaucoma_fundus.pth",
        },
        "PAPILA": {
            "file_id": "1CraCqBclTSCSNzn0jogyIqjBNYcep9rx",
            "filename": "checkpoint-best-PAPILA.pth",
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Telecharge les datasets et checkpoints glaucome dans ce projet."
    )
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--check-dir", default="check")
    parser.add_argument(
        "--skip-datasets",
        action="store_true",
        help="Ne telecharge pas les archives de datasets.",
    )
    parser.add_argument(
        "--skip-checkpoints",
        action="store_true",
        help="Ne telecharge pas les checkpoints.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Supprime la cible existante avant retelechargement.",
    )
    return parser.parse_args()


def remove_existing(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def download_file(file_id: str, destination: Path) -> None:
    try:
        import gdown
    except ImportError as exc:
        raise SystemExit(
            "Le paquet 'gdown' est requis. Installez d'abord les dependances avec "
            "'python3 -m pip install -r requirements.txt'."
        ) from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url=url, output=str(destination), quiet=False, fuzzy=True)
    if not destination.exists():
        raise FileNotFoundError(f"Le telechargement a echoue: {destination}")


def normalize_extracted_dir(extracted_root: Path, expected_name: str) -> Path:
    children = [path for path in extracted_root.iterdir() if path.name != "__MACOSX"]
    if len(children) == 1 and children[0].is_dir():
        return children[0]

    candidate = extracted_root / expected_name
    if candidate.exists() and candidate.is_dir():
        return candidate

    return extracted_root


def extract_dataset_archive(archive_path: Path, dataset_dir: Path, force: bool) -> None:
    if dataset_dir.exists():
        if not force:
            print(f"[dataset] {dataset_dir.name}: deja present, skip")
            return
        remove_existing(dataset_dir)

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        tmp_dir = Path(tmp_dir_name)
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(tmp_dir)

        extracted_source = normalize_extracted_dir(tmp_dir, dataset_dir.name)
        shutil.copytree(extracted_source, dataset_dir)


def download_dataset(name: str, dataset_dir: Path, file_id: str, force: bool) -> None:
    target_dir = dataset_dir / name
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        archive_path = Path(tmp_dir_name) / f"{name}.zip"
        print(f"[dataset] {name}: telechargement")
        download_file(file_id, archive_path)
        print(f"[dataset] {name}: extraction")
        extract_dataset_archive(archive_path, target_dir, force=force)
        print(f"[dataset] {name}: pret -> {target_dir}")


def download_checkpoint(
    dataset_name: str,
    check_dir: Path,
    file_id: str,
    filename: str,
    force: bool,
) -> None:
    target_path = check_dir / filename
    if target_path.exists():
        if not force:
            print(f"[checkpoint] {dataset_name}: deja present, skip")
            return
        remove_existing(target_path)

    with tempfile.TemporaryDirectory() as tmp_dir_name:
        tmp_file = Path(tmp_dir_name) / "checkpoint-best.pth"
        print(f"[checkpoint] {dataset_name}: telechargement")
        download_file(file_id, tmp_file)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp_file), str(target_path))
        print(f"[checkpoint] {dataset_name}: renomme -> {target_path.name}")


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    check_dir = Path(args.check_dir)

    if not args.skip_datasets:
        for name, asset in ASSETS["datasets"].items():
            download_dataset(
                name=name,
                dataset_dir=dataset_dir,
                file_id=asset["file_id"],
                force=args.force,
            )

    if not args.skip_checkpoints:
        for dataset_name, asset in ASSETS["checkpoints"].items():
            download_checkpoint(
                dataset_name=dataset_name,
                check_dir=check_dir,
                file_id=asset["file_id"],
                filename=asset["filename"],
                force=args.force,
            )

    print("[done] Assets telecharges et prets.")


if __name__ == "__main__":
    main()
