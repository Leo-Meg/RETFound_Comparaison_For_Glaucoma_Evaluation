#!/usr/bin/env python3
"""Download glaucoma datasets and checkpoints into the local project."""

from __future__ import annotations

import argparse
import html
import http.cookiejar
import re
import shutil
import tempfile
import urllib.parse
import urllib.request
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


def _is_download_response(response: urllib.request.addinfourl) -> bool:
    content_disposition = response.headers.get("Content-Disposition", "")
    content_type = response.headers.get("Content-Type", "")
    if "attachment" in content_disposition.lower():
        return True
    return "text/html" not in content_type.lower()


def _write_response_to_file(
    response: urllib.request.addinfourl,
    destination: Path,
    chunk_size: int = 1024 * 1024,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            handle.write(chunk)


def _extract_confirm_from_cookies(cookie_jar: http.cookiejar.CookieJar) -> str | None:
    for cookie in cookie_jar:
        if cookie.name.startswith("download_warning"):
            return cookie.value
    return None


def _extract_confirm_form(html_text: str) -> tuple[str, dict[str, str]] | None:
    form_match = re.search(
        r'<form[^>]+id="download-form"[^>]+action="([^"]+)"[^>]*>(.*?)</form>',
        html_text,
        flags=re.DOTALL,
    )
    if not form_match:
        return None

    action = html.unescape(form_match.group(1))
    form_body = form_match.group(2)
    inputs: dict[str, str] = {}
    for input_match in re.finditer(r"<input\b[^>]*>", form_body):
        input_tag = input_match.group(0)
        name_match = re.search(r'name="([^"]+)"', input_tag)
        if not name_match:
            continue
        value_match = re.search(r'value="([^"]*)"', input_tag)
        inputs[name_match.group(1)] = html.unescape(value_match.group(1) if value_match else "")

    if not inputs:
        return None
    return action, inputs


def _download_with_google_drive_fallback(file_id: str, destination: Path) -> None:
    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    base_url = "https://drive.google.com/uc?export=download"
    initial_url = f"{base_url}&id={urllib.parse.quote(file_id)}"

    with opener.open(initial_url) as response:
        if _is_download_response(response):
            _write_response_to_file(response, destination)
            return
        html_text = response.read().decode("utf-8", errors="ignore")

    confirm_token = _extract_confirm_from_cookies(cookie_jar)
    if confirm_token:
        confirm_url = (
            f"{base_url}&id={urllib.parse.quote(file_id)}"
            f"&confirm={urllib.parse.quote(confirm_token)}"
        )
        with opener.open(confirm_url) as response:
            if _is_download_response(response):
                _write_response_to_file(response, destination)
                return

    form_data = _extract_confirm_form(html_text)
    if form_data is None:
        raise RuntimeError(
            "Impossible de recuperer le formulaire de confirmation Google Drive. "
            "Installez 'gdown' ou reessayez plus tard."
        )

    action, inputs = form_data
    inputs.setdefault("id", file_id)
    request_url = urllib.parse.urljoin("https://drive.google.com/", action)
    request_url = f"{request_url}?{urllib.parse.urlencode(inputs)}"
    with opener.open(request_url) as response:
        if not _is_download_response(response):
            raise RuntimeError(
                "Google Drive a renvoye une page HTML inattendue au lieu du fichier."
            )
        _write_response_to_file(response, destination)


def download_file(file_id: str, destination: Path) -> None:
    try:
        import gdown
    except ImportError:
        print(
            "[warn] Le paquet 'gdown' est absent. Utilisation du telechargement "
            "Google Drive de secours base sur la bibliotheque standard."
        )
        _download_with_google_drive_fallback(file_id, destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://drive.google.com/uc?id={file_id}"
        try:
            gdown.download(url=url, output=str(destination), quiet=False, fuzzy=True)
        except TypeError as exc:
            if "fuzzy" not in str(exc):
                raise
            gdown.download(url=url, output=str(destination), quiet=False)
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
