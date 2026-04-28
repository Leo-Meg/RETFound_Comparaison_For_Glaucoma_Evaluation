"""Configuration for the glaucoma datasets used in cross-dataset evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


NUM_CLASSES = 3
INPUT_SIZE = 224
SPLITS = ("train", "val", "test")

CLASS_NAMES = (
    "0 - Normal",
    "1 - Suspect ou precoce",
    "2 - Glaucome confirme ou avance",
)

CLASS_DESCRIPTIONS = (
    "Controle normal",
    "Atteinte suspecte ou glaucome precoce",
    "Glaucome confirme ou avance",
)


@dataclass(frozen=True)
class DatasetConfig:
    """Local dataset metadata."""

    name: str
    path: str
    folder_to_label: dict[str, int]
    checkpoint_name: str
    description: str

    @property
    def root(self) -> Path:
        return Path(self.path)


GLAUCOMA_DATASETS: dict[str, DatasetConfig] = {
    "Glaucoma_fundus": DatasetConfig(
        name="Glaucoma_fundus",
        path="dataset/Glaucoma_fundus",
        checkpoint_name="checkpoint-best-Glaucoma_fundus.pth",
        description="Controle normal, glaucome precoce, glaucome avance.",
        folder_to_label={
            "anormal_control": 0,
            "bearly_glaucoma": 1,
            "cadvanced_glaucoma": 2,
        },
    ),
    "PAPILA": DatasetConfig(
        name="PAPILA",
        path="dataset/PAPILA",
        checkpoint_name="checkpoint-best-PAPILA.pth",
        description="Normal, suspect glaucome, glaucome.",
        folder_to_label={
            "anormal": 0,
            "bsuspectglaucoma": 1,
            "cglaucoma": 2,
        },
    ),
}


def dataset_names() -> list[str]:
    return list(GLAUCOMA_DATASETS)


def get_dataset_config(name: str) -> DatasetConfig:
    try:
        return GLAUCOMA_DATASETS[name]
    except KeyError as exc:
        valid = ", ".join(dataset_names())
        raise KeyError(f"Dataset inconnu: {name}. Choix valides: {valid}") from exc


def checkpoint_path(check_dir: str | Path, dataset_name: str) -> Path:
    cfg = get_dataset_config(dataset_name)
    return Path(check_dir) / cfg.checkpoint_name
