"""Dataset and transform utilities for glaucoma grading datasets."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .config import INPUT_SIZE, SPLITS, DatasetConfig


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def eval_transform(input_size: int = INPUT_SIZE):
    """Preprocessing used for deterministic RETFound evaluation."""

    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def iter_images(folder: Path) -> Iterable[Path]:
    for path in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


class GlaucomaGradingDataset(Dataset):
    """Dataset spanning one or more train/val/test splits."""

    def __init__(
        self,
        config: DatasetConfig,
        splits: Iterable[str] = SPLITS,
        transform=None,
        repo_root: str | Path = ".",
    ):
        self.config = config
        self.root = Path(repo_root) / config.root
        self.splits = tuple(splits)
        self.transform = transform
        self.samples: list[tuple[Path, int, str]] = []

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset introuvable: {self.root}")

        for split in self.splits:
            for folder_name, label in config.folder_to_label.items():
                class_dir = self.root / split / folder_name
                if not class_dir.exists():
                    continue
                for image_path in iter_images(class_dir):
                    self.samples.append((image_path, label, split))

        if not self.samples:
            raise RuntimeError(
                f"Aucune image trouvee pour {config.name} dans {self.root} "
                f"avec les splits {self.splits}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label, split = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label, str(image_path), split

    def class_counts(self) -> dict[int, int]:
        return dict(Counter(label for _, label, _ in self.samples))

    def split_counts(self) -> dict[str, int]:
        return dict(Counter(split for _, _, split in self.samples))
