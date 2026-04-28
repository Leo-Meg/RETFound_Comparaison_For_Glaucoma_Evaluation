"""High-level evaluation runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import CLASS_NAMES, NUM_CLASSES, SPLITS, checkpoint_path, get_dataset_config
from .data import GlaucomaGradingDataset, eval_transform
from .device import choose_device
from .metrics import compute_metrics
from .model import load_checkpoint
from .plots import save_confusion_matrix, save_roc_curves


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def run_evaluation(
    train_dataset: str,
    eval_dataset: str,
    repo_root: str | Path = ".",
    check_dir: str | Path = "check",
    output_dir: str | Path = "results",
    splits: tuple[str, ...] = SPLITS,
    batch_size: int = 16,
    num_workers: int = 0,
    device_name: str = "auto",
    save_plots: bool = True,
    split_mode: str = "all",
    external_only: bool = False,
) -> dict[str, Any]:
    """Evaluate one source checkpoint on one target dataset."""

    repo_root = Path(repo_root)
    output_dir = Path(output_dir)
    pair_name = f"train-{train_dataset}__eval-{eval_dataset}"
    pair_dir = output_dir / pair_name
    pair_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(device_name)
    target_cfg = get_dataset_config(eval_dataset)
    ckpt = checkpoint_path(repo_root / check_dir, train_dataset)

    model, checkpoint_metadata = load_checkpoint(ckpt, device=device)
    dataset = GlaucomaGradingDataset(
        config=target_cfg,
        splits=splits,
        transform=eval_transform(),
        repo_root=repo_root,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[list[float]] = []
    image_paths: list[str] = []
    image_splits: list[str] = []

    with torch.no_grad():
        for images, labels, paths, batch_splits in tqdm(
            loader,
            desc=f"{train_dataset} -> {eval_dataset}",
            leave=False,
        ):
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())
            image_paths.extend(paths)
            image_splits.extend(batch_splits)

    y_true_array = np.asarray(y_true, dtype=int)
    y_pred_array = np.asarray(y_pred, dtype=int)
    y_prob_array = np.asarray(y_prob, dtype=float)

    metrics, details = compute_metrics(y_true_array, y_pred_array, y_prob_array)
    metrics.update(
        {
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "checkpoint": str(ckpt),
            "device": str(device),
            "splits": ",".join(splits),
            "split_mode": split_mode,
            "comparison_scope": "external_only" if external_only else "internal_external",
        }
    )

    predictions = pd.DataFrame(
        {
            "image_path": image_paths,
            "split": image_splits,
            "true_label": y_true_array,
            "pred_label": y_pred_array,
            "true_class": [CLASS_NAMES[i] for i in y_true_array],
            "pred_class": [CLASS_NAMES[i] for i in y_pred_array],
            "correct": y_true_array == y_pred_array,
            **{
                f"prob_class_{class_idx}": y_prob_array[:, class_idx]
                for class_idx in range(NUM_CLASSES)
            },
        }
    )
    predictions.to_csv(pair_dir / "predictions.csv", index=False)

    pd.DataFrame([metrics]).to_csv(pair_dir / "metrics.csv", index=False)
    with (pair_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": metrics,
                "checkpoint_metadata": checkpoint_metadata,
                "classification_report": details["classification_report"],
                "confusion_matrix": details["confusion_matrix"],
            },
            f,
            indent=2,
            default=_json_default,
        )

    if save_plots:
        save_confusion_matrix(
            details["confusion_matrix"],
            pair_dir / "confusion_matrix.png",
            title=f"RETFound {train_dataset} -> {eval_dataset}",
        )
        save_roc_curves(
            y_true_array,
            y_prob_array,
            pair_dir / "roc_curves.png",
            title=f"ROC RETFound {train_dataset} -> {eval_dataset}",
        )

    return {"metrics": metrics, "output_dir": str(pair_dir)}
