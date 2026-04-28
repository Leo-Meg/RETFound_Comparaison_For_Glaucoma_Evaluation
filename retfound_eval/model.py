"""RETFound ViT-L/16 model definition and checkpoint loading."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import timm

from .config import INPUT_SIZE, NUM_CLASSES


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """RETFound-compatible ViT with optional global average pooling."""

    def __init__(self, global_pool: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.global_pool = global_pool

        if self.global_pool:
            norm_layer = kwargs["norm_layer"]
            embed_dim = kwargs["embed_dim"]
            self.fc_norm = norm_layer(embed_dim)
            del self.norm

    def forward_features(self, x, **kwargs):
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1, keepdim=True)
            return self.fc_norm(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x, *args, **kwargs):
        x = self.forward_features(x)
        if x.ndim == 3:
            x = x.squeeze(1)
        return self.head(x)


def create_retfound_model(
    num_classes: int = NUM_CLASSES,
    input_size: int = INPUT_SIZE,
    global_pool: bool = True,
) -> VisionTransformer:
    """Create the RETFound ViT-L/16 architecture."""

    return VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=num_classes,
        global_pool=global_pool,
        img_size=input_size,
    )


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Format de checkpoint non supporte: {type(checkpoint)!r}")


def _sanitize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    return cleaned


def load_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    num_classes: int = NUM_CLASSES,
    input_size: int = INPUT_SIZE,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a RETFound checkpoint and return an eval-mode model."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable: {checkpoint_path}")

    model = create_retfound_model(num_classes=num_classes, input_size=input_size)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _sanitize_state_dict(_extract_state_dict(checkpoint))
    load_result = model.load_state_dict(state_dict, strict=False)

    metadata: dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path),
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
    }
    if isinstance(checkpoint, dict):
        for key in ("epoch", "args"):
            if key in checkpoint:
                metadata[key] = checkpoint[key]

    model.eval()
    model.to(device)
    return model, metadata
