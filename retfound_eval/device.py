"""Device selection helpers."""

from __future__ import annotations

import torch


def choose_device(preferred: str = "auto") -> torch.device:
    """Choose a torch device with clear auto-fallback behavior."""

    preferred = preferred.lower()
    if preferred != "auto":
        device = torch.device(preferred)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA demande mais non disponible sur cette machine.")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS demande mais non disponible sur cette machine.")
        return device

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
