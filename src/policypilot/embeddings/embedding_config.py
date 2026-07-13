from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for the PolicyPilot embedding model."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "auto"
    batch_size: int = 32
    normalize_embeddings: bool = True
    show_progress_bar: bool = True

    def resolve_device(self) -> str:
        """Select GPU when available; otherwise use CPU."""

        if self.device == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"

        if self.device == "cpu":
            return "cpu"

        if self.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was requested, but no NVIDIA GPU is available."
                )

            return self.device

        raise ValueError(
            "Device must be 'auto', 'cpu', 'cuda', or 'cuda:0'."
        )
    