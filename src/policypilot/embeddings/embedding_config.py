from __future__ import annotations

from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field


class EmbeddingConfig(BaseModel):
    """
    Configuration for the PolicyPilot embedding model.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        min_length=1,
    )

    device: str = Field(
        default="auto",
        min_length=1,
    )

    batch_size: int = Field(
        default=32,
        gt=0,
    )

    normalize_embeddings: bool = True
    show_progress_bar: bool = True

    def resolve_device(self) -> str:
        """
        Resolve automatic device selection.

        Returns:
            CUDA when available; otherwise CPU.
        """

        if self.device != "auto":
            return self.device

        return "cuda:0" if torch.cuda.is_available() else "cpu"