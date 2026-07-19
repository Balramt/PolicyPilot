from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from policypilot.ingestion.config import DATA_DIR


class ChromaConfig(BaseModel):
    """
    Configuration for the PolicyPilot ChromaDB collection.

    Attributes:
        collection_name:
            Name of the ChromaDB collection.

        persist_directory:
            Directory where the persistent ChromaDB files are stored.

        distance_metric:
            Distance metric used when comparing embedding vectors.

        write_batch_size:
            Maximum number of documents added to ChromaDB in one batch.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    collection_name: str = Field(
        default="policypilot_policy_chunks",
        min_length=1,
        description="Name of the ChromaDB collection.",
    )

    persist_directory: Path = Field(
        default=DATA_DIR / "vectorstore" / "chroma",
        description="Directory used to persist the ChromaDB collection.",
    )

    distance_metric: Literal["cosine", "l2", "ip"] = Field(
        default="cosine",
        description="Distance metric used to compare embedding vectors.",
    )

    write_batch_size: int = Field(
        default=100,
        gt=0,
        description="Number of documents written to ChromaDB in one batch.",
    )