from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

from policypilot.embeddings.embedding_config import EmbeddingConfig


def create_huggingface_embedder(
    config: EmbeddingConfig | None = None,
) -> HuggingFaceEmbeddings:
    """
    Create and configure the Hugging Face embedding model.

    The model runs on:
    - CPU when no GPU is available
    - CUDA GPU when available
    """

    embedding_config = config or EmbeddingConfig()
    selected_device = embedding_config.resolve_device()

    embedder = HuggingFaceEmbeddings(
        model_name=embedding_config.model_name,
        model_kwargs={
            "device": selected_device,
        },
        encode_kwargs={
            "batch_size": embedding_config.batch_size,
            "normalize_embeddings": (
                embedding_config.normalize_embeddings
            ),
        },
        show_progress=embedding_config.show_progress_bar,
    )

    return embedder