from __future__ import annotations

import torch

from policypilot.embeddings import (
    EmbeddingConfig,
    create_huggingface_embedder,
)


def main() -> None:
    # "auto" uses GPU when available; otherwise it uses CPU.
    config = EmbeddingConfig(
        device="auto",
        batch_size=32,
    )

    selected_device = config.resolve_device()

    print("=" * 60)
    print("PolicyPilot Embedding Test")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Selected device: {selected_device}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    print("\nLoading embedding model...")

    embedder = create_huggingface_embedder(config)

    documents = [
        "The insurance policy covers hospitalization expenses.",
        "The policyholder must pay the applicable deductible.",
        "Pre-existing conditions may have a waiting period.",
    ]

    query = "Does the policy cover hospitalization?"

    print("Generating document embeddings...")

    document_embeddings = embedder.embed_documents(documents)

    print("Generating query embedding...")

    query_embedding = embedder.embed_query(query)

    print("\nEmbedding generation completed successfully.")
    print(f"Number of documents: {len(documents)}")
    print(f"Number of document vectors: {len(document_embeddings)}")
    print(f"Document vector dimension: {len(document_embeddings[0])}")
    print(f"Query vector dimension: {len(query_embedding)}")
    print(f"First five vector values: {document_embeddings[0][:5]}")


if __name__ == "__main__":
    main()