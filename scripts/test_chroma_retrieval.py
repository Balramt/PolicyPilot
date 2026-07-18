from __future__ import annotations

import torch

from policypilot.embeddings.embedding_config import (
    EmbeddingConfig,
)
from policypilot.vectorstore.chroma_config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIRECTORY,
)
from policypilot.vectorstore.chroma_store import (
    ChromaStore,
)


TEST_QUERY = "What is the minimum age for applicants?"
NUMBER_OF_RESULTS = 3


def main() -> None:
    """
    Test retrieval from the existing Chroma collection.
    """

    embedding_config = EmbeddingConfig(
        device="auto",
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print("=" * 60)
    print("PolicyPilot Chroma Retrieval Test")
    print("=" * 60)

    print(f"Query: {TEST_QUERY}")
    print(f"Embedding model: {embedding_config.model_name}")
    print(
        "Selected device: "
        f"{embedding_config.resolve_device()}"
    )
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Chroma collection: {CHROMA_COLLECTION_NAME}")
    print(
        "Chroma persistence directory: "
        f"{CHROMA_PERSIST_DIRECTORY}"
    )

    try:
        print("\nOpening existing Chroma vector store...")

        chroma_store = ChromaStore(
            embedding_config=embedding_config,
        )

        stored_document_count = chroma_store.count()

        print(
            "Documents stored in Chroma: "
            f"{stored_document_count}"
        )

        if stored_document_count == 0:
            raise RuntimeError(
                "The Chroma collection is empty. "
                "Run run_chroma_indexing.py first."
            )

        number_of_results = min(
            NUMBER_OF_RESULTS,
            stored_document_count,
        )

        print("\nRunning similarity search...")

        results = chroma_store.similarity_search_with_score(
            query=TEST_QUERY,
            number_of_results=number_of_results,
        )

    except Exception as error:
        print("\nRetrieval test failed.")
        print(f"Reason: {error}")

        raise SystemExit(1) from error

    print(
        f"\nRetrieved results: {len(results)}"
    )

    for result_number, (document, score) in enumerate(
        results,
        start=1,
    ):
        print("\n" + "-" * 60)
        print(f"Result {result_number}")
        print("-" * 60)

        print(
            "Chunk ID: "
            f"{document.metadata.get('chunk_id', 'not available')}"
        )

        print(
            "Chunk index: "
            f"{document.metadata.get('chunk_index', 'not available')}"
        )

        print(f"Distance score: {score:.6f}")

        print("\nContent:")
        print(document.page_content)

    print("\n" + "=" * 60)
    print("Retrieval test completed successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()