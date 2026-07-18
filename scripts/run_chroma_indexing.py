from __future__ import annotations

import torch

from policypilot.embeddings.embedding_config import (
    EmbeddingConfig,
)
from policypilot.embeddings.jsonl_chunk_reader import (
    read_jsonl_chunks,
)
from policypilot.ingestion.config import (
    DEFAULT_CHUNKS_OUTPUT_FILE,
)
from policypilot.vectorstore.chroma_config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIRECTORY,
)
from policypilot.vectorstore.chroma_store import (
    ChromaStore,
)


def main() -> None:
    """
    Load document chunks from JSONL and index them in ChromaDB.

    Chroma uses the configured Hugging Face embedding model to:
    - Generate document embeddings
    - Store the documents, metadata, IDs, and vectors
    """

    embedding_config = EmbeddingConfig(
        device="auto",
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    selected_device = embedding_config.resolve_device()

    print("=" * 60)
    print("PolicyPilot Chroma Indexing Pipeline")
    print("=" * 60)

    print(f"Input JSONL file: {DEFAULT_CHUNKS_OUTPUT_FILE}")
    print(f"Embedding model: {embedding_config.model_name}")
    print(f"Selected device: {selected_device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Batch size: {embedding_config.batch_size}")
    print(f"Chroma collection: {CHROMA_COLLECTION_NAME}")
    print(
        "Chroma persistence directory: "
        f"{CHROMA_PERSIST_DIRECTORY}"
    )

    try:
        print("\nLoading document chunks from JSONL...")

        documents = read_jsonl_chunks(
            file_path=DEFAULT_CHUNKS_OUTPUT_FILE,
        )

        print(f"Documents loaded: {len(documents)}")

        print("\nOpening Chroma vector store...")

        chroma_store = ChromaStore(
            embedding_config=embedding_config,
        )

        records_before_indexing = chroma_store.count()

        print(
            "Records currently in Chroma: "
            f"{records_before_indexing}"
        )

        print("\nGenerating embeddings and indexing documents...")

        stored_ids = chroma_store.add_documents(
            documents=documents,
        )

        records_after_indexing = chroma_store.count()

    except Exception as error:
        print("\nChroma indexing pipeline failed.")
        print(f"Reason: {error}")

        raise SystemExit(1) from error

    print("\nChroma indexing completed successfully.")

    print(f"Documents sent to Chroma: {len(documents)}")
    print(f"IDs returned by Chroma: {len(stored_ids)}")
    print(
        "Records before indexing: "
        f"{records_before_indexing}"
    )
    print(
        "Records after indexing: "
        f"{records_after_indexing}"
    )

    print("\nStored document IDs:")

    for document_id in stored_ids:
        print(f"- {document_id}")

    first_document = documents[0]

    print("\n" + "=" * 60)
    print("First indexed document")
    print("=" * 60)

    print(
        "Chunk ID: "
        f"{first_document.metadata.get('chunk_id', 'not available')}"
    )

    print("\nMetadata:")
    print(first_document.metadata)

    print("\nContent preview:")
    print(first_document.page_content[:300])

    print(
        "\nThe documents and their embeddings are now "
        "persistently stored in Chroma."
    )


if __name__ == "__main__":
    main()