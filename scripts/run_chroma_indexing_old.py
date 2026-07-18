from __future__ import annotations

from typing import Any

import torch

from policypilot.embeddings.embed_documents import (
    EXPECTED_EMBEDDING_DIMENSION,
    embed_documents_from_jsonl,
)
from policypilot.embeddings.embedding_config import (
    EmbeddingConfig,
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


def display_stored_records(
    records: dict[str, Any],
) -> None:
    """
    Display documents and metadata stored in Chroma.

    Args:
        records:
            Records returned by Chroma's get operation.
    """

    ids = records.get("ids", [])
    documents = records.get("documents") or []
    metadatas = records.get("metadatas") or []

    print("\n" + "=" * 60)
    print("Records stored in Chroma")
    print("=" * 60)

    if not ids:
        print("No records were found in the collection.")
        return

    for record_number, (
        record_id,
        document,
        metadata,
    ) in enumerate(
        zip(
            ids,
            documents,
            metadatas,
            strict=True,
        ),
        start=1,
    ):
        print("\n" + "-" * 60)
        print(f"Record number: {record_number}")
        print(f"Chroma ID: {record_id}")

        print("\nMetadata:")
        print(metadata)

        print("\nDocument preview:")

        if isinstance(document, str):
            print(document[:300])
        else:
            print("Document content is not available.")


def main() -> None:
    """
    Generate document embeddings and store them in ChromaDB.
    """

    embedding_config = EmbeddingConfig(
        device="auto",
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    selected_device = (
        embedding_config.resolve_device()
    )

    print("=" * 60)
    print("PolicyPilot Chroma Indexing Pipeline")
    print("=" * 60)

    print(
        f"Input JSONL file: "
        f"{DEFAULT_CHUNKS_OUTPUT_FILE}"
    )

    print(
        f"Embedding model: "
        f"{embedding_config.model_name}"
    )

    print(f"Selected device: {selected_device}")
    print(
        f"CUDA available: "
        f"{torch.cuda.is_available()}"
    )

    print(
        f"Expected embedding dimension: "
        f"{EXPECTED_EMBEDDING_DIMENSION}"
    )

    print(
        f"Chroma collection: "
        f"{CHROMA_COLLECTION_NAME}"
    )

    print(
        f"Chroma persistence directory: "
        f"{CHROMA_PERSIST_DIRECTORY}"
    )

    try:
        print("\nLoading document chunks from JSONL...")

        embedding_result = (
            embed_documents_from_jsonl(
                jsonl_file=DEFAULT_CHUNKS_OUTPUT_FILE,
                config=embedding_config,
                expected_dimension=(
                    EXPECTED_EMBEDDING_DIMENSION
                ),
            )
        )

        print(
            f"Documents loaded: "
            f"{embedding_result.document_count}"
        )

        print(
            f"Embeddings generated: "
            f"{embedding_result.embedding_count}"
        )

        print(
            f"Embedding dimension: "
            f"{embedding_result.embedding_dimension}"
        )

        print("\nOpening Chroma vector store...")

        chroma_store = ChromaStore()

        print("Storing documents and embeddings...")

        records_sent = (
            chroma_store.upsert_document_embeddings(
                result=embedding_result,
            )
        )

        total_records = chroma_store.count()

        stored_records = chroma_store.get_all()

    except (
        FileNotFoundError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as error:
        print("\nChroma indexing pipeline failed.")
        print(f"Reason: {error}")

        raise SystemExit(1) from error

    print("\nChroma indexing completed successfully.")

    print(
        f"Records sent to Chroma: "
        f"{records_sent}"
    )

    print(
        f"Total records in collection: "
        f"{total_records}"
    )

    print(
        f"Collection name: "
        f"{chroma_store.collection_name}"
    )

    print(
        f"Database directory: "
        f"{chroma_store.persist_directory}"
    )

    display_stored_records(
        records=stored_records,
    )

    print(
        "\nThe document embeddings are now "
        "persistently stored in Chroma."
    )


if __name__ == "__main__":
    main()