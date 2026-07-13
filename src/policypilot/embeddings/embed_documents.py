from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document

from policypilot.embeddings.embedding_config import EmbeddingConfig
from policypilot.embeddings.huggingface_embedder import (
    create_huggingface_embedder,
)
from policypilot.embeddings.jsonl_chunk_reader import (
    read_jsonl_chunks,
)


EXPECTED_EMBEDDING_DIMENSION = 384


@dataclass(frozen=True)
class DocumentEmbeddingResult:
    """
    Store documents together with their generated embedding vectors.

    The document and embedding positions correspond to each other:

    documents[0] belongs to embeddings[0]
    documents[1] belongs to embeddings[1]
    """

    documents: list[Document]
    embeddings: list[list[float]]

    @property
    def document_count(self) -> int:
        """Return the number of loaded documents."""

        return len(self.documents)

    @property
    def embedding_count(self) -> int:
        """Return the number of generated embedding vectors."""

        return len(self.embeddings)

    @property
    def embedding_dimension(self) -> int:
        """
        Return the dimension of one embedding vector.

        Return zero when the embedding list is empty.
        """

        if not self.embeddings:
            return 0

        return len(self.embeddings[0])


def validate_documents(
    documents: list[Document],
) -> None:
    """
    Validate documents before embedding generation.

    Args:
        documents:
            LangChain Document objects that will be embedded.

    Raises:
        ValueError:
            If the document list is empty or a document has empty content.

        TypeError:
            If an item is not a LangChain Document or its content
            is not a string.
    """

    if not documents:
        raise ValueError(
            "No documents were provided for embedding."
        )

    for document_index, document in enumerate(documents):
        if not isinstance(document, Document):
            raise TypeError(
                f"Item at index {document_index} is not a "
                "LangChain Document."
            )

        if not isinstance(document.page_content, str):
            raise TypeError(
                f"Document at index {document_index} does not "
                "contain string page content."
            )

        if not document.page_content.strip():
            chunk_id = document.metadata.get(
                "chunk_id",
                f"document-{document_index}",
            )

            raise ValueError(
                f"Document '{chunk_id}' contains empty page content."
            )


def validate_embeddings(
    documents: list[Document],
    embeddings: list[list[float]],
    expected_dimension: int,
) -> None:
    """
    Validate generated embedding vectors.

    Args:
        documents:
            Documents used for embedding generation.

        embeddings:
            Embedding vectors returned by the embedding model.

        expected_dimension:
            Expected number of values in each embedding vector.

    Raises:
        ValueError:
            If the expected dimension is invalid.

        RuntimeError:
            If the embedding count or dimensions are incorrect.
    """

    if expected_dimension <= 0:
        raise ValueError(
            "Expected embedding dimension must be greater than zero."
        )

    if not embeddings:
        raise RuntimeError(
            "The embedding model did not return any vectors."
        )

    if len(documents) != len(embeddings):
        raise RuntimeError(
            "Document and embedding counts do not match. "
            f"Documents: {len(documents)}, "
            f"embeddings: {len(embeddings)}."
        )

    for embedding_index, embedding in enumerate(embeddings):
        actual_dimension = len(embedding)

        if actual_dimension != expected_dimension:
            raise RuntimeError(
                f"Embedding at index {embedding_index} has "
                f"dimension {actual_dimension}. "
                f"Expected {expected_dimension}."
            )


def generate_document_embeddings(
    documents: list[Document],
    config: EmbeddingConfig | None = None,
    expected_dimension: int = EXPECTED_EMBEDDING_DIMENSION,
) -> DocumentEmbeddingResult:
    """
    Generate embeddings for LangChain documents.

    Args:
        documents:
            LangChain documents whose page content should be embedded.

        config:
            Optional embedding configuration. The default configuration
            is used when this argument is omitted.

        expected_dimension:
            Expected size of every embedding vector.

    Returns:
        Documents and their corresponding embedding vectors.

    Raises:
        ValueError:
            If document validation fails.

        RuntimeError:
            If embedding generation produces an invalid result.
    """

    embedding_config = config or EmbeddingConfig()

    validate_documents(documents)

    document_texts = [
        document.page_content
        for document in documents
    ]

    embedder = create_huggingface_embedder(
        config=embedding_config,
    )

    embeddings = embedder.embed_documents(
        document_texts
    )

    validate_embeddings(
        documents=documents,
        embeddings=embeddings,
        expected_dimension=expected_dimension,
    )

    return DocumentEmbeddingResult(
        documents=documents,
        embeddings=embeddings,
    )


def embed_documents_from_jsonl(
    jsonl_file: str | Path,
    config: EmbeddingConfig | None = None,
    expected_dimension: int = EXPECTED_EMBEDDING_DIMENSION,
) -> DocumentEmbeddingResult:
    """
    Load document chunks from JSONL and generate their embeddings.

    Args:
        jsonl_file:
            Path to the JSONL file containing document chunks.

        config:
            Optional embedding configuration.

        expected_dimension:
            Expected size of every generated embedding vector.

    Returns:
        Loaded documents and their corresponding embeddings.
    """

    documents = read_jsonl_chunks(
        file_path=jsonl_file,
    )

    return generate_document_embeddings(
        documents=documents,
        config=config,
        expected_dimension=expected_dimension,
    )