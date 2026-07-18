from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from policypilot.embeddings.embedding_config import (
    EmbeddingConfig,
)
from policypilot.embeddings.huggingface_embedder import (
    create_huggingface_embedder,
)
from policypilot.vectorstore.chroma_config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DISTANCE_METRIC,
    CHROMA_PERSIST_DIRECTORY,
)


class ChromaStore:
    """
    Manage the PolicyPilot Chroma vector store.

    The same Hugging Face embedding model is used for:

    - Generating document embeddings during indexing
    - Generating query embeddings during similarity search
    """

    def __init__(
        self,
        embedding_config: EmbeddingConfig | None = None,
        persist_directory: str | Path = CHROMA_PERSIST_DIRECTORY,
        collection_name: str = CHROMA_COLLECTION_NAME,
    ) -> None:
        """
        Create or load a persistent Chroma vector store.

        Args:
            embedding_config:
                Configuration for the Hugging Face embedding model.

            persist_directory:
                Directory where Chroma stores its database files.

            collection_name:
                Name of the Chroma collection.
        """

        self.embedding_config = (
            embedding_config or EmbeddingConfig()
        )

        self.persist_directory = (
            Path(persist_directory)
            .expanduser()
            .resolve()
        )

        self.collection_name = collection_name

        self.persist_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.embedding_function = (
            create_huggingface_embedder(
                config=self.embedding_config,
            )
        )

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=str(
                self.persist_directory
            ),
            collection_configuration={
                "hnsw": {
                    "space": CHROMA_DISTANCE_METRIC,
                }
            },
        )

    def add_documents(
        self,
        documents: list[Document],
    ) -> list[str]:
        """
        Add LangChain documents to Chroma.

        The chunk_id stored in each document's metadata is used
        as the unique Chroma ID.

        Args:
            documents:
                LangChain Document chunks to store.

        Returns:
            IDs of the documents stored in Chroma.

        Raises:
            ValueError:
                If the document list is empty or a chunk_id is missing.
        """

        if not documents:
            raise ValueError(
                "No documents were provided for Chroma indexing."
            )

        ids: list[str] = []

        for document_index, document in enumerate(documents):
            chunk_id = document.metadata.get("chunk_id")

            if not isinstance(chunk_id, str):
                raise ValueError(
                    f"Document at index {document_index} does not "
                    "contain a valid string chunk_id."
                )

            if not chunk_id.strip():
                raise ValueError(
                    f"Document at index {document_index} contains "
                    "an empty chunk_id."
                )

            ids.append(chunk_id)

        if len(ids) != len(set(ids)): # The code checks for duplicates
            raise ValueError(
                "Duplicate chunk IDs were found."
            )

        return self.vector_store.add_documents(
            documents=documents,
            ids=ids,
        )

    def get_all(
        self,
    ) -> dict[str, Any]:
        """
        Return all stored document IDs, texts, and metadata.
        """

        return self.vector_store.get(
            include=[
                "documents",
                "metadatas",
            ],
        )

    def count(self) -> int:
        """
        Return the number of records stored in Chroma.
        """

        records = self.vector_store.get(
            include=[],
        )

        return len(records.get("ids", []))

    def similarity_search(
        self,
        query: str,
        number_of_results: int = 3,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Find document chunks semantically similar to a query.

        Args:
            query:
                User question or search text.

            number_of_results:
                Maximum number of matching chunks to return.

            metadata_filter:
                Optional metadata filter.

        Returns:
            Similar LangChain Document objects.
        """

        self._validate_search_input(
            query=query,
            number_of_results=number_of_results,
        )

        return self.vector_store.similarity_search(
            query=query.strip(),
            k=number_of_results,
            filter=metadata_filter,
        )

    def similarity_search_with_score(
        self,
        query: str,
        number_of_results: int = 3,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Find similar chunks and include their distance scores.

        A lower Chroma distance generally represents a closer match
        when using cosine distance.

        Args:
            query:
                User question or search text.

            number_of_results:
                Maximum number of matching chunks to return.

            metadata_filter:
                Optional metadata filter.

        Returns:
            Document and score pairs.
        """

        self._validate_search_input(
            query=query,
            number_of_results=number_of_results,
        )

        return self.vector_store.similarity_search_with_score(
            query=query.strip(),
            k=number_of_results,
            filter=metadata_filter,
        )

    def delete_documents(
        self,
        ids: list[str],
    ) -> None:
        """
        Delete documents from Chroma using their chunk IDs.
        """

        if not ids:
            raise ValueError(
                "At least one document ID must be provided."
            )

        self.vector_store.delete(
            ids=ids,
        )

    def as_retriever(
        self,
        number_of_results: int = 3,
    ):
        """
        Return the Chroma vector store as a LangChain retriever.

        This will later be used by the RAG pipeline.
        """

        if number_of_results <= 0:
            raise ValueError(
                "Number of results must be greater than zero."
            )

        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": number_of_results,
            },
        )

    @staticmethod
    def _validate_search_input(
        query: str,
        number_of_results: int,
    ) -> None:
        """
        Validate semantic-search input.
        """

        if not isinstance(query, str):
            raise TypeError(
                "Search query must be a string."
            )

        if not query.strip():
            raise ValueError(
                "Search query must not be empty."
            )

        if number_of_results <= 0:
            raise ValueError(
                "Number of results must be greater than zero."
            )