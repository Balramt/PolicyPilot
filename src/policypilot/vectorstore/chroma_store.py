from __future__ import annotations

from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from policypilot.embeddings.embedding_config import EmbeddingConfig
from policypilot.embeddings.huggingface_embedder import (
    create_huggingface_embedder,
)
from policypilot.vectorstore.chroma_config import ChromaConfig


class ChromaStore:
    """
    Manage the PolicyPilot Chroma vector store.

    The same Hugging Face embedding model is used for:

    - Generating document embeddings during indexing.
    - Generating query embeddings during similarity search.
    """

    def __init__(
        self,
        embedding_config: EmbeddingConfig | None = None,
        chroma_config: ChromaConfig | None = None,
    ) -> None:
        """
        Create or load a persistent Chroma vector store.

        Args:
            embedding_config:
                Configuration for the Hugging Face embedding model.

            chroma_config:
                Configuration for the ChromaDB collection,
                persistence directory, distance metric, and
                write batch size.

        Raises:
            TypeError:
                If an incorrect configuration object is provided.
        """

        if (
            embedding_config is not None
            and not isinstance(embedding_config, EmbeddingConfig)
        ):
            raise TypeError(
                "embedding_config must be an EmbeddingConfig "
                f"object or None, but received "
                f"{type(embedding_config).__name__}."
            )

        if (
            chroma_config is not None
            and not isinstance(chroma_config, ChromaConfig)
        ):
            raise TypeError(
                "chroma_config must be a ChromaConfig "
                f"object or None, but received "
                f"{type(chroma_config).__name__}."
            )

        self.embedding_config = (
            embedding_config
            if embedding_config is not None
            else EmbeddingConfig()
        )

        self.chroma_config = (
            chroma_config
            if chroma_config is not None
            else ChromaConfig()
        )

        self.persist_directory = (
            self.chroma_config.persist_directory
            .expanduser()
            .resolve()
        )

        self.collection_name = (
            self.chroma_config.collection_name
        )

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
            persist_directory=str(self.persist_directory),
            collection_configuration={
                "hnsw": {
                    "space": self.chroma_config.distance_metric,
                }
            },
        )

    def add_documents(
        self,
        documents: list[Document],
    ) -> list[str]:
        """
        Add LangChain documents to ChromaDB.

        Documents are written in batches using the batch size
        defined in ChromaConfig.

        The chunk_id stored in each document's metadata is used
        as the unique ChromaDB ID.

        Args:
            documents:
                LangChain document chunks to store.

        Returns:
            IDs of the documents stored in ChromaDB.

        Raises:
            TypeError:
                If documents is not a list or contains an item
                that is not a LangChain Document.

            ValueError:
                If the document list is empty, a chunk_id is
                missing, or duplicate chunk IDs are found.
        """

        if not isinstance(documents, list):
            raise TypeError(
                "documents must be provided as a list."
            )

        if not documents:
            raise ValueError(
                "No documents were provided for Chroma indexing."
            )

        ids: list[str] = []

        for document_index, document in enumerate(documents):
            if not isinstance(document, Document):
                raise TypeError(
                    f"Item at index {document_index} must be a "
                    "LangChain Document, "
                    f"but received {type(document).__name__}."
                )

            chunk_id = document.metadata.get("chunk_id")

            if not isinstance(chunk_id, str):
                raise ValueError(
                    f"Document at index {document_index} does not "
                    "contain a valid string chunk_id."
                )

            normalized_chunk_id = chunk_id.strip()

            if not normalized_chunk_id:
                raise ValueError(
                    f"Document at index {document_index} contains "
                    "an empty chunk_id."
                )

            ids.append(normalized_chunk_id)

        if len(ids) != len(set(ids)):
            raise ValueError(
                "Duplicate chunk IDs were found."
            )

        stored_ids: list[str] = []
        batch_size = self.chroma_config.write_batch_size

        for start_index in range(
            0,
            len(documents),
            batch_size,
        ):
            end_index = start_index + batch_size

            document_batch = documents[
                start_index:end_index
            ]

            id_batch = ids[
                start_index:end_index
            ]

            batch_result = self.vector_store.add_documents(
                documents=document_batch,
                ids=id_batch,
            )

            stored_ids.extend(batch_result)

        return stored_ids

    def get_all(self) -> dict[str, Any]:
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
        Return the number of records stored in ChromaDB.
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
                Optional ChromaDB metadata filter.

        Returns:
            Similar LangChain Document objects.
        """

        normalized_query = self._validate_search_input(
            query=query,
            number_of_results=number_of_results,
        )

        return self.vector_store.similarity_search(
            query=normalized_query,
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

        A lower ChromaDB distance generally represents a closer
        semantic match when cosine distance is used.

        Args:
            query:
                User question or search text.

            number_of_results:
                Maximum number of matching chunks to return.

            metadata_filter:
                Optional ChromaDB metadata filter.

        Returns:
            Document and distance-score pairs.
        """

        normalized_query = self._validate_search_input(
            query=query,
            number_of_results=number_of_results,
        )

        return self.vector_store.similarity_search_with_score(
            query=normalized_query,
            k=number_of_results,
            filter=metadata_filter,
        )

    def delete_documents(
        self,
        ids: list[str],
    ) -> None:
        """
        Delete documents from ChromaDB using chunk IDs.

        Args:
            ids:
                Chunk IDs that should be deleted.

        Raises:
            ValueError:
                If no IDs are provided.
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

        Args:
            number_of_results:
                Maximum number of documents returned by the
                retriever.

        Returns:
            A LangChain vector-store retriever.

        Raises:
            TypeError:
                If number_of_results is not an integer.

            ValueError:
                If number_of_results is less than one.
        """

        self._validate_number_of_results(
            number_of_results
        )

        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": number_of_results,
            },
        )

    @classmethod
    def _validate_search_input(
        cls,
        query: str,
        number_of_results: int,
    ) -> str:
        """
        Validate and normalize semantic-search input.

        Args:
            query:
                Search query to validate.

            number_of_results:
                Number of results requested.

        Returns:
            The cleaned search query.
        """

        if not isinstance(query, str):
            raise TypeError(
                "Search query must be a string."
            )

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError(
                "Search query must not be empty."
            )

        cls._validate_number_of_results(
            number_of_results
        )

        return normalized_query

    @staticmethod
    def _validate_number_of_results(
        number_of_results: int,
    ) -> None:
        """
        Validate the requested number of search results.
        """

        if not isinstance(number_of_results, int):
            raise TypeError(
                "Number of results must be an integer."
            )

        if number_of_results <= 0:
            raise ValueError(
                "Number of results must be greater than zero."
            )