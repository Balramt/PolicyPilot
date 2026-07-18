from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

from policypilot.embeddings.embed_documents import (
    DocumentEmbeddingResult,
)
from policypilot.vectorstore.chroma_config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DISTANCE_METRIC,
    CHROMA_PERSIST_DIRECTORY,
    CHROMA_WRITE_BATCH_SIZE,
)


class ChromaStore:
    """
    Store and retrieve PolicyPilot document embeddings using ChromaDB.

    PolicyPilot generates embeddings before calling this class.
    Chroma only stores and searches the supplied vectors.
    """

    def __init__(
        self,
        persist_directory: str | Path = CHROMA_PERSIST_DIRECTORY,
        collection_name: str = CHROMA_COLLECTION_NAME,
        distance_metric: str = CHROMA_DISTANCE_METRIC,
        write_batch_size: int = CHROMA_WRITE_BATCH_SIZE,
    ) -> None:
        """
        Create a persistent Chroma client and collection.

        Args:
            persist_directory:
                Directory where Chroma stores its database files.

            collection_name:
                Name of the Chroma collection.

            distance_metric:
                Vector distance metric used by Chroma.

            write_batch_size:
                Number of records written in one database operation.
        """

        if write_batch_size <= 0:
            raise ValueError(
                "Chroma write batch size must be greater than zero."
            )

        self.persist_directory = (
            Path(persist_directory)
            .expanduser()
            .resolve()
        )

        self.collection_name = collection_name
        self.distance_metric = distance_metric
        self.write_batch_size = write_batch_size

        self.persist_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=None,
            configuration={
                "hnsw": {
                    "space": self.distance_metric,
                }
            },
        )

    def upsert_document_embeddings(
        self,
        result: DocumentEmbeddingResult,
    ) -> int:
        """
        Store documents and their existing embeddings in Chroma.

        Existing records with the same chunk ID are updated.
        New chunk IDs are inserted.

        Args:
            result:
                Documents and embeddings produced by the PolicyPilot
                embedding pipeline.

        Returns:
            Number of records sent to Chroma.

        Raises:
            ValueError:
                If the result is empty, counts do not match, or a
                document does not contain a valid chunk ID.
        """

        if result.document_count == 0:
            raise ValueError(
                "No documents were provided for Chroma indexing."
            )

        if result.document_count != result.embedding_count:
            raise ValueError(
                "Document and embedding counts do not match. "
                f"Documents: {result.document_count}, "
                f"embeddings: {result.embedding_count}."
            )

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        embeddings: list[list[float]] = []

        for document, embedding in zip(
            result.documents,
            result.embeddings,
            strict=True,
        ):
            chunk_id = document.metadata.get("chunk_id")

            if not isinstance(chunk_id, str):
                raise ValueError(
                    "Every document must contain a string "
                    "'chunk_id' in its metadata."
                )

            if not chunk_id.strip():
                raise ValueError(
                    "Document chunk IDs must not be empty."
                )

            ids.append(chunk_id)
            documents.append(document.page_content)
            metadatas.append(dict(document.metadata))
            embeddings.append(embedding)

        if len(ids) != len(set(ids)):
            raise ValueError(
                "Duplicate chunk IDs were found in the "
                "embedding result."
            )

        for batch_start in range(
            0,
            len(ids),
            self.write_batch_size,
        ):
            batch_end = (
                batch_start + self.write_batch_size
            )

            self.collection.upsert(
                ids=ids[batch_start:batch_end],
                documents=documents[batch_start:batch_end],
                metadatas=metadatas[batch_start:batch_end],
                embeddings=embeddings[batch_start:batch_end],
            )

        return len(ids)

    def count(self) -> int:
        """
        Return the number of records stored in the collection.
        """

        return self.collection.count()

    def get_all(
        self,
    ) -> dict[str, Any]:
        """
        Return all stored IDs, documents, and metadata.

        Embedding vectors are excluded because they are large and
        are not normally required when inspecting stored records.
        """

        return self.collection.get(
            include=[
                "documents",
                "metadatas",
            ],
        )

    def get_by_ids(
        self,
        ids: list[str],
    ) -> dict[str, Any]:
        """
        Return selected records using their Chroma IDs.

        Args:
            ids:
                Chunk IDs to retrieve.
        """

        if not ids:
            raise ValueError(
                "At least one chunk ID must be provided."
            )

        return self.collection.get(
            ids=ids,
            include=[
                "documents",
                "metadatas",
            ],
        )

    def query_by_embedding(
        self,
        query_embedding: list[float],
        number_of_results: int = 3,
        metadata_filter: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Search Chroma using an existing query embedding.

        Args:
            query_embedding:
                Vector generated from the user's query using the same
                embedding model used for the stored documents.

            number_of_results:
                Maximum number of similar chunks to return.

            metadata_filter:
                Optional Chroma metadata filter.

        Returns:
            Matching IDs, documents, metadata, and distances.

        Raises:
            ValueError:
                If the query embedding is empty, the result count is
                invalid, or the collection is empty.
        """

        if not query_embedding:
            raise ValueError(
                "Query embedding must not be empty."
            )

        if number_of_results <= 0:
            raise ValueError(
                "Number of results must be greater than zero."
            )

        collection_count = self.count()

        if collection_count == 0:
            raise ValueError(
                "The Chroma collection is empty."
            )

        result_limit = min(
            number_of_results,
            collection_count,
        )

        query_arguments: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": result_limit,
            "include": [
                "documents",
                "metadatas",
                "distances",
            ],
        }

        if metadata_filter is not None:
            query_arguments["where"] = metadata_filter

        return self.collection.query(
            **query_arguments,
        )