from __future__ import annotations

from policypilot.embeddings.embedding_config import EmbeddingConfig
from policypilot.llm import LLMConfig, create_llm_client
from policypilot.ragservice import RAGService
from policypilot.vectorstore.chroma_store import ChromaStore


TEST_QUESTION = "What is the minimum age for applicants?"


def main() -> None:
    """
    Test the complete PolicyPilot RAG pipeline.

    The test performs these steps:

    1. Load the embedding configuration.
    2. Connect to the existing ChromaDB collection.
    3. Create the Ollama language model.
    4. Retrieve relevant policy chunks.
    5. Generate an answer using the retrieved context.
    """

    print("=" * 60)
    print("PolicyPilot RAG Pipeline Test")
    print("=" * 60)
    print(f"Question: {TEST_QUESTION}")
    print()

    print("Creating embedding configuration...")

    embedding_config = EmbeddingConfig()

    print(f"Embedding model: {embedding_config.model_name}")
    print()

    print("Connecting to ChromaDB...")

    vector_store = ChromaStore(
        embedding_config=embedding_config,
    )

    print("ChromaDB connection created.")
    print()

    print("Creating Ollama LLM client...")

    llm_config = LLMConfig()
    llm = create_llm_client(llm_config)

    print(f"LLM model: {llm_config.model_name}")
    print()

    print("Creating RAG service...")

    rag_service = RAGService(
        vector_store=vector_store,
        llm=llm,
        number_of_results=3,
    )

    print("RAG service created.")
    print()

    print("Retrieving policy context and generating answer...")

    result = rag_service.answer_question(TEST_QUESTION)

    print()
    print("=" * 60)
    print("PolicyPilot Answer")
    print("=" * 60)
    print(result.answer)

    print()
    print("=" * 60)
    print("Retrieved Sources")
    print("=" * 60)

    if not result.sources:
        print("No policy sources were retrieved.")
    else:
        for position, source in enumerate(
            result.sources,
            start=1,
        ):
            print(f"Source {position}")
            print(f"Chunk ID: {source.chunk_id}")
            print(f"Document: {source.source}")
            print(f"Similarity score: {source.score}")
            print("Content:")
            print(source.content)
            print("-" * 60)

    print()
    print("RAG pipeline test completed successfully.")


if __name__ == "__main__":
    main()