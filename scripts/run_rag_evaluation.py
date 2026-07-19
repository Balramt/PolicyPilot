from __future__ import annotations

from policypilot.embeddings import EmbeddingConfig
from policypilot.evaluation.evaluation_runner import run_evaluation
from policypilot.llm import LLMConfig, create_llm_client
from policypilot.ragservice import RAGService
from policypilot.vectorstore.chroma_config import ChromaConfig
from policypilot.vectorstore.chroma_store import ChromaStore


def main() -> None:
    """
    Create the PolicyPilot RAG pipeline and run evaluation.
    """

    embedding_config = EmbeddingConfig()

    chroma_config = ChromaConfig()

    chroma_store = ChromaStore(
        embedding_config=embedding_config,
        chroma_config=chroma_config,
    )

    llm_config = LLMConfig()

    llm = create_llm_client(
        config=llm_config,
    )

    rag_service = RAGService(
        vector_store=chroma_store,
        llm=llm,
    )

    run_evaluation(
        rag_service=rag_service,
    )


if __name__ == "__main__":
    main()