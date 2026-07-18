from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


SYSTEM_PROMPT = """
You are PolicyPilot, an assistant that answers questions about policy documents.

Follow these rules:

1. Answer only from the provided policy context.
2. Do not use outside knowledge.
3. Do not invent rules, dates, requirements, or conditions.
4. Keep the answer clear and concise.
5. If the answer is not available in the context, say:
   "I could not find this information in the provided policy documents."
""".strip()


def create_rag_prompt() -> ChatPromptTemplate:
    """
    Create the prompt template used by the PolicyPilot RAG service.

    The prompt contains:

    - System instructions for the language model.
    - Policy context retrieved from ChromaDB.
    - The user's question.

    Returns:
        A configured ChatPromptTemplate.
    """

    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_PROMPT,
            ),
            (
                "human",
                """
Policy context:

{context}

User question:

{question}

Answer the question using only the policy context.
""".strip(),
            ),
        ]
    )