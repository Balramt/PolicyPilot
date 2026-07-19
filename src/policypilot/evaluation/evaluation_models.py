from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EvaluationCase(BaseModel):
    """
    Represent one PolicyPilot evaluation question.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )

    case_id: str = Field(
        min_length=1,
        description="Unique ID of the evaluation case.",
    )

    question: str = Field(
        min_length=1,
        description="Question sent to the RAG pipeline.",
    )

    expected_answer: str = Field(
        min_length=1,
        description="Expected answer for the question.",
    )

    expected_chunk_id: str | None = Field(
        default=None,
        description="Expected ChromaDB chunk ID.",
    )

    required_term: str | None = Field(
        default=None,
        description="Important text expected in the generated answer.",
    )

    answer_available: bool = Field(
        description="Whether the policy contains the answer.",
    )