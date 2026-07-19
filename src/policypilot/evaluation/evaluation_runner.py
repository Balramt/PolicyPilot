from __future__ import annotations

from policypilot.ragservice import RAGService

from .evaluation_metrics import (
    chunk_match,
    fallback_answer_match,
    required_term_match,
)
from .evaluation_models import EvaluationCase
from .evaluation_reader import load_evaluation_cases


def run_evaluation(
    rag_service: RAGService,
) -> None:
    """
    Run all evaluation cases through the RAG pipeline.
    """

    cases = load_evaluation_cases()

    passed_cases = 0

    for case in cases:
        case_passed = evaluate_case(
            rag_service=rag_service,
            case=case,
        )

        if case_passed:
            passed_cases += 1

    print("\nEvaluation Summary")
    print("------------------")
    print(f"Total cases: {len(cases)}")
    print(f"Passed cases: {passed_cases}")
    print(f"Failed cases: {len(cases) - passed_cases}")


def evaluate_case(
    rag_service: RAGService,
    case: EvaluationCase,
) -> bool:
    """
    Run and evaluate one question.
    """

    response = rag_service.answer_question(
        case.question
    )

    retrieved_chunk_ids = [
        source.chunk_id
        for source in response.sources
    ]

    if case.answer_available:
        chunk_correct = chunk_match(
            case.expected_chunk_id,
            retrieved_chunk_ids,
        )

        answer_correct = required_term_match(
            case.required_term,
            response.answer,
        )

        case_passed = chunk_correct and answer_correct

    else:
        chunk_correct = True
        answer_correct = fallback_answer_match(
            response.answer
        )

        case_passed = answer_correct

    print("\n------------------------------")
    print(f"Case: {case.case_id}")
    print(f"Question: {case.question}")
    print(f"Expected answer: {case.expected_answer}")
    print(f"Generated answer: {response.answer}")
    print(f"Retrieved chunks: {retrieved_chunk_ids}")
    print(f"Chunk match: {chunk_correct}")
    print(f"Answer match: {answer_correct}")
    print(f"Passed: {case_passed}")

    return case_passed