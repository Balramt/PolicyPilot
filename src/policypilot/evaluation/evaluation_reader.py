from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from policypilot.ingestion.config import DATA_DIR

from .evaluation_models import EvaluationCase


DEFAULT_EVALUATION_PATH = (
    DATA_DIR / "evaluation" / "markdown_cases.jsonl"
)


def load_evaluation_cases(
    file_path: str | Path = DEFAULT_EVALUATION_PATH,
) -> list[EvaluationCase]:
    """
    Load evaluation cases from a JSONL file.

    Args:
        file_path:
            Path to the evaluation JSONL file.

    Returns:
        Validated EvaluationCase objects.
    """

    path = Path(file_path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Evaluation file was not found: {path}"
        )

    cases: list[EvaluationCase] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                case = EvaluationCase.model_validate_json(line)
            except ValidationError as exc:
                raise ValueError(
                    f"Invalid evaluation case at line "
                    f"{line_number}:\n{exc}"
                ) from exc

            cases.append(case)

    if not cases:
        raise ValueError(
            "The evaluation dataset contains no cases."
        )

    return cases