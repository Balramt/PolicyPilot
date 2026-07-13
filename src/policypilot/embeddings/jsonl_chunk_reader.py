from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


SUPPORTED_CONTENT_KEYS = (
    "page_content",
    "content",
    "text",
)


def read_jsonl_chunks(
    file_path: str | Path,
    *,
    content_key: str | None = None,
    metadata_key: str = "metadata",
    skip_empty: bool = True,
) -> list[Document]:
    """
    Read document chunks from a JSONL file.

    Each non-empty line must contain one JSON object.

    Supported text fields:
    - page_content
    - content
    - text

    Args:
        file_path:
            Path to the JSONL file.

        content_key:
            Explicit name of the field containing the chunk text.
            When omitted, the reader automatically detects a supported key.

        metadata_key:
            Name of the field containing metadata.

        skip_empty:
            Skip chunks whose text is empty. When False, empty chunks
            raise a ValueError.

    Returns:
        A list of LangChain Document objects.

    Raises:
        FileNotFoundError:
            When the JSONL file does not exist.

        ValueError:
            When a line contains invalid JSON or has an invalid structure.
    """

    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"JSONL chunk file does not exist: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Expected a file, but received: {path}"
        )

    documents: list[Document] = []

    with path.open("r", encoding="utf-8") as jsonl_file:
        for line_number, raw_line in enumerate(jsonl_file, start=1):
            line = raw_line.strip()

            # Ignore completely empty lines.
            if not line:
                continue

            record = _parse_json_line(
                line=line,
                line_number=line_number,
                file_path=path,
            )

            selected_content_key = _resolve_content_key(
                record=record,
                requested_content_key=content_key,
                line_number=line_number,
            )

            page_content = record[selected_content_key]

            if not isinstance(page_content, str):
                raise ValueError(
                    f"Line {line_number}: field "
                    f"'{selected_content_key}' must contain a string."
                )

            if not page_content.strip():
                if skip_empty:
                    continue

                raise ValueError(
                    f"Line {line_number}: chunk text is empty."
                )

            metadata = _extract_metadata(
                record=record,
                metadata_key=metadata_key,
                content_key=selected_content_key,
                line_number=line_number,
                file_path=path,
            )

            document = Document(
                page_content=page_content,
                metadata=metadata,
            )

            documents.append(document)

    return documents


def _parse_json_line(
    *,
    line: str,
    line_number: int,
    file_path: Path,
) -> dict[str, Any]:
    """Parse and validate one JSONL line."""

    try:
        record = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON at {file_path}, line {line_number}: "
            f"{error.msg}"
        ) from error

    if not isinstance(record, dict):
        raise ValueError(
            f"Line {line_number}: each JSONL line must contain "
            "a JSON object."
        )

    return record


def _resolve_content_key(
    *,
    record: dict[str, Any],
    requested_content_key: str | None,
    line_number: int,
) -> str:
    """Find the field containing the chunk text."""

    if requested_content_key is not None:
        if requested_content_key not in record:
            raise ValueError(
                f"Line {line_number}: content field "
                f"'{requested_content_key}' was not found."
            )

        return requested_content_key

    for supported_key in SUPPORTED_CONTENT_KEYS:
        if supported_key in record:
            return supported_key

    supported_keys = ", ".join(SUPPORTED_CONTENT_KEYS)

    raise ValueError(
        f"Line {line_number}: no supported content field was found. "
        f"Expected one of: {supported_keys}."
    )


def _extract_metadata(
    *,
    record: dict[str, Any],
    metadata_key: str,
    content_key: str,
    line_number: int,
    file_path: Path,
) -> dict[str, Any]:
    """Extract metadata and preserve useful top-level JSONL fields."""

    raw_metadata = record.get(metadata_key, {})

    if raw_metadata is None:
        raw_metadata = {}

    if not isinstance(raw_metadata, dict):
        raise ValueError(
            f"Line {line_number}: field '{metadata_key}' "
            "must contain a JSON object."
        )

    metadata = dict(raw_metadata)

    # Preserve top-level fields such as chunk_id or document_id.
    excluded_keys = {
        content_key,
        metadata_key,
    }

    for key, value in record.items():
        if key not in excluded_keys:
            metadata.setdefault(key, value)

    # Helpful debugging information.
    metadata.setdefault("jsonl_line_number", line_number)
    metadata.setdefault("jsonl_source_file", str(file_path))

    return metadata