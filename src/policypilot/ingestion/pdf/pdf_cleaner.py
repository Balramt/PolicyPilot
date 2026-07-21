from __future__ import annotations

import re

from langchain_core.documents import Document


NEWS_HEADER_PATTERN = re.compile(
    r"^News & Press - .+$",
    re.IGNORECASE,
)

DATE_TIME_PATTERN = re.compile(
    r"^\d{1,2}/\d{1,2}/\d{2,4},\s+"
    r"\d{1,2}:\d{2}\s+(AM|PM)$",
    re.IGNORECASE,
)

URL_PATTERN = re.compile(
    r"^https?://\S+$",
    re.IGNORECASE,
)

PAGE_COUNTER_PATTERN = re.compile(
    r"^\d+\s*/\s*\d+$",
)


def clean_pdf_page_documents(
    page_documents: list[Document],
) -> list[Document]:
    """
    Clean repeated browser-print noise from PDF page Documents.

    Each page remains a separate LangChain Document.
    The original metadata is preserved.
    """

    if not page_documents:
        raise ValueError("page_documents cannot be empty.")

    cleaned_documents: list[Document] = []

    for document in page_documents:
        cleaned_content = clean_pdf_page_content(
            document.page_content
        )

        cleaned_metadata = {
            **document.metadata,
            "cleaning_method": "basic_browser_noise_removal",
        }

        cleaned_document = Document(
            page_content=cleaned_content,
            metadata=cleaned_metadata,
        )

        cleaned_documents.append(cleaned_document)

    return cleaned_documents


def clean_pdf_page_content(
    page_content: str,
) -> str:
    """
    Remove basic browser-print noise from one PDF page.
    """

    cleaned_lines: list[str] = []

    for raw_line in page_content.splitlines():
        line = raw_line.strip()

        if is_browser_noise(line):
            continue

        cleaned_lines.append(raw_line.rstrip())

    cleaned_content = "\n".join(cleaned_lines)

    cleaned_content = re.sub(
        r"\n{3,}",
        "\n\n",
        cleaned_content,
    )

    return cleaned_content.strip()


def is_browser_noise(line: str) -> bool:
    """
    Return True when a line is browser-print noise.
    """

    if NEWS_HEADER_PATTERN.fullmatch(line):
        return True

    if DATE_TIME_PATTERN.fullmatch(line):
        return True

    if URL_PATTERN.fullmatch(line):
        return True

    if PAGE_COUNTER_PATTERN.fullmatch(line):
        return True

    return False