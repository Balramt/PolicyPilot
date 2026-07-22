from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document

SECTION_HEADING_PATTERN = re.compile(
    r"^##\s+(?:\*\*)?"
    r"Section\s+(?P<section_number>\d+[A-Za-z]?)"
    r"\s*[-–—:]\s*"
    r"(?P<section_title>.+?)"
    r"(?:\*\*)?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

PAGE_MARKER_PATTERN = re.compile(
    r"<!--\s*PDF_PAGE_(?:START|END):\s*(\d+)\s*-->",
    re.IGNORECASE,
)


def parse_pdf_sections(
    assembled_document: Document,
) -> list[Document]:
    """
    Split one assembled PDF Document into logical section Documents.

    Text before the first detected section is returned as a
    separate preamble Document.

    Section headings are expected to use Markdown such as:

        ## Section 1 - General information

    or:

        ## **Section 1 - General information**

    Page markers remain inside the section content for later
    page-aware chunking. Page ranges are also added to metadata.

    Args:
        assembled_document:
            One PDF Document created by the PDF assembler.

    Returns:
        A list containing the document preamble and one Document
        for every detected section.

    Raises:
        ValueError:
            If the assembled Document is empty or contains no
            detectable section headings.
    """

    assembled_content = assembled_document.page_content.strip()

    if not assembled_content:
        raise ValueError("assembled_document cannot contain empty content.")

    section_matches = list(SECTION_HEADING_PATTERN.finditer(assembled_content))

    if not section_matches:
        raise ValueError("No PDF section headings were detected.")

    section_documents: list[Document] = []
    total_sections = len(section_matches)

    first_section_start = section_matches[0].start()

    preamble_content = assembled_content[:first_section_start].strip()

    if preamble_content:
        preamble_metadata = _create_block_metadata(
            original_metadata=assembled_document.metadata,
            block_content=preamble_content,
            content_type="preamble",
            section_index=0,
            section_title="Document Preamble",
            section_number=None,
            section_id=_create_preamble_id(assembled_document.metadata),
            total_sections=total_sections,
        )

        preamble_document = Document(
            page_content=preamble_content,
            metadata=preamble_metadata,
        )

        section_documents.append(preamble_document)

    for match_index, section_match in enumerate(section_matches):
        section_start = section_match.start()

        if match_index + 1 < total_sections:
            section_end = section_matches[match_index + 1].start()
        else:
            section_end = len(assembled_content)

        section_content = assembled_content[section_start:section_end].strip()

        section_number = section_match.group("section_number").strip()

        section_title = section_match.group("section_title").strip()

        section_title = section_title.strip("*").strip()

        section_index = match_index + 1

        section_id = _create_section_id(
            metadata=assembled_document.metadata,
            section_number=section_number,
        )

        section_metadata = _create_block_metadata(
            original_metadata=assembled_document.metadata,
            block_content=section_content,
            content_type="section",
            section_index=section_index,
            section_title=section_title,
            section_number=section_number,
            section_id=section_id,
            total_sections=total_sections,
        )

        section_document = Document(
            page_content=section_content,
            metadata=section_metadata,
        )

        section_documents.append(section_document)

    return section_documents


def _create_block_metadata(
    *,
    original_metadata: dict[str, Any],
    block_content: str,
    content_type: str,
    section_index: int,
    section_title: str,
    section_number: str | None,
    section_id: str,
    total_sections: int,
) -> dict[str, Any]:
    """
    Create metadata for one preamble or section Document.
    """

    metadata = {
        **original_metadata,
    }

    metadata.pop(
        "page_start",
        None,
    )

    metadata.pop(
        "page_end",
        None,
    )

    metadata.update(
        {
            "content_type": content_type,
            "section_index": section_index,
            "section_title": section_title,
            "section_id": section_id,
            "total_sections": total_sections,
            "section_parser": ("markdown_section_heading"),
        }
    )

    if section_number is not None:
        metadata["section_number"] = section_number

    page_range = _extract_page_range(block_content)

    if page_range is not None:
        page_start, page_end = page_range

        metadata["page_start"] = page_start
        metadata["page_end"] = page_end

    return metadata


def _extract_page_range(
    content: str,
) -> tuple[int, int] | None:
    """
    Extract the smallest and largest PDF page numbers
    found in one content block.
    """

    page_numbers = [
        int(page_number) for page_number in PAGE_MARKER_PATTERN.findall(content)
    ]

    if not page_numbers:
        return None

    return (
        min(page_numbers),
        max(page_numbers),
    )


def _create_preamble_id(
    metadata: dict[str, Any],
) -> str:
    """
    Create a stable identifier for the PDF preamble.
    """

    file_name = metadata.get(
        "file_name",
        "pdf-document",
    )

    return f"{file_name}-preamble"


def _create_section_id(
    *,
    metadata: dict[str, Any],
    section_number: str,
) -> str:
    """
    Create a stable identifier for one PDF section.
    """

    file_name = metadata.get(
        "file_name",
        "pdf-document",
    )

    return f"{file_name}-section-{section_number}"
