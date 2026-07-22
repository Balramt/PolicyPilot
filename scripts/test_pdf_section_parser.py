from __future__ import annotations

import json

from policypilot.ingestion.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from policypilot.ingestion.pdf.pdf_assembler import (
    assemble_pdf_document,
)
from policypilot.ingestion.pdf.pdf_cleaner import (
    clean_pdf_page_documents,
)
from policypilot.ingestion.pdf.pdf_loader import (
    load_pdf_document,
)
from policypilot.ingestion.pdf.pdf_section_parser import (
    parse_pdf_sections,
)


PDF_FILE = (
    RAW_DATA_DIR
    / "pdf"
    / "vvg_infov.pdf"
)

DEBUG_OUTPUT_FILE = (
    PROCESSED_DATA_DIR
    / "debug"
    / "vvg_infov_sections.md"
)

EXPECTED_SECTION_NUMBERS = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
]


def main() -> None:
    """
    Test the complete PDF section-parsing pipeline.
    """

    print("=" * 70)
    print("PolicyPilot PDF Section Parser Test")
    print("=" * 70)

    print()
    print(f"PDF file: {PDF_FILE.resolve()}")

    page_documents = load_pdf_document(
        file_path=PDF_FILE,
        show_progress=True,
    )

    print(
        f"Loaded PDF pages: "
        f"{len(page_documents)}"
    )

    cleaned_page_documents = clean_pdf_page_documents(
        page_documents=page_documents,
    )

    print(
        f"Cleaned PDF pages: "
        f"{len(cleaned_page_documents)}"
    )

    assembled_document = assemble_pdf_document(
        page_documents=cleaned_page_documents,
    )

    print(
        "Assembled document characters: "
        f"{len(assembled_document.page_content)}"
    )

    section_documents = parse_pdf_sections(
        assembled_document=assembled_document,
    )

    preamble_documents = [
        document
        for document in section_documents
        if document.metadata.get("content_type")
        == "preamble"
    ]

    parsed_sections = [
        document
        for document in section_documents
        if document.metadata.get("content_type")
        == "section"
    ]

    print()
    print("Section parsing summary")
    print("-" * 70)
    print(
        f"Total parsed blocks: "
        f"{len(section_documents)}"
    )
    print(
        f"Preamble blocks: "
        f"{len(preamble_documents)}"
    )
    print(
        f"Detected sections: "
        f"{len(parsed_sections)}"
    )

    _validate_section_results(
        section_documents=section_documents,
        preamble_documents=preamble_documents,
        parsed_sections=parsed_sections,
    )

    _write_debug_output(
        section_documents=section_documents,
        output_file=DEBUG_OUTPUT_FILE,
    )

    print()
    print("=" * 70)
    print("Parsed Block Preview")
    print("=" * 70)

    for document_index, document in enumerate(
        section_documents,
        start=1,
    ):
        metadata = document.metadata

        print()
        print("-" * 70)
        print(
            f"Block: "
            f"{document_index}/{len(section_documents)}"
        )
        print(
            f"Content type: "
            f"{metadata.get('content_type')}"
        )
        print(
            f"Section number: "
            f"{metadata.get('section_number')}"
        )
        print(
            f"Section title: "
            f"{metadata.get('section_title')}"
        )
        print(
            f"Page range: "
            f"{metadata.get('page_start')} "
            f"to {metadata.get('page_end')}"
        )
        print(
            f"Characters: "
            f"{len(document.page_content)}"
        )
        print(
            f"Section ID: "
            f"{metadata.get('section_id')}"
        )

        preview = document.page_content[:300]

        if len(document.page_content) > 300:
            preview = f"{preview}..."

        print("Content preview:")
        print(preview)

    print()
    print(f"Debug output: {DEBUG_OUTPUT_FILE.resolve()}")

    print()
    print("=" * 70)
    print("PDF section parser test completed successfully.")
    print("=" * 70)


def _validate_section_results(
    *,
    section_documents: list,
    preamble_documents: list,
    parsed_sections: list,
) -> None:
    """
    Validate the section parser result for the sample PDF.
    """

    if not section_documents:
        raise AssertionError(
            "The section parser returned no Documents."
        )

    if len(preamble_documents) != 1:
        raise AssertionError(
            "Expected exactly one preamble Document, "
            f"but received {len(preamble_documents)}."
        )

    actual_section_numbers = [
        document.metadata.get("section_number")
        for document in parsed_sections
    ]

    if actual_section_numbers != EXPECTED_SECTION_NUMBERS:
        raise AssertionError(
            "Unexpected section numbers.\n"
            f"Expected: {EXPECTED_SECTION_NUMBERS}\n"
            f"Actual: {actual_section_numbers}"
        )

    for document_index, document in enumerate(
        section_documents,
        start=1,
    ):
        if not document.page_content.strip():
            raise AssertionError(
                f"Parsed block {document_index} has empty content."
            )

        if not document.metadata.get("section_id"):
            raise AssertionError(
                f"Parsed block {document_index} has no section_id."
            )

    print()
    print("Validation results")
    print("-" * 70)
    print("Preamble validation: passed")
    print("Section-number validation: passed")
    print("Non-empty content validation: passed")
    print("Section-ID validation: passed")


def _write_debug_output(
    *,
    section_documents: list,
    output_file,
) -> None:
    """
    Write parsed sections and metadata to a Markdown debug file.
    """

    output_parts: list[str] = []

    for document_index, document in enumerate(
        section_documents,
        start=1,
    ):
        metadata_json = json.dumps(
            document.metadata,
            indent=2,
            ensure_ascii=False,
        )

        output_part = (
            f"# Parsed Block {document_index}\n\n"
            f"## Metadata\n\n"
            f"```json\n"
            f"{metadata_json}\n"
            f"```\n\n"
            f"## Content\n\n"
            f"{document.page_content}\n"
        )

        output_parts.append(output_part)

    output_content = "\n\n---\n\n".join(
        output_parts
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file.write_text(
        output_content,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()