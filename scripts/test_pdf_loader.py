from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from policypilot.ingestion.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from policypilot.ingestion.pdf.pdf_loader import (
    load_pdf_document,
)


PDF_FILE = (
    RAW_DATA_DIR
    / "pdf"
    / "vvg_infov.pdf"
)

DEBUG_OUTPUT_FILE = (
    PROCESSED_DATA_DIR
    / "debug"
    / "vvg_infov_extracted.md"
)


def write_debug_markdown(
    documents: list[Document],
    output_file: Path,
) -> None:
    """
    Save extracted page Markdown for manual inspection.

    This is only a development/debugging file. It is not the
    final PolicyPilot chunk output.
    """

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        for document in documents:
            page_number = document.metadata["page_number"]

            file.write(
                f"\n\n<!-- PDF PAGE {page_number} -->\n\n"
            )

            file.write(document.page_content)

            file.write("\n")


def main() -> None:
    """
    Load the sample PDF and inspect the extraction result.
    """

    print("=" * 70)
    print("PolicyPilot PDF Loader Test")
    print("=" * 70)
    print(f"PDF file: {PDF_FILE.resolve()}")

    documents = load_pdf_document(
        file_path=PDF_FILE,
        show_progress=True,
    )

    empty_page_numbers = [
        document.metadata["page_number"]
        for document in documents
        if not document.page_content.strip()
    ]

    print()
    print("Extraction summary")
    print("------------------")
    print(f"Loaded page documents: {len(documents)}")
    print(
        "Total PDF pages: "
        f"{documents[0].metadata['total_pages']}"
    )
    print(f"Empty extracted pages: {empty_page_numbers}")

    write_debug_markdown(
        documents=documents,
        output_file=DEBUG_OUTPUT_FILE,
    )

    print(f"Debug Markdown: {DEBUG_OUTPUT_FILE.resolve()}")

    print()
    print("Page details")
    print("------------")

    for document in documents:
        page_number = document.metadata["page_number"]
        character_count = len(document.page_content)

        print()
        print("-" * 70)
        print(f"Page number: {page_number}")
        print(f"Character count: {character_count}")
        print(f"Metadata: {document.metadata}")
        print("-" * 70)

        print(document.page_content)


if __name__ == "__main__":
    main()