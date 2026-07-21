from __future__ import annotations

from policypilot.ingestion.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from policypilot.ingestion.pdf.pdf_loader import (
    load_pdf_document,
)
from policypilot.ingestion.pdf.pdf_assembler import (
    assemble_pdf_document,
)
from policypilot.ingestion.pdf.pdf_cleaner import (
    clean_pdf_page_documents,
)


PDF_FILE = (
    RAW_DATA_DIR
    / "pdf"
    / "vvg_infov.pdf"
)

DEBUG_OUTPUT_FILE = (
    PROCESSED_DATA_DIR
    / "debug"
    / "vvg_infov_cleaned_assembled.md"
)


def main() -> None:
    """
    Load, clean, and assemble the sample PDF.
    """

    print("=" * 70)
    print("PolicyPilot PDF Cleaner Test")
    print("=" * 70)

    page_documents = load_pdf_document(
        file_path=PDF_FILE,
        show_progress=True,
    )

    cleaned_page_documents = clean_pdf_page_documents(
        page_documents=page_documents,
    )

    assembled_document = assemble_pdf_document(
        page_documents=cleaned_page_documents,
    )

    DEBUG_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    DEBUG_OUTPUT_FILE.write_text(
        assembled_document.page_content,
        encoding="utf-8",
    )

    original_character_count = sum(
        len(document.page_content)
        for document in page_documents
    )

    cleaned_character_count = sum(
        len(document.page_content)
        for document in cleaned_page_documents
    )

    removed_character_count = (
        original_character_count
        - cleaned_character_count
    )

    print()
    print("Cleaning summary")
    print("----------------")
    print(f"PDF pages: {len(page_documents)}")
    print(
        f"Original characters: "
        f"{original_character_count}"
    )
    print(
        f"Cleaned characters: "
        f"{cleaned_character_count}"
    )
    print(
        f"Removed characters: "
        f"{removed_character_count}"
    )
    print(
        f"Output file: "
        f"{DEBUG_OUTPUT_FILE.resolve()}"
    )


if __name__ == "__main__":
    main()