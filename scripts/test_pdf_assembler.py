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


PDF_FILE = (
    RAW_DATA_DIR
    / "pdf"
    / "vvg_infov.pdf"
)

DEBUG_OUTPUT_FILE = (
    PROCESSED_DATA_DIR
    / "debug"
    / "vvg_infov_assembled.md"
)


def main() -> None:
    """
    Load the PDF and combine all page Documents.
    """

    print("=" * 70)
    print("PolicyPilot PDF Assembler Test")
    print("=" * 70)

    page_documents = load_pdf_document(
        file_path=PDF_FILE,
        show_progress=True,
    )

    assembled_document = assemble_pdf_document(
        page_documents=page_documents,
    )

    DEBUG_OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    DEBUG_OUTPUT_FILE.write_text(
        assembled_document.page_content,
        encoding="utf-8",
    )

    print()
    print(f"Input pages: {len(page_documents)}")
    print(
        "Assembled characters: "
        f"{len(assembled_document.page_content)}"
    )
    print(f"Metadata: {assembled_document.metadata}")
    print(f"Output file: {DEBUG_OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()