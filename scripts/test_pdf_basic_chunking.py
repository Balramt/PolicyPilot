from __future__ import annotations

from policypilot.ingestion.chunkers.recursive_character_chunker import (
    split_documents_recursively,
)
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
from policypilot.ingestion.writers.jsonl_writer import (
    write_documents_to_jsonl,
)


PDF_FILE = (
    RAW_DATA_DIR
    / "pdf"
    / "vvg_infov.pdf"
)

OUTPUT_FILE = (
    PROCESSED_DATA_DIR
    / "debug"
    / "vvg_infov_basic_chunks.jsonl"
)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
PREVIEW_LENGTH = 300


def main() -> None:
    """
    Test the basic PDF ingestion and recursive chunking pipeline.
    """

    print("=" * 70)
    print("PolicyPilot PDF Basic Chunking Test")
    print("=" * 70)

    print()
    print(f"PDF file: {PDF_FILE.resolve()}")

    page_documents = load_pdf_document(
        file_path=PDF_FILE,
        show_progress=True,
    )

    print(f"Loaded PDF pages: {len(page_documents)}")

    cleaned_page_documents = clean_pdf_page_documents(
        page_documents=page_documents,
    )

    print(
        "Cleaned PDF pages: "
        f"{len(cleaned_page_documents)}"
    )

    assembled_document = assemble_pdf_document(
        page_documents=cleaned_page_documents,
    )

    print(
        "Assembled document characters: "
        f"{len(assembled_document.page_content)}"
    )

    chunks = split_documents_recursively(
        documents=[assembled_document],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    print(f"Generated chunks: {len(chunks)}")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_documents_to_jsonl(
        chunks,
        OUTPUT_FILE,
    )

    print(f"JSONL output: {OUTPUT_FILE.resolve()}")

    print()
    print("=" * 70)
    print("Chunk Preview")
    print("=" * 70)

    for chunk_index, chunk in enumerate(
        chunks,
        start=1,
    ):
        preview = chunk.page_content[:PREVIEW_LENGTH]

        if len(chunk.page_content) > PREVIEW_LENGTH:
            preview = f"{preview}..."

        print()
        print("-" * 70)
        print(f"Chunk: {chunk_index}/{len(chunks)}")
        print(f"Characters: {len(chunk.page_content)}")
        print(f"Metadata: {chunk.metadata}")
        print("Content:")
        print(preview)

    print()
    print("=" * 70)
    print("Basic PDF chunking test completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()