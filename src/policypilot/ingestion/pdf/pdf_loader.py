from __future__ import annotations

from pathlib import Path
from typing import Any

import pymupdf4llm
from langchain_core.documents import Document


def load_pdf_document(
    file_path: str | Path,
    *,
    show_progress: bool = False,
) -> list[Document]:
    """
    Load one text-based PDF using PyMuPDF4LLM.

    The PDF is extracted page by page as Markdown. Each PDF page
    becomes one LangChain Document for the initial baseline test.

    OCR is explicitly disabled. Image-only and scanned PDFs are
    not supported at this stage.

    Args:
        file_path:
            Path to the PDF file.

        show_progress:
            Whether PyMuPDF4LLM should display extraction progress.

    Returns:
        One LangChain Document for each physical PDF page.

    Raises:
        TypeError:
            If file_path has an unsupported type or show_progress
            is not a boolean.

        FileNotFoundError:
            If the PDF file does not exist.

        ValueError:
            If the path is not a file, is not a PDF, or no pages
            are returned.

        RuntimeError:
            If PyMuPDF4LLM returns an unexpected result or PDF
            extraction fails.
    """

    if not isinstance(file_path, (str, Path)):
        raise TypeError(
            "file_path must be a string or Path object, "
            f"but received {type(file_path).__name__}."
        )

    if not isinstance(show_progress, bool):
        raise TypeError(
            "show_progress must be a boolean, "
            f"but received {type(show_progress).__name__}."
        )

    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"PDF file was not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Expected a PDF file, but received: {path}"
        )

    if path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected a PDF file, but received: {path.suffix}"
        )

    try:
        page_chunks = pymupdf4llm.to_markdown(
            str(path),
            page_chunks=True,
            use_ocr=False,
            force_ocr=False,
            write_images=False,
            embed_images=False,
            show_progress=show_progress,
        )
    except Exception as error:
        raise RuntimeError(
            f"Failed to extract PDF content from: {path}"
        ) from error

    if not isinstance(page_chunks, list):
        raise RuntimeError(
            "Expected PyMuPDF4LLM to return a list of "
            "page dictionaries."
        )

    if not page_chunks:
        raise ValueError(
            f"No PDF pages were extracted from: {path}"
        )

    documents: list[Document] = []

    for page_index, page_chunk in enumerate(
        page_chunks,
        start=1,
    ):
        document = _create_page_document(
            page_chunk=page_chunk,
            fallback_page_number=page_index,
            fallback_total_pages=len(page_chunks),
            source_path=path,
        )

        documents.append(document)

    return documents


def _create_page_document(
    *,
    page_chunk: dict[str, Any],
    fallback_page_number: int,
    fallback_total_pages: int,
    source_path: Path,
) -> Document:
    """
    Convert one PyMuPDF4LLM page result into a LangChain Document.

    Args:
        page_chunk:
            One page dictionary returned by PyMuPDF4LLM.

        fallback_page_number:
            Page number used when PyMuPDF4LLM metadata does not
            contain a valid page number.

        fallback_total_pages:
            Total page count used when PyMuPDF4LLM metadata does
            not contain the page count.

        source_path:
            Resolved path to the original PDF.

    Returns:
        A LangChain Document containing one page of Markdown.

    Raises:
        RuntimeError:
            If the page result has an invalid structure.
    """

    if not isinstance(page_chunk, dict):
        raise RuntimeError(
            "Each PyMuPDF4LLM page result must be a dictionary."
        )

    page_content = page_chunk.get("text")

    if not isinstance(page_content, str):
        raise RuntimeError(
            f"PDF page {fallback_page_number} does not contain "
            "valid string content."
        )

    raw_metadata = page_chunk.get("metadata", {})

    if raw_metadata is None:
        raw_metadata = {}

    if not isinstance(raw_metadata, dict):
        raise RuntimeError(
            f"PDF page {fallback_page_number} contains "
            "invalid metadata."
        )

    page_number = raw_metadata.get(
        "page_number",
        fallback_page_number,
    )

    total_pages = raw_metadata.get(
        "page_count",
        fallback_total_pages,
    )

    metadata = {
        "source": str(source_path),
        "file_name": source_path.name,
        "file_type": "pdf",
        "document_type": "insurance_policy",
        "page_number": page_number,
        "total_pages": total_pages,
        "extraction_method": "pymupdf4llm",
    }

    return Document(
        page_content=page_content,
        metadata=metadata,
    )