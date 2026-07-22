from __future__ import annotations

from langchain_core.documents import Document


def assemble_pdf_document(
    page_documents: list[Document],
) -> Document:
    """
    Combine all page-level PDF Documents into one Document.

    Page markers are added so that physical page information
    remains available for later processing.
    """

    if not page_documents:
        raise ValueError("page_documents cannot be empty.")

    ordered_documents = sorted(
        page_documents,
        key=lambda document: document.metadata["page_number"],
    )

    assembled_parts: list[str] = []

    for document in ordered_documents:
        page_number = document.metadata["page_number"]

        page_content = (
            f"<!-- PDF_PAGE_START: {page_number} -->\n\n"
            f"{document.page_content.strip()}\n\n"
            f"<!-- PDF_PAGE_END: {page_number} -->"
        )

        assembled_parts.append(page_content)

    assembled_content = "\n\n".join(assembled_parts)

    first_document = ordered_documents[0]
    last_document = ordered_documents[-1]

    metadata = {
        "source": first_document.metadata["source"],
        "file_name": first_document.metadata["file_name"],
        "file_type": "pdf",
        "document_type": first_document.metadata["document_type"],
        "page_start": first_document.metadata["page_number"],
        "page_end": last_document.metadata["page_number"],
        "total_pages": len(ordered_documents),
        "extraction_method": first_document.metadata["extraction_method"],
        "assembly_method": "page_marker_join",
    }

    return Document(
        page_content=assembled_content,
        metadata=metadata,
    )
