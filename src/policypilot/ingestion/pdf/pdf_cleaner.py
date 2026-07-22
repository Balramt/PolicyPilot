from __future__ import annotations

import re

from langchain_core.documents import Document


INLINE_HTML_TAG_PATTERN = re.compile(
    r"</?(mark|u)>",
    re.IGNORECASE,
)


def clean_pdf_page_documents(
    page_documents: list[Document],
) -> list[Document]:
    """
    Clean formatting artifacts from PDF page Documents.

    Each PDF page remains a separate LangChain Document.
    The existing metadata is preserved.
    """

    if not page_documents:
        raise ValueError("page_documents cannot be empty.")

    cleaned_documents: list[Document] = []

    for document in page_documents:
        cleaned_content = clean_pdf_page_content(document.page_content)

        cleaned_metadata = {
            **document.metadata,
            "cleaning_method": "basic_formatting_cleanup",
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
    Remove basic formatting artifacts from one PDF page.

    The visible text is preserved while unnecessary formatting
    symbols and tags are removed.
    """

    cleaned_content = INLINE_HTML_TAG_PATTERN.sub(
        "",
        page_content,
    )

    cleaned_content = cleaned_content.replace(
        "~~",
        "",
    )

    cleaned_content = re.sub(
        r"[ \t]+\n",
        "\n",
        cleaned_content,
    )

    cleaned_content = re.sub(
        r"\n{3,}",
        "\n\n",
        cleaned_content,
    )

    return cleaned_content.strip()
