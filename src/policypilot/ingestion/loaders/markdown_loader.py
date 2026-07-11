from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document


def load_markdown_document(file_path: Path) -> list[Document]:
    """
    Load one Markdown file as LangChain Document objects.

    Args:
        file_path: Path to the Markdown file.

    Returns:
        LangChain Document objects containing the Markdown content.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a Markdown file.
    """

    if not file_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {file_path}")

    if file_path.suffix.lower() != ".md":
        raise ValueError(
            f"Expected a Markdown file, but received: {file_path.suffix}"
        )

    loader = TextLoader(
        file_path=str(file_path),
        encoding="utf-8",
    )

    documents = loader.load()

    for document in documents:
        document.metadata["file_name"] = file_path.name
        document.metadata["file_type"] = "markdown"
        document.metadata["document_type"] = "insurance_policy"

    return documents