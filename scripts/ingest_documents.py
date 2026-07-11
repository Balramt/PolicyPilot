from pathlib import Path

from langchain_core.documents import Document

from policypilot.ingestion.chunkers.recursive_character_chunker import (
    split_documents_recursively,
)
from policypilot.ingestion.config import (
    DEFAULT_CHUNKS_OUTPUT_FILE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MARKDOWN_FILE,
)
from policypilot.ingestion.loaders.markdown_loader import (
    load_markdown_document,
)

from policypilot.ingestion.writers.jsonl_writer import (
    write_documents_to_jsonl,
)


def display_chunks(chunks: list[Document]) -> None:
    """
    Display chunk metadata and content in the terminal.

    Args:
        chunks: LangChain Document chunks to display.
    """

    for chunk in chunks:
        print("\n" + "=" * 60)
        print(f"Chunk ID: {chunk.metadata['chunk_id']}")
        print("=" * 60)

        print("\nMetadata:")
        print(chunk.metadata)

        print("\nChunk content:")
        print(chunk.page_content)


def run_markdown_ingestion(
    input_file: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """
    Load a Markdown document and split it into smaller chunks.

    Args:
        input_file: Path to the Markdown file.
        chunk_size: Maximum approximate number of characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        A list of chunked LangChain Document objects.
    """

    print(f"Loading Markdown document: {input_file}")

    documents = load_markdown_document(input_file)

    print(f"Number of documents loaded: {len(documents)}")

    chunks = split_documents_recursively(
        documents=documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    print(f"Number of chunks created: {len(chunks)}")

    return chunks


def main() -> None:
    """Run the ingestion pipeline using the default configuration."""

    chunks = run_markdown_ingestion(
        input_file=DEFAULT_MARKDOWN_FILE,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )

    write_documents_to_jsonl(
        documents=chunks,
        output_file=DEFAULT_CHUNKS_OUTPUT_FILE,
    )

    print(
        f"Chunks saved to: {DEFAULT_CHUNKS_OUTPUT_FILE}"
    )

    display_chunks(chunks)


if __name__ == "__main__":
    main()