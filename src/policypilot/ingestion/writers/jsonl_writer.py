import json
from pathlib import Path
from langchain_core.documents import Document

def write_documents_to_jsonl(
    documents: list[Document],
    output_file: Path,
) -> None:
    """
    Write LangChain Document objects to a JSONL file.

    Each document is stored as one JSON object on a separate line.

    Args:
        documents: LangChain Document objects to save.
        output_file: Path where the JSONL file will be created.

    Raises:
        ValueError: If the document list is empty.
    """

    if not documents:
        raise ValueError("No documents were provided for writing.")

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        for document in documents:
            record = {
                "page_content": document.page_content,
                "metadata": document.metadata,
            }

            json_line = json.dumps(
                record,
                ensure_ascii=False,
            )

            file.write(json_line + "\n")