from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter



def split_documents_recursively(
        documents: list[Document],
        chunk_size: int = 500,   # in characters
        chunk_overlap: int = 100,

) -> list[Document]:
    """
    Split LangChain documents into smaller overlapping chunks using
    RecursiveCharacterTextSplitter.

    Args:
        documents: LangChain documents that need to be split.
        chunk_size: Maximum approximate number of characters in each chunk.
        chunk_overlap: Number of characters shared between consecutive chunks.

    Returns:
        A list of smaller LangChain Document objects.

    Raises:
        ValueError: If the document list or chunk settings are invalid.
    """
    if not documents:
        raise ValueError("No documents were provided for chunking.")
    
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)

    for chunk_index, chunk in enumerate(chunks):
        file_name = chunk.metadata.get(
            "file_name",
            "unknown_document",
        )

        
        chunk.metadata["chunk_index"] = chunk_index
        chunk.metadata["chunk_id"] = (
            f"{file_name}-chunk-{chunk_index:04d}"
        )

    return chunks

