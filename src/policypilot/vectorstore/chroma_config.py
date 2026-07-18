from pathlib import Path

from policypilot.ingestion.config import DATA_DIR


CHROMA_PERSIST_DIRECTORY: Path = (
    DATA_DIR / "vectorstore" / "chroma"
)

CHROMA_COLLECTION_NAME = "policypilot_policy_chunks"

CHROMA_DISTANCE_METRIC = "cosine"

CHROMA_WRITE_BATCH_SIZE = 100