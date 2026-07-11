from pathlib import Path

from policypilot.ingestion.loaders.markdown_loader import load_markdown_document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "sample_policy.md"


def main() -> None:
    print(f"Loading Markdown document: {INPUT_FILE}")

    documents = load_markdown_document(INPUT_FILE)

    print(f"\nNumber of documents loaded: {len(documents)}")

    for index, document in enumerate(documents, start=1):
        print("\n" + "=" * 60)
        print(f"Document number: {index}")
        print("=" * 60)

        print("\nObject type:")
        print(type(document))

        print("\nMetadata:")
        print(document.metadata)

        print("\nPage content:")
        print(document.page_content)


if __name__ == "__main__":
    main()