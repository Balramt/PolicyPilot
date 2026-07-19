from __future__ import annotations

from langchain_core.messages import AIMessage

from policypilot.llm.llm_client import create_llm_client
from policypilot.llm.llm_config_old import LLMConfig


TEST_PROMPT = (
    "Answer in one short sentence: "
    "What is a policy document?"
)


def main() -> None:
    """
    Create the PolicyPilot Ollama client and test one simple prompt.

    This script only tests the LLM connection. It does not use
    ChromaDB, document retrieval, or a RAG prompt.
    """

    config = LLMConfig()

    print("=" * 60)
    print("PolicyPilot LLM Client Test")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Base URL: {config.base_url}")
    print(f"Temperature: {config.temperature}")
    print(f"Maximum output tokens: {config.max_output_tokens}")
    print()

    try:
        print("Creating Ollama LLM client...")

        llm = create_llm_client(config)

        print("LLM client created successfully.")
        print()
        print(f"Test prompt: {TEST_PROMPT}")
        print()
        print("Sending prompt to Ollama...")

        response = llm.invoke(TEST_PROMPT)

    except Exception as exc:
        raise RuntimeError(
            "The LLM test failed. Make sure Ollama is running "
            f"and the model '{config.model_name}' is installed."
        ) from exc

    if not isinstance(response, AIMessage):
        raise TypeError(
            "Expected an AIMessage response, "
            f"but received {type(response).__name__}."
        )

    if not isinstance(response.content, str):
        raise TypeError(
            "Expected the response content to be a string, "
            f"but received {type(response.content).__name__}."
        )

    answer = response.content.strip()

    if not answer:
        raise ValueError("Ollama returned an empty response.")

    print()
    print("=" * 60)
    print("LLM Response")
    print("=" * 60 )
    print(answer)
    print()
    print("LLM client test completed successfully.")


if __name__ == "__main__":
    main()