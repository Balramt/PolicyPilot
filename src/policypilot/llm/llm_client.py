from __future__ import annotations

from langchain_ollama import ChatOllama

from .llm_config import LLMConfig


def create_llm_client(config: LLMConfig | None = None) -> ChatOllama:
    """
    Create and return a configured ChatOllama client.

    If no configuration is provided, the default LLMConfig
    values are used.

    Args:
        config:
            Optional configuration for the Ollama language model.

    Returns:
        A configured ChatOllama client.

    Raises:
        TypeError:
            If config is not an instance of LLMConfig.
    """

    if config is None:
        config = LLMConfig()

    _validate_config(config)

    return ChatOllama(
        model=config.model_name,
        base_url=config.base_url,
        temperature=config.temperature,
        num_predict=config.max_output_tokens,
        validate_model_on_init=config.validate_model_on_init,
    )


def _validate_config(config: LLMConfig) -> None:
    """
    Check that the provided configuration is an LLMConfig object.

    Validation of individual values such as the model name,
    temperature, base URL, and maximum output tokens is handled
    inside LLMConfig.

    Args:
        config:
            The configuration object to validate.

    Raises:
        TypeError:
            If config is not an LLMConfig object.
    """

    if not isinstance(config, LLMConfig):
        raise TypeError(
            "config must be an instance of LLMConfig, "
            f"but received {type(config).__name__}."
        )