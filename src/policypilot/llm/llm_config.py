from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMConfig:
    """
    Configuration for the PolicyPilot language model.

    The LLM receives:
    - The user's question
    - Relevant policy chunks retrieved from ChromaDB

    It then generates the final answer.
    """

    model_name: str = "llama3.2:1b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    max_output_tokens: int = 512
    validate_model_on_init: bool = True

    def __post_init__(self) -> None:
        """
        Validate the LLM configuration after initialization.

        Raises:
            TypeError:
                If a configuration value has an incorrect type.

            ValueError:
                If a configuration value is empty or outside
                the allowed range.
        """

        if not isinstance(self.model_name, str):
            raise TypeError(
                "LLM model name must be a string."
            )

        if not self.model_name.strip():
            raise ValueError(
                "LLM model name must not be empty."
            )

        if not isinstance(self.base_url, str):
            raise TypeError(
                "Ollama base URL must be a string."
            )

        if not self.base_url.strip():
            raise ValueError(
                "Ollama base URL must not be empty."
            )

        if not isinstance(self.temperature, (int, float)):
            raise TypeError(
                "LLM temperature must be a number."
            )

        if not 0.0 <= self.temperature <= 1.0:
            raise ValueError(
                "LLM temperature must be between 0.0 and 1.0."
            )

        if not isinstance(self.max_output_tokens, int):
            raise TypeError(
                "Maximum output tokens must be an integer."
            )

        if self.max_output_tokens <= 0:
            raise ValueError(
                "Maximum output tokens must be greater than zero."
            )

        if not isinstance(
            self.validate_model_on_init,
            bool,
        ):
            raise TypeError(
                "validate_model_on_init must be a boolean."
            )