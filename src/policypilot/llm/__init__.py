"""
Language model components for PolicyPilot.

This package contains the LLM configuration and the function
used to create the local Ollama client.
"""

from .llm_client import create_llm_client
from .llm_config import LLMConfig

__all__ = [
    "LLMConfig",
    "create_llm_client",
]