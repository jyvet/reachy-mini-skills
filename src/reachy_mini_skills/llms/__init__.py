"""LLM providers for Reachy Mini Library.

Usage:
    from reachy_mini_skills.llms import cerebras, ollama, huggingface
    
    llm = cerebras.create()
    response = await llm.complete(session, messages)
"""

from .base import LLMProvider, LLMResult
from .ollama import OllamaLLM
from .cerebras import CerebrasLLM
from .huggingface import HuggingFaceLLM
from . import ollama
from . import cerebras
from . import huggingface

__all__ = [
    "LLMProvider",
    "LLMResult",
    "OllamaLLM",
    "CerebrasLLM",
    "HuggingFaceLLM",
    "ollama",
    "cerebras",
    "huggingface",
    "get_llm_provider",
]


def get_llm_provider(provider_name: str, config) -> LLMProvider:
    """Factory function to get LLM provider by name."""
    providers = {
        "ollama": OllamaLLM,
        "cerebras": CerebrasLLM,
        "huggingface": HuggingFaceLLM,
    }
    
    provider_class = providers.get(provider_name.lower())
    if provider_class is None:
        raise ValueError(f"Unknown LLM provider: {provider_name}. Available: {list(providers.keys())}")
    
    return provider_class(config)
