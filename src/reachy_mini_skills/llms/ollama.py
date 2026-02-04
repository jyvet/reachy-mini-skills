"""Ollama LLM provider.

Usage:
    from reachy_mini_skills.llms import ollama
    
    # With defaults
    llm = ollama.create()
    
    # With custom parameters
    llm = ollama.create(
        stream=False,
        temperature=0.5,
        num_predict=2048,
        top_p=0.9
    )
    
    response = await llm.complete(session, messages)
"""

import time
from dataclasses import dataclass
from typing import List, Dict, Optional, TYPE_CHECKING

import aiohttp

from .base import LLMProvider, LLMResult
from ..config import LLMConfig

if TYPE_CHECKING:
    from ..config import Config

__all__ = ["OllamaLLM", "OllamaParams", "LLMResult", "create"]


@dataclass
class OllamaParams:
    """Parameters for Ollama LLM generation."""
    stream: bool = False
    temperature: float = 0.7
    num_predict: int = 1024  # Ollama uses num_predict instead of max_completion_tokens
    top_p: float = 1.0


class OllamaLLM(LLMProvider):
    """Ollama local LLM provider."""
    
    def __init__(self, config: LLMConfig, params: Optional[OllamaParams] = None):
        super().__init__(config)
        self.params = params or OllamaParams()
    
    @property
    def name(self) -> str:
        return "ollama"
    
    @property
    def default_model(self) -> str:
        return self.config.ollama_default_model
    
    async def complete(
        self,
        session: aiohttp.ClientSession,
        conversation: List[Dict[str, str]],
        model: Optional[str] = None
    ) -> LLMResult:
        """Generate completion using Ollama."""
        start = time.perf_counter()
        
        payload = {
            "model": model or self.default_model,
            "messages": conversation,
            "stream": self.params.stream,
            "options": {
                "temperature": self.params.temperature,
                "num_predict": self.params.num_predict,
                "top_p": self.params.top_p,
            }
        }

        try:
            async with session.post(self.config.ollama_url, json=payload) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    latency = time.perf_counter() - start
                    return LLMResult(
                        text=f"Error: Ollama returned status {resp.status}: {error_text}",
                        latency=latency
                    )
                
                data = await resp.json()
                
                # Check if response has expected structure
                if "message" not in data or "content" not in data.get("message", {}):
                    latency = time.perf_counter() - start
                    error_msg = data.get("error", "Unknown error - unexpected response format")
                    return LLMResult(text=f"Error: {error_msg}", latency=latency)

                latency = time.perf_counter() - start
                text = data["message"]["content"]
                
                return LLMResult(text=text, latency=latency)
        except aiohttp.ClientError as e:
            latency = time.perf_counter() - start
            return LLMResult(
                text=f"Error: Could not connect to Ollama at {self.config.ollama_url}: {str(e)}",
                latency=latency
            )


def create(
    config: "Config" = None,
    *,
    stream: bool = False,
    temperature: float = 0.7,
    num_predict: int = 1024,
    top_p: float = 1.0,
) -> OllamaLLM:
    """Create an Ollama LLM instance.
    
    Args:
        config: Configuration object. If None, uses default Config.
        stream: Whether to stream responses. Default False.
        temperature: Sampling temperature (0.0-2.0). Default 0.7.
        num_predict: Maximum tokens to generate. Default 1024.
        top_p: Top-p (nucleus) sampling. Default 1.0.
    
    Returns:
        Configured OllamaLLM instance.
    """
    from ..config import Config
    if config is None:
        config = Config()
    
    params = OllamaParams(
        stream=stream,
        temperature=temperature,
        num_predict=num_predict,
        top_p=top_p,
    )
    return OllamaLLM(config.llm, params)
