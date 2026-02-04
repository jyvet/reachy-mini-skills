"""Cerebras LLM provider.

Usage:
    from reachy_mini_skills.llms import cerebras
    
    # With defaults
    llm = cerebras.create()
    
    # With custom parameters
    llm = cerebras.create(
        stream=True,
        temperature=0.5,
        max_completion_tokens=2048,
        top_p=0.9
    )
    
    response = await llm.complete(session, messages)
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, TYPE_CHECKING

import aiohttp

from .base import LLMProvider, LLMResult
from ..config import LLMConfig

if TYPE_CHECKING:
    from ..config import Config

__all__ = ["CerebrasLLM", "CerebrasParams", "LLMResult", "create"]


@dataclass
class CerebrasParams:
    """Parameters for Cerebras LLM generation."""
    stream: Optional[bool] = None  # None means use config default
    temperature: float = 0.7
    max_completion_tokens: int = 1024
    top_p: float = 1.0


class CerebrasLLM(LLMProvider):
    """Cerebras cloud LLM provider."""
    
    def __init__(self, config: LLMConfig, params: Optional[CerebrasParams] = None):
        super().__init__(config)
        self.params = params or CerebrasParams()
    
    @property
    def name(self) -> str:
        return "cerebras"
    
    @property
    def default_model(self) -> str:
        return self.config.cerebras_default_model
    
    @property
    def use_streaming(self) -> bool:
        """Determine if streaming should be used."""
        if self.params.stream is not None:
            return self.params.stream
        return self.config.cerebras_streaming
    
    async def complete(
        self,
        session: aiohttp.ClientSession,
        conversation: List[Dict[str, str]],
        model: Optional[str] = None
    ) -> LLMResult:
        """Generate completion using Cerebras."""
        from cerebras.cloud.sdk import Cerebras
        
        start = time.perf_counter()
        
        client = Cerebras(api_key=self.config.cerebras_api_key)
        model_name = model or self.default_model
        
        if self.use_streaming:
            def stream_cerebras():
                chunks = []
                stream = client.chat.completions.create(
                    messages=conversation,
                    model=model_name,
                    stream=True,
                    max_completion_tokens=self.params.max_completion_tokens,
                    temperature=self.params.temperature,
                    top_p=self.params.top_p,
                )
                
                for chunk in stream:
                    content = chunk.choices[0].delta.content or ""
                    if content:
                        chunks.append(content)
                        print(content, end="", flush=True)
                
                print()
                return "".join(chunks)
            
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(None, stream_cerebras)
        else:
            def call_cerebras():
                response = client.chat.completions.create(
                    messages=conversation,
                    model=model_name,
                    stream=False,
                    max_completion_tokens=self.params.max_completion_tokens,
                    temperature=self.params.temperature,
                    top_p=self.params.top_p,
                )
                
                text = response.choices[0].message.content
                print(text)
                return text
            
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(None, call_cerebras)

        latency = time.perf_counter() - start
        return LLMResult(text=text, latency=latency)


def create(
    config: "Config" = None,
    *,
    stream: Optional[bool] = None,
    temperature: float = 0.7,
    max_completion_tokens: int = 1024,
    top_p: float = 1.0,
) -> CerebrasLLM:
    """Create a Cerebras LLM instance.
    
    Args:
        config: Configuration object. If None, uses default Config.
        stream: Whether to stream responses. None uses config default.
        temperature: Sampling temperature (0.0-2.0). Default 0.7.
        max_completion_tokens: Maximum tokens to generate. Default 1024.
        top_p: Top-p (nucleus) sampling. Default 1.0.
    
    Returns:
        Configured CerebrasLLM instance.
    """
    from ..config import Config
    if config is None:
        config = Config()
    
    params = CerebrasParams(
        stream=stream,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        top_p=top_p,
    )
    return CerebrasLLM(config.llm, params)
