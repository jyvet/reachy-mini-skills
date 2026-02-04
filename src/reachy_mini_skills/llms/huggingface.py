"""Hugging Face LLM provider.

Usage:
    from reachy_mini_skills.llms import huggingface
    
    # With defaults
    llm = huggingface.create()
    
    # With custom parameters
    llm = huggingface.create(
        temperature=0.5,
        max_tokens=2048,
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

__all__ = ["HuggingFaceLLM", "HuggingFaceParams", "LLMResult", "create"]


@dataclass
class HuggingFaceParams:
    """Parameters for Hugging Face LLM generation."""
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 0.95
    repetition_penalty: float = 1.0


class HuggingFaceLLM(LLMProvider):
    """Hugging Face Inference API LLM provider."""
    
    def __init__(self, config: LLMConfig, params: Optional[HuggingFaceParams] = None):
        super().__init__(config)
        self.params = params or HuggingFaceParams()
    
    @property
    def name(self) -> str:
        return "huggingface"
    
    @property
    def default_model(self) -> str:
        return self.config.huggingface_default_model
    
    @property
    def api_url(self) -> str:
        """Get the Hugging Face API URL."""
        return self.config.huggingface_api_url
    
    async def complete(
        self,
        session: aiohttp.ClientSession,
        conversation: List[Dict[str, str]],
        model: Optional[str] = None
    ) -> LLMResult:
        """Generate completion using Hugging Face Inference API.
        
        Supports both the serverless Inference API and dedicated Inference Endpoints.
        """
        start = time.perf_counter()
        
        model_name = model or self.default_model
        
        # Build the API URL - supports both serverless and dedicated endpoints
        if self.api_url:
            # Use custom endpoint (dedicated inference endpoint)
            url = self.api_url
        else:
            # Use router API (OpenAI-compatible endpoint)
            url = "https://router.huggingface.co/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.config.huggingface_api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": model_name,
            "messages": conversation,
            "max_tokens": self.params.max_tokens,
            "temperature": self.params.temperature,
            "top_p": self.params.top_p,
        }
        
        # Add repetition penalty if not default
        if self.params.repetition_penalty != 1.0:
            payload["repetition_penalty"] = self.params.repetition_penalty
        
        try:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Hugging Face API error ({response.status}): {error_text}")
                
                data = await response.json()
                
                # Extract the response text from the OpenAI-compatible format
                if "choices" in data and len(data["choices"]) > 0:
                    text = data["choices"][0]["message"]["content"]
                else:
                    # Fallback for older API format
                    text = data.get("generated_text", str(data))
                
                print(text)
                
        except aiohttp.ClientError as e:
            raise Exception(f"Network error calling Hugging Face API: {e}")
        
        latency = time.perf_counter() - start
        return LLMResult(text=text, latency=latency)


def create(
    config: "Config" = None,
    *,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    top_p: float = 0.95,
    repetition_penalty: float = 1.0,
) -> HuggingFaceLLM:
    """Create a Hugging Face LLM instance.
    
    Args:
        config: Configuration object. If None, uses default Config.
        temperature: Sampling temperature (0.0-2.0). Default 0.7.
        max_tokens: Maximum tokens to generate. Default 1024.
        top_p: Top-p (nucleus) sampling. Default 0.95.
        repetition_penalty: Repetition penalty. Default 1.0 (disabled).
    
    Returns:
        Configured HuggingFaceLLM instance.
    """
    from ..config import Config
    if config is None:
        config = Config()
    
    params = HuggingFaceParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    return HuggingFaceLLM(config.llm, params)
