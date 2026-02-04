"""Base LLM provider interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional

import aiohttp

from ..config import LLMConfig


@dataclass
class LLMResult:
    """Result from LLM completion."""
    text: str
    latency: float


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass
    
    @property
    @abstractmethod
    def default_model(self) -> str:
        """Default model for this provider."""
        pass
    
    @abstractmethod
    async def complete(
        self,
        session: aiohttp.ClientSession,
        conversation: List[Dict[str, str]],
        model: Optional[str] = None
    ) -> LLMResult:
        """
        Generate a completion for the conversation.
        
        Args:
            session: aiohttp session for HTTP requests
            conversation: List of message dicts with 'role' and 'content'
            model: Optional model override
            
        Returns:
            LLMResult with response text and latency
        """
        pass
