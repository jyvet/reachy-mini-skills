"""Base TTS provider interface."""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from ...config import TTSConfig


@dataclass
class TTSResult:
    """Result from text-to-speech synthesis."""
    ttfb: float  # Time to first byte
    total_time: float
    was_interrupted: bool = False


class TTSProvider(ABC):
    """Abstract base class for TTS providers."""
    
    def __init__(self, config: TTSConfig):
        self.config = config
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass
    
    @abstractmethod
    async def synthesize(
        self,
        text: str,
        audio_queue: Optional[asyncio.Queue] = None,
        interrupt_event: Optional[asyncio.Event] = None,
        speaker_id: Optional[int] = None,
        emotion: Optional[str] = None
    ) -> TTSResult:
        """
        Synthesize text to speech and play it.
        
        Args:
            text: Text to synthesize
            audio_queue: Optional queue for barge-in detection
            interrupt_event: Optional event to signal barge-in
            speaker_id: Optional audio output device ID
            emotion: Optional emotion for TTS
            
        Returns:
            TTSResult with timing info
        """
        pass
