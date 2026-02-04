"""Base STT provider interface."""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from ...config import STTConfig


# Sentence-ending punctuation for early termination
SENTENCE_ENDERS = {'.', '!', '?'}


def has_sentence_ending(text: str) -> bool:
    """Check if text ends with sentence-ending punctuation."""
    if not text:
        return False
    stripped = text.strip()
    return len(stripped) > 0 and stripped[-1] in SENTENCE_ENDERS


@dataclass
class STTResult:
    """Result from speech-to-text transcription."""
    text: str
    ttfb: float  # Time to first byte/word
    total_time: float
    was_interrupted: bool = False


class STTProvider(ABC):
    """Abstract base class for STT providers."""
    
    def __init__(self, config: STTConfig):
        self.config = config
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass
    
    @abstractmethod
    async def transcribe(
        self,
        audio_queue: asyncio.Queue,
        bypass_vad: bool = False
    ) -> STTResult:
        """
        Transcribe audio from the queue.
        
        Args:
            audio_queue: Queue of audio chunks from microphone
            bypass_vad: If True, skip local VAD
            
        Returns:
            STTResult with transcription and timing info
        """
        pass
