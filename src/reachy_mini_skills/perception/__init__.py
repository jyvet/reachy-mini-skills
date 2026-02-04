"""Perception module - Audio (STT) and Vision providers.

Usage:
    # Audio (STT)
    from reachy_mini_skills.perception.audio import cartesia
    stt = cartesia.create()
    result = await stt.transcribe(audio_queue)
    
    # Vision
    from reachy_mini_skills.perception.vision import vl_ollama
    vl = vl_ollama.create()
    description = await vl.describe_image(session, img)
"""

from . import audio
from . import vision

# Re-export audio (STT) classes
from .audio import (
    STTProvider,
    STTResult,
    UnmuteSTT,
    CartesiaSTT,
    WhisperSocketSTT,
    WhisperSTT,  # Backwards compatibility alias
    get_stt_provider,
)

__all__ = [
    # Modules
    "audio",
    "vision",
    # STT classes
    "STTProvider",
    "STTResult",
    "UnmuteSTT",
    "CartesiaSTT",
    "WhisperSocketSTT",
    "WhisperSTT",  # Backwards compatibility alias
    "get_stt_provider",
]
