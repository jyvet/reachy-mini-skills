"""Speech module - TTS (Text-to-Speech) providers.

Usage:
    from reachy_mini_skills.speech import tts_cartesia, tts_unmute
    
    tts = tts_cartesia.create()
    result = await tts.synthesize("Hello world!")

For STT (Speech-to-Text), use perception.audio:
    from reachy_mini_skills.perception.audio import stt_cartesia, stt_deepgram
"""

from . import tts_cartesia
from . import tts_unmute

# TTS base classes
from .tts_base import TTSProvider, TTSResult
from .tts_cartesia import CartesiaTTS
from .tts_unmute import UnmuteTTS

__all__ = [
    # TTS modules
    "tts_cartesia",
    "tts_unmute",
    # TTS classes
    "TTSProvider",
    "TTSResult",
    "CartesiaTTS",
    "UnmuteTTS",
    "get_tts_provider",
]


def get_tts_provider(provider_name: str, config) -> TTSProvider:
    """Factory function to get TTS provider by name."""
    providers = {
        "unmute": UnmuteTTS,
        "cartesia": CartesiaTTS,
    }
    
    provider_class = providers.get(provider_name.lower())
    if provider_class is None:
        raise ValueError(f"Unknown TTS provider: {provider_name}. Available: {list(providers.keys())}")
    
    return provider_class(config)
