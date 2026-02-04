"""Audio perception (STT) providers for Reachy Mini Library.

Usage:
    from reachy_mini_skills.perception.audio import stt_cartesia
    
    stt = stt_cartesia.create()
    result = await stt.transcribe(audio_queue)
    
    # VAD can be used standalone
    from reachy_mini_skills.perception.audio.vad import EnergyVAD, WhisperSocketVAD
    
    # Audio I/O
    from reachy_mini_skills.perception.audio.io import AudioManager, list_speakers
"""

from .stt_base import STTProvider, STTResult
from .stt_unmute import UnmuteSTT
from .stt_cartesia import CartesiaSTT
from .stt_whispersocket import WhisperSocketSTT, WhisperSTT  # WhisperSTT is backwards compat alias
from .vad import VADBase, EnergyVAD, WhisperSocketVAD, WhisperVAD, VADResult, compute_rms_energy
from .io import AudioManager, ReachyAudioManager, list_speakers, list_microphones
from . import stt_unmute
from . import stt_cartesia
from . import stt_whispersocket
from . import vad
from . import io

# Backwards compatibility alias
stt_whisper = stt_whispersocket

__all__ = [
    "STTProvider",
    "STTResult",
    "UnmuteSTT",
    "CartesiaSTT",
    "WhisperSocketSTT",
    "WhisperSTT",  # Backwards compatibility alias
    "VADBase",
    "EnergyVAD",
    "WhisperSocketVAD",
    "WhisperVAD",  # Backwards compatibility alias
    "VADResult",
    "compute_rms_energy",
    "AudioManager",
    "ReachyAudioManager",
    "list_speakers",
    "list_microphones",
    "stt_unmute",
    "stt_cartesia",
    "stt_whispersocket",
    "stt_whisper",  # Backwards compatibility alias
    "vad",
    "io",
    "get_stt_provider",
]


def get_stt_provider(provider_name: str, config) -> STTProvider:
    """Factory function to get STT provider by name."""
    providers = {
        "unmute": UnmuteSTT,
        "cartesia": CartesiaSTT,
        "whispersocket": WhisperSocketSTT,
        "whisper": WhisperSocketSTT,  # Backwards compatibility
    }
    
    provider_class = providers.get(provider_name.lower())
    if provider_class is None:
        raise ValueError(f"Unknown STT provider: {provider_name}. Available: {list(providers.keys())}")
    
    return provider_class(config)
