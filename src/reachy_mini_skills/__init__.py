"""Reachy Mini Skills - Modular components for building Reachy Mini applications.

Easy import patterns by category:

    # Perception (Audio STT & Vision)
    from reachy_mini_skills.perception.audio import cartesia
    stt = cartesia.create()
    
    from reachy_mini_skills.perception.vision import vl_ollama
    vl = vl_ollama.create()
    
    # Speech (TTS)
    from reachy_mini_skills.speech import tts_cartesia
    tts = tts_cartesia.create()
    
    # LLMs
    from reachy_mini_skills.llms import cerebras, ollama
    llm = cerebras.create()
    
    # Robot
    from reachy_mini_skills.robot import movement, emotions, face_tracking
    
    # Conversation orchestration
    from reachy_mini_skills.conversation import (
        ConversationOrchestrator,
        ProviderManager,
        determine_default_providers,
    )
    
    # API utilities (for building FastAPI apps)
    from reachy_mini_skills.api_utils import (
        ToggleState, ApiKeyUpdate, TTSRequest, LLMRequest,
        list_audio_speakers, list_audio_microphones,
        fetch_huggingface_llm_models, fetch_huggingface_vl_models,
    )
"""

from .config import (
    Config,
    STTConfig,
    TTSConfig,
    LLMConfig,
    AudioConfig,
    VisionConfig,
    RobotConfig,
    DEFAULT_SENTIMENT_TO_EMOTION,
    SYSTEM_PROMPT_SUFFIX,
)

# Conversation orchestration
from .conversation import (
    ConversationOrchestrator,
    ProviderManager,
    ProviderState,
    ConversationState,
    AudioState,
    determine_default_providers,
    set_system_output_volume,
    set_system_input_volume,
    create_audio_manager,
    monitor_barge_in,
)

# API utilities for building apps
from .api_utils import (
    # Pydantic models
    ToggleState,
    ApiKeyUpdate,
    TTSRequest,
    LLMRequest,
    SystemPromptUpdate,
    ProviderUpdate,
    VoiceIdUpdate,
    SpeedUpdate,
    ModelUpdate,
    VolumeUpdate,
    EmotionRequest,
    KyutaiTTSConfigUpdate,
    KyutaiSTTConfigUpdate,
    WhisperSocketConfigUpdate,
    OllamaConfigUpdate,
    HuggingFaceConfigUpdate,
    VADConfigUpdate,
    AudioOutputUpdate,
    AudioInputUpdate,
    VisionConfigUpdate,
    VisionHuggingFaceConfigUpdate,
    # Helper functions
    list_audio_speakers,
    list_audio_microphones,
    fetch_huggingface_llm_models,
    fetch_huggingface_vl_models,
    fetch_cerebras_models,
    validate_provider_api_keys,
    get_api_key_preview,
)

from .app import ReachyMiniApp, AppSettings
from .perception.audio import AudioManager, ReachyAudioManager, compute_rms_energy, list_speakers, list_microphones

# STT providers (from perception.audio)
from .perception.audio import (
    STTProvider,
    STTResult,
    UnmuteSTT,
    CartesiaSTT,
    WhisperSocketSTT,
    WhisperSTT,  # Backwards compatibility alias
    get_stt_provider,
)

# TTS providers
from .actuation.speech import (
    TTSProvider,
    TTSResult,
    UnmuteTTS,
    CartesiaTTS,
    get_tts_provider,
)

# LLM providers
from .llms import (
    LLMProvider,
    LLMResult,
    OllamaLLM,
    CerebrasLLM,
    HuggingFaceLLM,
    get_llm_provider,
)

# Robot components
from .actuation.motion import (
    FaceTrackingWorker,
    MotionController,
    BreathingMove,
    TalkingMove,
    TransitionToNeutralMove,
    SimpleController,  # Backwards compatibility alias for MotionController
    load_emotions,
    get_emotion_for_sentiment,
    parse_emotion_tags,
    get_animations_for_tag,
    convert_emotion_tags_for_cartesia,
)

# Vision
from .perception.vision import VisionManager

# Category modules
from .actuation import speech
from . import perception
from . import llms
from .actuation import motion as movement

__all__ = [
    # Config
    "Config",
    "STTConfig",
    "TTSConfig",
    "LLMConfig",
    "AudioConfig",
    "VisionConfig",
    "RobotConfig",
    "DEFAULT_SENTIMENT_TO_EMOTION",
    # App
    "ReachyMiniApp",
    "AppSettings",
    # Audio
    "AudioManager",
    "compute_rms_energy",
    "list_speakers",
    # Vision
    "VisionManager",
    # STT
    "STTProvider",
    "STTResult",
    "UnmuteSTT",
    "CartesiaSTT",
    "WhisperSocketSTT",
    "WhisperSTT",  # Backwards compatibility alias
    "get_stt_provider",
    # TTS
    "TTSProvider",
    "TTSResult",
    "UnmuteTTS",
    "CartesiaTTS",
    "get_tts_provider",
    # LLM
    "LLMProvider",
    "LLMResult",
    "OllamaLLM",
    "CerebrasLLM",
    "HuggingFaceLLM",
    "get_llm_provider",
    # Robot
    "FaceTrackingWorker",
    "MotionController",
    "BreathingMove",
    "TalkingMove",
    "TransitionToNeutralMove",
    "SimpleController",  # Backwards compatibility alias
    "load_emotions",
    "get_emotion_for_sentiment",
    "parse_emotion_tags",
    "get_animations_for_tag",
    "convert_emotion_tags_for_cartesia",
    # Conversation orchestration
    "ConversationOrchestrator",
    "ProviderManager",
    "ProviderState",
    "ConversationState",
    "AudioState",
    "determine_default_providers",
    "set_system_output_volume",
    "set_system_input_volume",
    "create_audio_manager",
    "monitor_barge_in",
    # API utilities - Pydantic models
    "ToggleState",
    "ApiKeyUpdate",
    "TTSRequest",
    "LLMRequest",
    "SystemPromptUpdate",
    "ProviderUpdate",
    "VoiceIdUpdate",
    "SpeedUpdate",
    "ModelUpdate",
    "VolumeUpdate",
    "EmotionRequest",
    "KyutaiTTSConfigUpdate",
    "KyutaiSTTConfigUpdate",
    "WhisperSocketConfigUpdate",
    "OllamaConfigUpdate",
    "HuggingFaceConfigUpdate",
    "VADConfigUpdate",
    "AudioOutputUpdate",
    "AudioInputUpdate",
    "VisionConfigUpdate",
    "VisionHuggingFaceConfigUpdate",
    # API utilities - Helper functions
    "list_audio_speakers",
    "list_audio_microphones",
    "fetch_huggingface_llm_models",
    "fetch_huggingface_vl_models",
    "fetch_cerebras_models",
    "validate_provider_api_keys",
    "get_api_key_preview",
    # Category modules
    "speech",
    "perception",
    "llms",
    "movement",
]
