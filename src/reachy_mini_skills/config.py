"""Configuration management for Reachy Mini Library."""

import os
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Any


@dataclass
class STTConfig:
    """Speech-to-text configuration."""
    # Kyutai (Unmute) - can be configured via KYUTAI_STT_IP and KYUTAI_STT_PORT env vars
    kyutai_stt_ip: str = field(default_factory=lambda: os.environ.get("KYUTAI_STT_IP", ""))
    kyutai_stt_port: int = field(default_factory=lambda: int(os.environ.get("KYUTAI_STT_PORT", "8088")))
    unmute_ws: str = field(default_factory=lambda: f"ws://{os.environ.get('KYUTAI_STT_IP', 'localhost')}:{os.environ.get('KYUTAI_STT_PORT', '8088')}")
    
    # Cartesia
    cartesia_api_key: str = field(default_factory=lambda: os.environ.get("CARTESIA_API_KEY", ""))
    cartesia_sample_rate: int = 16000
    cartesia_chunk_size: int = 640
    
    # Deepgram
    deepgram_sample_rate: int = 16000
    deepgram_model: str = "flux-general-en"
    
    # WhisperSocket (WebSocket-based, see: https://github.com/jyvet/whispersocket)
    # Can be configured via WHISPERSOCKET_IP and WHISPERSOCKET_PORT env vars
    whispersocket_ip: str = field(default_factory=lambda: os.environ.get("WHISPERSOCKET_IP", ""))
    whispersocket_port: int = field(default_factory=lambda: int(os.environ.get("WHISPERSOCKET_PORT", "5000")))
    whispersocket_ws: str = field(default_factory=lambda: f"ws://{os.environ.get('WHISPERSOCKET_IP', 'localhost')}:{os.environ.get('WHISPERSOCKET_PORT', '5000')}/ws/transcribe")
    whispersocket_sample_rate: int = 16000
    whispersocket_chunk_duration: float = 0.1
    whispersocket_vad_threshold: int = 200
    whispersocket_silence_chunks: int = 5
    whispersocket_silence_chunks_short: int = 10
    whispersocket_pre_speech_buffer: int = 4
    whispersocket_min_silence_for_count: int = 2
    
    # Common STT settings
    energy_threshold: float = 0.01
    energy_frames_required: int = 3
    pre_speech_timeout: float = 60.0
    silence_timeout: float = 1.2
    silence_timeout_with_punct: float = 1.0
    max_duration: float = 30.0
    buffer_seconds: float = 3.0
    
    # VAD
    pause_prediction_head_index: int = 0
    pause_prediction_head_index_medium: int = 1
    pause_prediction_threshold: float = 0.75
    pause_prediction_threshold_with_punct: float = 0.70

    def update(self, **kwargs) -> "STTConfig":
        """Return a new config with updated values."""
        return replace(self, **kwargs)


@dataclass
class TTSConfig:
    """Text-to-speech configuration."""
    # Kyutai (Unmute) - can be configured via KYUTAI_TTS_IP, KYUTAI_TTS_PORT, and KYUTAI_TTS_VOICE env vars
    kyutai_tts_ip: str = field(default_factory=lambda: os.environ.get("KYUTAI_TTS_IP", ""))
    kyutai_tts_port: int = field(default_factory=lambda: int(os.environ.get("KYUTAI_TTS_PORT", "8089")))
    unmute_stream: str = field(default_factory=lambda: f"ws://{os.environ.get('KYUTAI_TTS_IP', 'localhost')}:{os.environ.get('KYUTAI_TTS_PORT', '8089')}")
    unmute_voice: str = field(default_factory=lambda: os.environ.get("KYUTAI_TTS_VOICE", "unmute-prod-website/ex04_narration_longform_00001.wav"))
    unmute_api_key: str = "public_token"
    
    # Cartesia - can be configured via CARTESIA_API_KEY, CARTESIA_TTS_VOICE, and CARTESIA_TTS_SPEED env vars
    cartesia_api_key: str = field(default_factory=lambda: os.environ.get("CARTESIA_API_KEY", ""))
    cartesia_voice_id: str = field(default_factory=lambda: os.environ.get("CARTESIA_TTS_VOICE", "cbaf8084-f009-4838-a096-07ee2e6612b1"))
    cartesia_sample_rate: int = 22050
    cartesia_silence_detect: float = 0.1
    cartesia_speed: str = field(default_factory=lambda: os.environ.get("CARTESIA_TTS_SPEED", "fast"))  # Speed: "slowest", "slow", "normal", "fast", "fastest" or -1.0 to 1.0

    def update(self, **kwargs) -> "TTSConfig":
        """Return a new config with updated values."""
        return replace(self, **kwargs)


@dataclass
class LLMConfig:
    """LLM configuration."""
    # Ollama - host and port can be set via OLLAMA_IP and OLLAMA_PORT env vars
    ollama_host: str = field(default_factory=lambda: os.environ.get("OLLAMA_IP", "localhost"))
    ollama_port: int = field(default_factory=lambda: int(os.environ.get("OLLAMA_PORT", "11434")))
    ollama_default_model: str = field(default_factory=lambda: os.environ.get("OLLAMA_LLM", "deepseek-llm:7b"))
    
    # Cerebras
    cerebras_api_key: str = field(default_factory=lambda: os.environ.get("CEREBRAS_API_KEY", ""))
    cerebras_default_model: str = field(default_factory=lambda: os.environ.get("CEREBRAS_LLM", "gpt-oss-120b"))
    cerebras_streaming: bool = False
    
    # Hugging Face
    huggingface_api_key: str = field(default_factory=lambda: os.environ.get("HUGGINGFACE_API_KEY", ""))
    huggingface_default_model: str = field(default_factory=lambda: os.environ.get("HUGGINGFACE_LLM", "openai/gpt-oss-120b:cerebras"))
    huggingface_api_url: str = ""  # Empty = use serverless API, or set to dedicated endpoint URL
    
    @property
    def ollama_url(self) -> str:
        """Construct Ollama URL from host and port."""
        return f"http://{self.ollama_host}:{self.ollama_port}/api/chat"

    def update(self, **kwargs) -> "LLMConfig":
        """Return a new config with updated values."""
        return replace(self, **kwargs)


@dataclass
class AudioConfig:
    """Audio configuration."""
    sample_rate: int = 24000
    block_size: int = 1920
    input_device_id: Optional[int] = None  # None = system default

    def update(self, **kwargs) -> "AudioConfig":
        """Return a new config with updated values."""
        return replace(self, **kwargs)


@dataclass
class VisionConfig:
    """Vision configuration."""
    model: str = field(default_factory=lambda: os.environ.get("OLLAMA_VLM", "qwen3-vl:2b"))  # Ollama model
    huggingface_model: str = field(default_factory=lambda: os.environ.get("HUGGINGFACE_VLM", "Qwen/Qwen3-VL-8B-Instruct"))  # Hugging Face VL model
    provider: str = "huggingface"  # "huggingface" or "ollama"
    trigger_word: str = field(default_factory=lambda: os.environ.get("VLM_TRIGGER_WORD", "picture"))
    attention_emotion: str = "yes1"

    def update(self, **kwargs) -> "VisionConfig":
        """Return a new config with updated values."""
        return replace(self, **kwargs)


@dataclass
class RobotConfig:
    """Robot configuration."""
    emotion_library: str = "jyvet/reachy-mini-emotions-library-noaudio"
    
    # Breathing
    breathing_z_amplitude: float = 0.008  # Reduced amplitude
    breathing_frequency: float = 0.05
    head_breathing_enabled: bool = True
    antenna_sway_amplitude_deg: float = 15.0
    antenna_frequency: float = 0.5
    antenna_breathing_enabled: bool = True
    antenna_excited_frequency: float = 3.0  # Faster frequency for excited/attention mode
    antenna_excited_mode: bool = False  # When True, use excited frequency
    
    # Talking
    talking_pitch_amplitude_deg: float = 3.0
    talking_yaw_amplitude_deg: float = 2.0
    talking_pitch_frequency: float = 2.5
    talking_yaw_frequency: float = 0.8
    talking_z_amplitude: float = 0.002
    talking_antenna_amplitude_deg: float = 5.0
    talking_antenna_frequency: float = 1.5
    
    # Control loop
    control_loop_frequency_hz: float = 100.0
    
    # Face tracking
    face_tracking_enabled: bool = True
    face_tracking_max_yaw_deg: float = 35.0
    face_tracking_max_pitch_deg: float = 25.0
    face_tracking_smoothing: float = 0.25  # Higher = more responsive
    face_tracking_decay: float = 0.92
    face_tracking_fps: int = 20
    
    # Body rotation for face tracking (body is PRIMARY, head makes fine adjustments)
    body_rotation_enabled: bool = True
    body_rotation_speed: float = 1.5  # Radians per second for body tracking
    body_rotation_max_deg: float = 120.0  # Maximum body rotation
    body_rotation_smoothing: float = 0.10  # Body smoothing factor (lower = smoother)
    head_fine_adjustment_max_deg: float = 15.0  # Max head yaw offset from body alignment
    
    # PID controller for face tracking (reduces oscillations)
    face_tracking_pid_kp: float = 0.8  # Proportional gain (higher = faster centering)
    face_tracking_pid_ki: float = 0.20  # Integral gain (eliminates steady-state error)
    face_tracking_pid_kd: float = 0.04  # Derivative gain (dampens oscillations with lower value)
    face_tracking_pid_integral_limit: float = 0.5  # Anti-windup limit for integral term
    face_tracking_deadzone: float = 0.02  # Ignore small offsets (normalized, 0-1)
    face_tracking_hold_time: float = 10.0  # Seconds to hold position after face is lost

    def update(self, **kwargs) -> "RobotConfig":
        """Return a new config with updated values."""
        return replace(self, **kwargs)


# Default sentiment to emotion mapping
DEFAULT_SENTIMENT_TO_EMOTION: Dict[str, List[str]] = {
    "welcoming": ["dance1"],
    "welcome": ["dance1"],
    "happy": ["cheerful1", "laughing1", "laughing2", "success1", "success2"],
    "excited": ["enthusiastic1"],
    "enthusiastic": ["enthusiastic1"],
    "elated": ["cheerful1", "laughing1", "dance1", "success1"],
    "euphoric": ["laughing1", "laughing2", "dance1", "electric1"],
    "triumphant": ["success1", "success2", "proud1", "proud2", "proud3"],
    "amazed": ["amazed1", "surprised1", "surprised2"],
    "surprised": ["surprised1", "surprised2", "amazed1"],
    "flirtatious": ["loving1", "cheerful1", "playful1"],
    "joking": ["laughing1", "laughing2", "cheerful1", "playful1"],
    "comedic": ["laughing1", "laughing2", "cheerful1"],
    "curious": ["inquiring1", "inquiring2", "inquiring3"],
    "contemplative": ["thoughtful1", "thoughtful2", "inquiring1"],
    "content": ["serenity1", "calming1", "cheerful1"],
    "peaceful": ["serenity1", "calming1"],
    "serene": ["serenity1", "calming1"],
    "calm": ["serenity1", "calming1", "attentive1", "attentive2"],
    "grateful": ["loving1", "cheerful1", "welcoming1"],
    "affectionate": ["loving1", "welcoming1", "welcoming2"],
    "trust": ["attentive1", "attentive2", "welcoming1", "helpful1"],
    "sympathetic": ["loving1", "sad1", "attentive1"],
    "anticipation": ["inquiring1", "enthusiastic1"],
    "mysterious": ["thoughtful1", "thoughtful2", "inquiring1"],
    "angry": ["angry1", "frustrated1", "agitated1"],
    "mad": ["angry1", "frustrated1"],
    "outraged": ["angry1", "frustrated1", "agitated1"],
    "frustrated": ["frustrated1", "agitated1", "oops1"],
    "agitated": ["agitated1", "frustrated1", "anxiety1"],
    "threatened": ["fear1", "scared1", "anxiety1"],
    "disgusted": ["disgusted1", "indifferent1"],
    "contempt": ["indifferent1", "disgusted1"],
    "envious": ["sad1", "indifferent1", "frustrated1"],
    "sarcastic": ["indifferent1", "playful1"],
    "ironic": ["indifferent1", "thoughtful1"],
    "sad": ["sad1", "sad2", "downcast1", "lonely1"],
    "dejected": ["sad1", "sad2", "downcast1", "resigned1"],
    "melancholic": ["sad1", "sad2", "lonely1", "wistful1"],
    "disappointed": ["sad1", "downcast1", "resigned1", "oops1"],
    "hurt": ["sad1", "sad2", "lonely1"],
    "guilty": ["sad1", "downcast1", "apologetic1"],
    "bored": ["indifferent1", "tired1", "resigned1"],
    "tired": ["tired1", "resigned1", "downcast1"],
    "rejected": ["sad1", "lonely1", "downcast1"],
    "nostalgic": ["thoughtful1", "sad1", "wistful1"],
    "wistful": ["thoughtful1", "sad1", "wistful1"],
    "apologetic": ["apologetic1", "sad1", "downcast1"],
    "hesitant": ["uncertain1", "confused1", "thoughtful1"],
    "insecure": ["uncertain1", "confused1", "anxiety1"],
    "confused": ["confused1", "uncertain1", "lost1", "incomprehensible2", "oops1", "oops2"],
    "resigned": ["resigned1", "sad1", "indifferent1"],
    "anxious": ["curious1"],
    "panicked": ["fear1", "scared1", "anxiety1"],
    "alarmed": ["fear1", "scared1", "surprised1"],
    "scared": ["rage1"],
    "neutral": ["attentive1", "attentive2", "welcoming1", "welcoming2", "helpful1", "helpful2"],
    "proud": ["proud1", "proud2", "proud3", "success1"],
    "confident": ["proud1", "proud2", "attentive1", "success1"],
    "distant": ["indifferent1", "thoughtful1"],
    "skeptical": ["inquiring1", "thoughtful1", "uncertain1"],
    "determined": ["attentive1", "attentive2", "proud1", "enthusiastic1"],
}


# System prompt suffix for TTS optimization and sentiment tagging
# This is automatically appended to user-defined system prompts
SYSTEM_PROMPT_SUFFIX = """

You must follow these rules:

**Additional Behavior Constraint**:
   - Never describe actions, emotions, or internal states in third person.
   - Never narrate what you are doing or why you are doing it.
   - Speak only as the character, through in-universe dialogue and system-style utterances.
   - Do not explain your personality, rules, or role.
   - Do not break character for meta commentary.

**TTS Optimization**:
   - Generate text that is 100% pronounceable (no em dashes, no symbols, no smileys, no ambiguous abbreviations).
   - Replace numbers/symbols with words if needed (e.g., '3' → 'three').
   - Use natural phrasing with no slang or jargon.
   - Don't use onomatopoeia like 'ahhhh' or 'ohhhh'. Don't use capital letters.

**Sentiment Tagging**:
   - Prepend an emotion label in square brackets. List of emotions is: [welcoming], [happy], [excited], [enthusiastic], [elated], [euphoric], [triumphant], [amazed], [surprised], [flirtatious], [joking/comedic], [curious], [content], [peaceful], [serene], [calm], [grateful], [affectionate], [trust], [sympathetic], [anticipation], [mysterious], [angry], [mad], [outraged], [frustrated], [agitated], [threatened], [disgusted], [contempt], [envious], [sarcastic], [ironic], [sad], [dejected], [melancholic], [disappointed], [hurt], [guilty], [bored], [tired], [rejected], [nostalgic], [wistful], [apologetic], [hesitant], [insecure], [confused], [resigned], [anxious], [panicked], [alarmed], [scared], [neutral], [proud], [confident], [distant], [skeptical], [contemplative], [determined].
   - These will be parsed out before TTS.
"""


@dataclass
class Config:
    """Main configuration container."""
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    sentiment_to_emotion: Dict[str, List[str]] = field(
        default_factory=lambda: DEFAULT_SENTIMENT_TO_EMOTION.copy()
    )
    
    @classmethod
    def default(cls) -> "Config":
        """Create default configuration."""
        return cls()
    
    def update_stt(self, **kwargs) -> "Config":
        """Return a new config with updated STT settings."""
        return replace(self, stt=self.stt.update(**kwargs))
    
    def update_tts(self, **kwargs) -> "Config":
        """Return a new config with updated TTS settings."""
        return replace(self, tts=self.tts.update(**kwargs))
    
    def update_llm(self, **kwargs) -> "Config":
        """Return a new config with updated LLM settings."""
        return replace(self, llm=self.llm.update(**kwargs))
    
    def update_audio(self, **kwargs) -> "Config":
        """Return a new config with updated audio settings."""
        return replace(self, audio=self.audio.update(**kwargs))
    
    def update_vision(self, **kwargs) -> "Config":
        """Return a new config with updated vision settings."""
        return replace(self, vision=self.vision.update(**kwargs))
    
    def update_robot(self, **kwargs) -> "Config":
        """Return a new config with updated robot settings."""
        return replace(self, robot=self.robot.update(**kwargs))
    
    def with_sentiment_mapping(self, mapping: Dict[str, List[str]]) -> "Config":
        """Return a new config with custom sentiment to emotion mapping."""
        return replace(self, sentiment_to_emotion=mapping)
