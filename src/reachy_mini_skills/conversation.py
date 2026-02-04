"""Conversation orchestration for Reachy Mini applications.

This module provides high-level conversation management including:
- Provider management (STT, TTS, LLM, Vision)
- Conversation loops with VAD
- Barge-in detection for interrupting TTS
- Audio routing (Reachy/system)
"""

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import aiohttp

from .config import Config, SYSTEM_PROMPT_SUFFIX
from .perception.audio import (
    AudioManager,
    ReachyAudioManager,
    CartesiaSTT,
    UnmuteSTT,
    WhisperSocketSTT,
    compute_rms_energy,
)
from .actuation.speech import CartesiaTTS, UnmuteTTS
from .llms import CerebrasLLM, OllamaLLM, HuggingFaceLLM
from .perception.vision import VisionManager
from .actuation.motion import (
    MotionController,
    parse_emotion_tags,
    get_animations_for_tag,
    convert_emotion_tags_for_cartesia,
)


@dataclass
class ProviderState:
    """Current state of all providers."""
    # STT
    stt_provider: str = "cartesia"
    stt: Optional[Union[CartesiaSTT, UnmuteSTT, WhisperSocketSTT]] = None
    
    # TTS
    tts_provider: str = "cartesia"
    tts: Optional[Union[CartesiaTTS, UnmuteTTS]] = None
    
    # LLM
    llm_provider: str = "huggingface"
    llm: Optional[Union[CerebrasLLM, OllamaLLM, HuggingFaceLLM]] = None
    
    # Vision
    vision_provider: str = "huggingface"
    vision_manager: Optional[VisionManager] = None


@dataclass
class ConversationState:
    """State tracking for active conversation."""
    active: bool = False
    status: str = "idle"  # idle, listening, processing, speaking
    is_listening: bool = False
    is_processing: bool = False
    
    # Feature toggles
    vision_enabled: bool = True
    
    # History
    history: List[Dict[str, str]] = field(default_factory=list)
    system_prompt: str = "You are a helpful AI assistant."
    
    # Last values
    last_transcription: str = ""
    last_tts_text: str = ""
    last_llm_response: str = ""
    last_emotion: str = ""
    last_was_interrupted: bool = False
    
    # Latencies
    last_stt_latency: Optional[float] = None
    last_llm_latency: Optional[float] = None
    last_tts_latency: Optional[float] = None
    last_vl_latency: Optional[float] = None
    last_total_latency: Optional[float] = None
    
    # Vision
    last_captured_frame: Optional[str] = None
    last_vision_description: Optional[str] = None


@dataclass
class AudioState:
    """Audio routing configuration."""
    # Output
    output_mode: str = "reachy"  # "reachy", "system", "custom"
    speaker_id: Optional[int] = None
    output_volume: int = 100
    
    # Input
    input_mode: str = "reachy"  # "reachy", "system", "custom"
    mic_id: Optional[int] = None
    input_volume: int = 100


def determine_default_providers(config: Config) -> Dict[str, str]:
    """Determine default providers based on environment variables.
    
    Priority logic:
    - STT: Default Cartesia. If WhisperSocket IP defined -> WhisperSocket.
           If Kyutai STT IP defined -> Kyutai (but WhisperSocket has higher priority if both defined)
    - TTS: Default Cartesia. If Kyutai TTS IP defined -> Kyutai.
    - LLM: Default Hugging Face. If Cerebras API key defined -> Cerebras.
           If Ollama host/IP defined (and no Cerebras key) -> Ollama.
           If Hugging Face API key defined -> Hugging Face.
    - VL:  Default Hugging Face. If Ollama host/IP defined -> Ollama.
           If Hugging Face API key defined -> Hugging Face.
    
    Args:
        config: The configuration object
        
    Returns:
        Dict with keys: stt_provider, tts_provider, llm_provider, vision_provider
    """
    # Read environment variables from config (which reads from os.environ)
    kyutai_stt_ip = config.stt.kyutai_stt_ip
    kyutai_tts_ip = config.tts.kyutai_tts_ip
    whispersocket_ip = config.stt.whispersocket_ip
    cerebras_api_key = config.llm.cerebras_api_key
    huggingface_api_key = config.llm.huggingface_api_key
    cartesia_api_key = config.stt.cartesia_api_key
    ollama_host = config.llm.ollama_host
    
    # Debug output to help diagnose env var detection
    print(f"[ProviderManager] Detected env vars:")
    print(f"  - CARTESIA_API_KEY: {'set' if cartesia_api_key else 'not set'}")
    print(f"  - CEREBRAS_API_KEY: {'set' if cerebras_api_key else 'not set'}")
    print(f"  - HUGGINGFACE_API_KEY: {'set' if huggingface_api_key else 'not set'}")
    print(f"  - KYUTAI_STT_IP: {kyutai_stt_ip or 'not set'}")
    print(f"  - KYUTAI_TTS_IP: {kyutai_tts_ip or 'not set'}")
    print(f"  - WHISPERSOCKET_IP: {whispersocket_ip or 'not set'}")
    print(f"  - OLLAMA_IP: {ollama_host}")
    
    # Determine STT provider
    # Priority: WhisperSocket > Kyutai > Cartesia (default)
    stt_provider = "cartesia"
    if kyutai_stt_ip:
        stt_provider = "kyutai"
    if whispersocket_ip:
        stt_provider = "whispersocket"
    
    # Determine TTS provider
    # Priority: Kyutai (if defined) > Cartesia (default)
    tts_provider = "cartesia"
    if kyutai_tts_ip:
        tts_provider = "kyutai"
    
    # Determine LLM provider
    # Priority: Cerebras (if key defined) > Ollama (if host defined and no Cerebras) > Hugging Face (default)
    llm_provider = "huggingface"
    ollama_explicitly_set = os.environ.get("OLLAMA_IP", "") != ""
    if ollama_explicitly_set and not cerebras_api_key:
        llm_provider = "ollama"
    if huggingface_api_key:
        llm_provider = "huggingface"
    if cerebras_api_key:
        llm_provider = "cerebras"
    
    # Determine Vision provider
    # Priority: Ollama (if host defined) > Hugging Face (default)
    vision_provider = "huggingface"
    if ollama_explicitly_set:
        vision_provider = "ollama"
    if huggingface_api_key:
        vision_provider = "huggingface"
    
    print(f"[ProviderManager] Selected providers: STT={stt_provider}, TTS={tts_provider}, LLM={llm_provider}, Vision={vision_provider}")
    
    return {
        "stt_provider": stt_provider,
        "tts_provider": tts_provider,
        "llm_provider": llm_provider,
        "vision_provider": vision_provider,
    }


class ProviderManager:
    """Manages STT, TTS, LLM, and Vision providers with dynamic switching."""
    
    def __init__(self, config: Config):
        """Initialize provider manager.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.state = ProviderState()
        
        # Determine default providers from environment
        defaults = determine_default_providers(config)
        self.state.stt_provider = defaults["stt_provider"]
        self.state.tts_provider = defaults["tts_provider"]
        self.state.llm_provider = defaults["llm_provider"]
        self.state.vision_provider = defaults["vision_provider"]
    
    def update_config(self, **kwargs) -> None:
        """Update configuration and recreate affected providers.
        
        Supports kwargs for any config section:
        - stt_*: STT config updates
        - tts_*: TTS config updates
        - llm_*: LLM config updates
        - vision_*: Vision config updates
        - audio_*: Audio config updates
        """
        stt_updates = {}
        tts_updates = {}
        llm_updates = {}
        vision_updates = {}
        audio_updates = {}
        
        for key, value in kwargs.items():
            if key.startswith("stt_"):
                stt_updates[key[4:]] = value
            elif key.startswith("tts_"):
                tts_updates[key[4:]] = value
            elif key.startswith("llm_"):
                llm_updates[key[4:]] = value
            elif key.startswith("vision_"):
                vision_updates[key[7:]] = value
            elif key.startswith("audio_"):
                audio_updates[key[6:]] = value
        
        if stt_updates:
            self.config = self.config.update_stt(**stt_updates)
            self._create_stt()
        if tts_updates:
            self.config = self.config.update_tts(**tts_updates)
            self._create_tts()
        if llm_updates:
            self.config = self.config.update_llm(**llm_updates)
            self._create_llm()
        if vision_updates:
            self.config = self.config.update_vision(**vision_updates)
            self._create_vision()
        if audio_updates:
            self.config = self.config.update_audio(**audio_updates)
    
    def set_stt_provider(self, provider: str) -> None:
        """Switch STT provider."""
        if provider not in ["cartesia", "kyutai", "whispersocket"]:
            raise ValueError(f"Invalid STT provider: {provider}")
        self.state.stt_provider = provider
        self._create_stt()
    
    def set_tts_provider(self, provider: str) -> None:
        """Switch TTS provider."""
        if provider not in ["cartesia", "kyutai"]:
            raise ValueError(f"Invalid TTS provider: {provider}")
        self.state.tts_provider = provider
        self._create_tts()
    
    def set_llm_provider(self, provider: str) -> None:
        """Switch LLM provider."""
        if provider not in ["huggingface", "cerebras", "ollama"]:
            raise ValueError(f"Invalid LLM provider: {provider}")
        self.state.llm_provider = provider
        self._create_llm()
    
    def set_vision_provider(self, provider: str) -> None:
        """Switch Vision provider."""
        if provider not in ["huggingface", "ollama"]:
            raise ValueError(f"Invalid Vision provider: {provider}")
        self.state.vision_provider = provider
        self.config = self.config.update_vision(provider=provider)
        self._create_vision()
    
    def _create_stt(self) -> None:
        """Create STT provider based on current state."""
        if self.state.stt_provider == "cartesia":
            self.state.stt = CartesiaSTT(self.config.stt)
        elif self.state.stt_provider == "kyutai":
            self.state.stt = UnmuteSTT(self.config.stt)
        else:  # whispersocket
            self.state.stt = WhisperSocketSTT(self.config.stt)
    
    def _create_tts(self) -> None:
        """Create TTS provider based on current state."""
        if self.state.tts_provider == "cartesia":
            self.state.tts = CartesiaTTS(self.config.tts)
        else:  # kyutai
            self.state.tts = UnmuteTTS(self.config.tts)
    
    def _create_llm(self) -> None:
        """Create LLM provider based on current state."""
        if self.state.llm_provider == "huggingface":
            self.state.llm = HuggingFaceLLM(self.config.llm)
        elif self.state.llm_provider == "cerebras":
            self.state.llm = CerebrasLLM(self.config.llm)
        else:  # ollama
            self.state.llm = OllamaLLM(self.config.llm)
    
    def _create_vision(self) -> None:
        """Create Vision manager based on current state."""
        self.state.vision_manager = VisionManager(self.config.vision, self.config.llm)
    
    def initialize_all(self) -> None:
        """Initialize all providers."""
        self._create_stt()
        self._create_tts()
        self._create_llm()
        self._create_vision()
    
    @property
    def stt(self):
        """Get current STT provider."""
        return self.state.stt
    
    @property
    def tts(self):
        """Get current TTS provider."""
        return self.state.tts
    
    @property
    def llm(self):
        """Get current LLM provider."""
        return self.state.llm
    
    @property
    def vision_manager(self):
        """Get current Vision manager."""
        return self.state.vision_manager
    
    def get_state(self) -> Dict[str, Any]:
        """Get full provider state as a dictionary.
        
        Returns:
            Dict with all provider configurations and API key statuses
        """
        cfg = self.config
        return {
            # API key statuses
            "cartesia_api_key_set": bool(cfg.stt.cartesia_api_key),
            "cartesia_api_key": cfg.stt.cartesia_api_key,
            "cartesia_voice_id": cfg.tts.cartesia_voice_id,
            "cartesia_speed": cfg.tts.cartesia_speed,
            "cerebras_api_key_set": bool(cfg.llm.cerebras_api_key),
            "cerebras_api_key": cfg.llm.cerebras_api_key,
            "cerebras_model": cfg.llm.cerebras_default_model,
            "huggingface_api_key_set": bool(cfg.llm.huggingface_api_key),
            "huggingface_api_key": cfg.llm.huggingface_api_key,
            "huggingface_model": cfg.llm.huggingface_default_model,
            "huggingface_api_url": cfg.llm.huggingface_api_url,
            # Provider selections
            "stt_provider": self.state.stt_provider,
            "tts_provider": self.state.tts_provider,
            "llm_provider": self.state.llm_provider,
            "vision_provider": self.state.vision_provider,
            # Provider configs
            "whispersocket_url": cfg.stt.whispersocket_ws,
            "kyutai_stt_url": cfg.stt.unmute_ws,
            "kyutai_stream_url": cfg.tts.unmute_stream,
            "kyutai_voice": cfg.tts.unmute_voice,
            "ollama_host": cfg.llm.ollama_host,
            "ollama_port": cfg.llm.ollama_port,
            "ollama_model": cfg.llm.ollama_default_model,
            # Ollama environment variable flags
            "ollama_llm_env_set": bool(os.environ.get("OLLAMA_LLM")),
            "ollama_vlm_env_set": bool(os.environ.get("OLLAMA_VLM")),
            # WhisperSocket environment variable flag
            "whispersocket_env_set": bool(os.environ.get("WHISPERSOCKET_IP")),
            # Vision config
            "vision_model": cfg.vision.model,
            "vision_huggingface_model": cfg.vision.huggingface_model,
            "vision_trigger_word": cfg.vision.trigger_word,
        }
    
    def validate_api_keys(self) -> Optional[str]:
        """Validate that required API keys are set for selected providers.
        
        Returns:
            Error message if validation fails, None if all required keys are set
        """
        cfg = self.config
        if self.state.stt_provider == "cartesia" and not cfg.stt.cartesia_api_key:
            return "Cartesia API key not set (required for Cartesia STT)"
        if self.state.tts_provider == "cartesia" and not cfg.tts.cartesia_api_key:
            return "Cartesia API key not set (required for Cartesia TTS)"
        if self.state.llm_provider == "cerebras" and not cfg.llm.cerebras_api_key:
            return "Cerebras API key not set"
        if self.state.llm_provider == "huggingface" and not cfg.llm.huggingface_api_key:
            return "Hugging Face API key not set"
        return None


def set_system_output_volume(volume_percent: int) -> bool:
    """Set system output volume using pactl or amixer.
    
    Args:
        volume_percent: Volume level 0-100
        
    Returns:
        True if successful, False otherwise
    """
    import subprocess
    try:
        subprocess.run(
            ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{volume_percent}%"],
            check=True, capture_output=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            subprocess.run(
                ["amixer", "set", "Master", f"{volume_percent}%"],
                check=True, capture_output=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False


def set_system_input_volume(volume_percent: int) -> bool:
    """Set system input volume/gain using pactl or amixer.
    
    Args:
        volume_percent: Volume level 0-200 (allows boost)
        
    Returns:
        True if successful, False otherwise
    """
    import subprocess
    try:
        subprocess.run(
            ["pactl", "set-source-volume", "@DEFAULT_SOURCE@", f"{volume_percent}%"],
            check=True, capture_output=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            subprocess.run(
                ["amixer", "set", "Capture", f"{volume_percent}%"],
                check=True, capture_output=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False


def create_audio_manager(
    config: Config,
    input_mode: str = "reachy",
    reachy_media: Any = None
) -> Union[AudioManager, ReachyAudioManager]:
    """Create the appropriate audio manager based on input mode.
    
    Args:
        config: Configuration object
        input_mode: "reachy", "system", or "custom"
        reachy_media: ReachyMini.media object for robot audio
        
    Returns:
        AudioManager or ReachyAudioManager depending on configuration
    """
    if input_mode == "reachy" and reachy_media:
        return ReachyAudioManager(reachy_media, config.audio)
    else:
        return AudioManager(config.audio)


async def monitor_barge_in(
    audio_queue: asyncio.Queue,
    interrupt_event: asyncio.Event,
    controller: Optional[MotionController] = None,
    talking_enabled: bool = True,
    use_reachy_speaker: bool = False,
    energy_threshold: float = 0.01,
    frames_required: int = 3,
) -> None:
    """Monitor audio for voice activity to trigger TTS interruption.
    
    Args:
        audio_queue: Queue of audio data to monitor
        interrupt_event: Event to set when barge-in detected
        controller: Optional MotionController to stop talking animation
        talking_enabled: Whether talking animation is enabled
        use_reachy_speaker: Whether Reachy's speaker is being used (higher threshold)
        energy_threshold: Base energy threshold for voice detection
        frames_required: Number of frames above threshold needed
    """
    # When using Reachy Mini speaker, use higher threshold to avoid self-triggering
    if use_reachy_speaker:
        threshold_multiplier = 3.0
        frames_multiplier = 1
    else:
        threshold_multiplier = 2.0
        frames_multiplier = 1
    
    effective_threshold = energy_threshold * threshold_multiplier
    effective_frames = frames_required * frames_multiplier
    
    print(f"[Barge-in] Monitor started: threshold={effective_threshold:.4f}, "
          f"frames_required={effective_frames}, reachy_speaker={use_reachy_speaker}", flush=True)
    
    # Track frames in a sliding window
    window_size = effective_frames * 3
    frame_history = []
    
    try:
        while not interrupt_event.is_set():
            try:
                audio_data = await asyncio.wait_for(audio_queue.get(), timeout=0.05)
                energy = compute_rms_energy(audio_data)
                
                # Track in sliding window
                is_above = energy > effective_threshold
                frame_history.append(is_above)
                if len(frame_history) > window_size:
                    frame_history.pop(0)
                
                frames_above = sum(frame_history)
                
                if is_above:
                    print(f"[Barge-in] Energy={energy:.4f} > threshold={effective_threshold:.4f}, "
                          f"frames_above={frames_above}/{effective_frames} (window={len(frame_history)})", flush=True)
                
                # Trigger if enough frames in the window are above threshold
                if frames_above >= effective_frames:
                    print("\n[Barge-in] Voice detected, interrupting TTS...", flush=True)
                    if talking_enabled and controller:
                        controller.stop_talking()
                    interrupt_event.set()
                    break
            except asyncio.TimeoutError:
                continue
    except asyncio.CancelledError:
        pass


class ConversationOrchestrator:
    """Orchestrates conversation flow with STT -> LLM -> TTS pipeline.
    
    This class handles the full conversation loop including:
    - Voice activity detection
    - Speech-to-text transcription
    - LLM response generation with emotion tags
    - Text-to-speech synthesis with optional barge-in
    - Robot animation coordination
    """
    
    def __init__(
        self,
        provider_manager: ProviderManager,
        controller: Optional[MotionController] = None,
        reachy_mini: Any = None,
        emotions_enabled: bool = True,
        talking_movement_enabled: bool = True,
        interrupt_tts_enabled: bool = True,
    ):
        """Initialize conversation orchestrator.
        
        Args:
            provider_manager: Provider manager instance
            controller: Optional MotionController for robot animations
            reachy_mini: Optional ReachyMini instance for robot audio
            emotions_enabled: Whether to trigger emotion animations
            talking_movement_enabled: Whether to enable head movement during TTS
            interrupt_tts_enabled: Whether user can interrupt TTS by speaking
        """
        self.providers = provider_manager
        self.controller = controller
        self.reachy_mini = reachy_mini
        self.emotions_enabled = emotions_enabled
        self.talking_movement_enabled = talking_movement_enabled
        self.interrupt_tts_enabled = interrupt_tts_enabled
        
        # State
        self.state = ConversationState()
        self.audio_state = AudioState()
        
        # Internal
        self._audio_manager: Optional[AudioManager] = None
        self._tts_interrupt_event: Optional[asyncio.Event] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._tts_lock = asyncio.Lock()
        self._tts_cleanup_task: Optional[asyncio.Task] = None
    
    @property
    def config(self) -> Config:
        """Get current configuration."""
        return self.providers.config
    
    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt for the conversation."""
        self.state.system_prompt = prompt
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.state.history = []
        self.state.last_llm_response = ""
    
    async def process_message(self, user_message: str) -> Dict[str, Any]:
        """Process a single user message through LLM and TTS.
        
        Args:
            user_message: The user's message
            
        Returns:
            Dict with success, response, was_interrupted, etc.
        """
        conversation_start = time.perf_counter()
        
        # Validate providers
        if self.providers.llm is None:
            return {"success": False, "error": "LLM provider not initialized", "step": "llm"}
        
        # Check for vision trigger
        vision_description = None
        if (self.state.vision_enabled and 
            self.providers.vision_manager and 
            self.providers.vision_manager.contains_trigger(user_message)):
            
            print(f"[Vision] Trigger word detected in: {user_message}", flush=True)
            vl_start = time.perf_counter()
            
            # Start antenna excited mode if controller available
            antenna_task = None
            if self.controller:
                antenna_task = asyncio.create_task(
                    self._antenna_excited_for_duration(duration=2.0)
                )
            
            # Capture frame from robot camera
            if self.reachy_mini:
                img_base64 = self.providers.vision_manager.capture_image(self.reachy_mini)
                if img_base64:
                    self.state.last_captured_frame = img_base64
                    print("[Vision] Frame captured", flush=True)
                    
                    async with aiohttp.ClientSession() as vision_session:
                        vision_description = await self.providers.vision_manager.describe_image(
                            vision_session, img_base64
                        )
                        self.state.last_vision_description = vision_description
                        self.state.last_vl_latency = time.perf_counter() - vl_start
                        print(f"[Vision] Description ({self.state.last_vl_latency:.2f}s): {vision_description}", flush=True)
            
            if antenna_task:
                await antenna_task
        
        # Build messages with system prompt
        full_system_prompt = self.state.system_prompt + SYSTEM_PROMPT_SUFFIX
        messages = [{"role": "system", "content": full_system_prompt}]
        messages.extend(self.state.history)
        
        # Add vision context if available
        if vision_description:
            enhanced_message = f"{user_message}\n\n[Visual context: {vision_description}]"
            messages.append({"role": "user", "content": enhanced_message})
        else:
            messages.append({"role": "user", "content": user_message})
        
        tts_was_interrupted = False
        
        try:
            # Get LLM response
            llm_start = time.perf_counter()
            async with aiohttp.ClientSession() as session:
                result = await self.providers.llm.complete(session, messages)
                self.state.last_llm_latency = time.perf_counter() - llm_start
                raw_response = result.text
                print(f"[LLM] Response received ({self.state.last_llm_latency:.2f}s)", flush=True)
                
                # Parse emotion tags from the response
                emotion_tags, clean_response = parse_emotion_tags(raw_response)
                tts_response = convert_emotion_tags_for_cartesia(raw_response)
                self.state.last_llm_response = clean_response
                
                # Update conversation history
                self.state.history.append({"role": "user", "content": user_message})
                self.state.history.append({"role": "assistant", "content": clean_response})
                
                # Keep history manageable
                if len(self.state.history) > 20:
                    self.state.history = self.state.history[-20:]
            
            # Trigger emotion animation
            if self.emotions_enabled and emotion_tags and self.controller:
                for tag in emotion_tags:
                    if tag == "neutral":
                        self.state.last_emotion = "neutral"
                        continue
                    
                    animation_name = get_animations_for_tag(tag, self.controller.emotion_names)
                    if animation_name:
                        print(f"[Conversation] Playing emotion '{animation_name}' for tag [{tag}]", flush=True)
                        self.state.last_emotion = tag
                        self.controller.queue_emotion(animation_name, priority=True)
                        break
            
            # Speak the response
            if self.providers.tts and tts_response:
                self.state.last_tts_text = tts_response
                tts_was_interrupted = await self._speak_with_barge_in(tts_response)
                
                if tts_was_interrupted:
                    self.state.last_was_interrupted = True
                    self.state.history.append({"role": "system", "content": "[interrupted by user]"})
            
            self.state.last_total_latency = time.perf_counter() - conversation_start
            
            return {
                "success": True,
                "user_message": user_message,
                "response": self.state.last_llm_response,
                "spoken": bool(self.providers.tts),
                "was_interrupted": tts_was_interrupted,
            }
            
        except Exception as e:
            return {"success": False, "error": str(e), "step": "llm"}
    
    async def _speak_with_barge_in(self, text: str) -> bool:
        """Speak text with optional barge-in detection.
        
        Returns True if speech was interrupted.
        """
        # Cancel any pending TTS cleanup task
        if self._tts_cleanup_task and not self._tts_cleanup_task.done():
            self._tts_cleanup_task.cancel()
            try:
                await self._tts_cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Ensure previous TTS is fully stopped before starting new one
        if self.audio_state.output_mode == "reachy" and self.reachy_mini and self.reachy_mini.media:
            try:
                self.reachy_mini.media.stop_playing()
            except Exception as e:
                print(f"[TTS] Error stopping previous Reachy audio: {e}", flush=True)
        
        tts_interrupt_event = None
        barge_in_audio_manager = None
        barge_in_task = None
        
        try:
            # Always create interrupt event for stop functionality
            tts_interrupt_event = asyncio.Event()
            self._tts_interrupt_event = tts_interrupt_event
            
            # Setup barge-in detection if enabled
            tts_audio_queue = None
            if self.interrupt_tts_enabled:
                barge_in_audio_manager = create_audio_manager(
                    self.config,
                    self.audio_state.input_mode,
                    self.reachy_mini.media if self.reachy_mini else None
                )
                tts_audio_queue = await barge_in_audio_manager.start_microphone_stream()
            
            if self.talking_movement_enabled and self.controller:
                self.controller.start_talking()
            
            # Create barge-in monitoring task (only if barge-in is enabled)
            if self.interrupt_tts_enabled and tts_audio_queue and tts_interrupt_event:
                barge_in_task = asyncio.create_task(
                    monitor_barge_in(
                        tts_audio_queue,
                        tts_interrupt_event,
                        self.controller,
                        self.talking_movement_enabled,
                        use_reachy_speaker=(self.audio_state.output_mode == "reachy"),
                        energy_threshold=self.config.stt.energy_threshold,
                        frames_required=self.config.stt.energy_frames_required,
                    )
                )
            
            # Determine audio output
            speaker = self.audio_state.speaker_id if self.audio_state.output_mode != "reachy" else None
            reachy_media = self.reachy_mini.media if (self.audio_state.output_mode == "reachy" and self.reachy_mini) else None
            
            tts_start = time.perf_counter()
            tts_task = asyncio.create_task(
                self.providers.tts.synthesize(
                    text,
                    audio_queue=tts_audio_queue,
                    interrupt_event=tts_interrupt_event,
                    speaker_id=speaker,
                    reachy_media=reachy_media,
                )
            )
            
            # Wait for TTS completion or interrupt
            while not tts_task.done():
                if tts_interrupt_event and tts_interrupt_event.is_set():
                    print("[Conversation] Interrupted, returning to listen immediately...", flush=True)
                    
                    if self.talking_movement_enabled and self.controller:
                        self.controller.stop_talking()
                    
                    if self.audio_state.output_mode == "reachy" and self.reachy_mini and self.reachy_mini.media:
                        try:
                            self.reachy_mini.media.stop_playing()
                        except Exception as e:
                            print(f"[Conversation] Error stopping Reachy audio: {e}", flush=True)
                    
                    # Schedule cleanup as background task (store reference for cancellation)
                    async def cleanup_tts():
                        try:
                            await tts_task
                        except:
                            pass
                        if barge_in_task:
                            barge_in_task.cancel()
                            try:
                                await barge_in_task
                            except asyncio.CancelledError:
                                pass
                        if barge_in_audio_manager:
                            await barge_in_audio_manager.stop_microphone_stream()
                    
                    self._tts_cleanup_task = asyncio.create_task(cleanup_tts())
                    return True
                
                await asyncio.sleep(0.05)
            
            # TTS completed normally
            try:
                tts_result = await tts_task
                self.state.last_tts_latency = time.perf_counter() - tts_start
                return tts_result.was_interrupted
            except Exception as e:
                print(f"[TTS] Error: {e}")
                self.state.last_tts_latency = 0.0
                return False
            
        finally:
            # Cleanup
            if barge_in_task:
                barge_in_task.cancel()
                try:
                    await barge_in_task
                except asyncio.CancelledError:
                    pass
            
            if self.talking_movement_enabled and self.controller:
                self.controller.stop_talking()
            
            if barge_in_audio_manager:
                await barge_in_audio_manager.stop_microphone_stream()
            
            self._tts_interrupt_event = None
    
    async def _antenna_excited_for_duration(self, duration: float = 2.0) -> None:
        """Enable antenna excited mode for a specified duration."""
        if not self.controller:
            return
        print(f"[Vision] Enabling antenna excited mode for {duration}s", flush=True)
        self.controller.antenna_excited_mode = True
        await asyncio.sleep(duration)
        self.controller.antenna_excited_mode = False
        print("[Vision] Antenna excited mode disabled", flush=True)
    
    async def start_conversation_loop(self, stop_event: asyncio.Event) -> None:
        """Run continuous conversation loop with VAD.
        
        Args:
            stop_event: Event to signal when to stop
        """
        self._stop_event = stop_event
        self.state.active = True
        self.state.status = "listening"
        
        while self.state.active and not stop_event.is_set():
            try:
                self.state.is_listening = True
                self.state.status = "listening"
                self.state.last_transcription = ""
                
                if self.providers.stt is None:
                    print("[Conversation] ERROR: STT provider not initialized!", flush=True)
                    self.state.active = False
                    break
                
                print(f"[Conversation] Using STT provider: {self.providers.stt.name}", flush=True)
                
                # Start audio capture
                self._audio_manager = create_audio_manager(
                    self.config,
                    self.audio_state.input_mode,
                    self.reachy_mini.media if self.reachy_mini else None
                )
                audio_queue = await self._audio_manager.start_microphone_stream()
                
                # Transcribe with VAD
                print("\n[Conversation] Waiting for voice...", flush=True)
                stt_start = time.perf_counter()
                result = await self.providers.stt.transcribe(audio_queue, bypass_vad=False)
                self.state.last_stt_latency = time.perf_counter() - stt_start
                
                # Stop audio capture
                await self._audio_manager.stop_microphone_stream()
                self._audio_manager = None
                self.state.is_listening = False
                
                # Check if we got valid transcription
                if result.text and not result.text.startswith("Error"):
                    self.state.last_transcription = result.text
                    print(f"\n[Conversation] You said: {self.state.last_transcription}", flush=True)
                    
                    # Process through LLM and TTS
                    self.state.is_processing = True
                    self.state.status = "processing"
                    conv_result = await self.process_message(self.state.last_transcription)
                    self.state.status = "speaking"
                    self.state.is_processing = False
                    
                    # Pause before next listen cycle to avoid TTS audio feedback
                    # Longer pause when barge-in is disabled (TTS was not interrupted)
                    if not conv_result.get("was_interrupted", False):
                        # Wait for audio to fully clear before starting to listen again
                        # This prevents TTS output from being picked up by STT
                        await asyncio.sleep(1.0)
                else:
                    print("\n[Conversation] No speech detected, continuing...", flush=True)
                    # Wait a bit longer before restarting to prevent rapid mic cycling
                    await asyncio.sleep(0.5)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"\n[Conversation] Error: {e}", flush=True)
                await asyncio.sleep(1.0)
        
        # Cleanup
        self.state.is_listening = False
        self.state.is_processing = False
        self.state.status = "idle"
        self.state.active = False
        
        if self._audio_manager:
            await self._audio_manager.stop_microphone_stream()
            self._audio_manager = None
    
    async def stop_conversation(self) -> None:
        """Stop the conversation loop."""
        self.state.active = False
        self.state.status = "idle"
        
        if self.talking_movement_enabled and self.controller:
            self.controller.stop_talking()
        
        if self._tts_interrupt_event:
            self._tts_interrupt_event.set()
        
        # Cancel any pending TTS cleanup task
        if self._tts_cleanup_task and not self._tts_cleanup_task.done():
            self._tts_cleanup_task.cancel()
            try:
                await self._tts_cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self.audio_state.output_mode == "reachy" and self.reachy_mini and self.reachy_mini.media:
            try:
                self.reachy_mini.media.stop_playing()
            except Exception as e:
                print(f"[Stop] Error stopping Reachy audio: {e}", flush=True)
        
        if self._audio_manager:
            await self._audio_manager.stop_microphone_stream()
            self._audio_manager = None
    
    def get_state(self) -> Dict[str, Any]:
        """Get current conversation state as dictionary."""
        return {
            "conversation_active": self.state.active,
            "conversation_status": self.state.status,
            "is_listening": self.state.is_listening,
            "is_processing": self.state.is_processing,
            "last_transcription": self.state.last_transcription,
            "last_tts_text": self.state.last_tts_text,
            "last_llm_response": self.state.last_llm_response,
            "last_emotion": self.state.last_emotion,
            "conversation_length": len(self.state.history),
            "last_was_interrupted": self.state.last_was_interrupted,
            "has_vision_frame": self.state.last_captured_frame is not None,
            "has_vision_description": self.state.last_vision_description is not None,
            "last_stt_latency": self.state.last_stt_latency,
            "last_llm_latency": self.state.last_llm_latency,
            "last_tts_latency": self.state.last_tts_latency,
            "last_vl_latency": self.state.last_vl_latency,
            "last_total_latency": self.state.last_total_latency,
        }
