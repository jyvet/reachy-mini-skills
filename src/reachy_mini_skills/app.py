"""Base application class for Reachy Mini apps."""

import asyncio
import re
import string
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional

import aiohttp

from .config import Config
from .perception.audio import AudioManager, get_stt_provider, STTProvider
from .perception.vision import VisionManager
from .actuation.speech import get_tts_provider, TTSProvider
from .llms import get_llm_provider, LLMProvider
from .actuation.motion import (
    MotionController,
    load_emotions,
)


@dataclass
class AppSettings:
    """Runtime settings for an app."""
    llm_backend: str = "ollama"
    llm_model: Optional[str] = None
    stt_provider: str = "unmute"
    tts_provider: str = "unmute"
    speaker_id: Optional[int] = None
    text_input: bool = False
    bypass_vad: bool = False
    enable_face_tracking: bool = True
    allow_interruption: bool = True
    enable_robot: bool = True


class ReachyMiniApp(ABC):
    """
    Base class for Reachy Mini applications.
    
    Subclass this to create your own apps with custom system prompts
    and behavior.
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        settings: Optional[AppSettings] = None
    ):
        self.config = config or Config.default()
        self.settings = settings or AppSettings()
        
        # Apply app-specific config customizations
        self.config = self.configure(self.config)
        
        # Components (initialized in setup)
        self.audio_manager: Optional[AudioManager] = None
        self.vision_manager: Optional[VisionManager] = None
        self.stt: Optional[STTProvider] = None
        self.tts: Optional[TTSProvider] = None
        self.llm: Optional[LLMProvider] = None
        
        # Robot components
        self.robot = None
        self.moves = None
        self.emotion_names: List[str] = []
        self.movement_controller: Optional[MotionController] = None
        
        # Conversation state
        self.conversation: List[Dict[str, str]] = []
    
    @property
    @abstractmethod
    def name(self) -> str:
        """App name."""
        pass
    
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt for the LLM."""
        pass
    
    def configure(self, config: Config) -> Config:
        """
        Override this method to customize configuration for your app.
        
        This is called during __init__ before any components are created.
        Return the modified config.
        
        Example:
            def configure(self, config: Config) -> Config:
                return (config
                    .update_vision(trigger_word="picture", model="llava:7b")
                    .update_tts(cartesia_voice_id="my-voice-id")
                    .update_robot(breathing_frequency=0.2))
        
        Args:
            config: The base configuration
            
        Returns:
            Modified configuration
        """
        return config
    
    def setup(self):
        """Initialize all components."""
        print(f"\n[{self.name}] Setting up...")
        
        # Audio manager
        self.audio_manager = AudioManager(self.config.audio)
        
        # Vision manager
        self.vision_manager = VisionManager(self.config.vision, self.config.llm)
        
        # STT provider
        self.stt = get_stt_provider(self.settings.stt_provider, self.config.stt)
        
        # TTS provider
        self.tts = get_tts_provider(self.settings.tts_provider, self.config.tts)
        
        # LLM provider
        self.llm = get_llm_provider(self.settings.llm_backend, self.config.llm)
        
        # Robot setup
        if self.settings.enable_robot:
            self._setup_robot()
        
        # Initialize conversation with system prompt
        self.conversation = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        print(f"[{self.name}] Setup complete")
        print(f"  STT: {self.stt.name}")
        print(f"  TTS: {self.tts.name}")
        print(f"  LLM: {self.llm.name} ({self.settings.llm_model or self.llm.default_model})")
        if self.robot:
            print(f"  Robot: Connected with {len(self.emotion_names)} emotions")
    
    def _setup_robot(self):
        """Set up robot components."""
        try:
            from reachy_mini import ReachyMini
            
            print(f"[{self.name}] Connecting to Reachy Mini...")
            self.robot = ReachyMini()
            self.robot.use_audio = False
            self.robot.__enter__()
            
            self.moves, self.emotion_names = load_emotions(self.config.robot)
            print(f"[{self.name}] Robot connected! Loaded {len(self.emotion_names)} emotions")
            
            # Movement controller (handles face tracking internally)
            self.movement_controller = MotionController(
                self.robot,
                config=self.config.robot,
                moves=self.moves,
                emotion_names=self.emotion_names,
                sentiment_mapping=self.config.sentiment_to_emotion
            )
            # Configure face tracking based on settings
            self.movement_controller.face_tracking_enabled = self.settings.enable_face_tracking
            self.movement_controller.start()
            if self.settings.enable_face_tracking:
                print(f"[{self.name}] Face tracking enabled")
            
        except Exception as e:
            print(f"[{self.name}] Could not connect to robot: {e}")
            print(f"[{self.name}] Continuing without robot...")
            self.robot = None
    
    def cleanup(self):
        """Clean up all resources."""
        print(f"\n[{self.name}] Cleaning up...")
        
        if self.movement_controller:
            self.movement_controller.stop()
        
        if self.robot:
            print(f"[{self.name}] Disconnecting from robot...")
            self.robot.__exit__(None, None, None)
        
        print(f"[{self.name}] Cleanup complete")
    
    async def run(self):
        """Main application loop."""
        self.setup()
        
        try:
            async with aiohttp.ClientSession() as session:
                # Start microphone if not using text input
                if not self.settings.text_input:
                    await self.audio_manager.start_microphone_stream()
                
                while True:
                    try:
                        should_continue = await self._conversation_turn(session)
                        if not should_continue:
                            break
                    except KeyboardInterrupt:
                        print(f"\n[{self.name}] Interrupted by user")
                        break
                    except Exception as e:
                        print(f"\n[{self.name}] Error in conversation turn: {e}")
                        continue
        finally:
            self.cleanup()
    
    async def _conversation_turn(self, session: aiohttp.ClientSession) -> bool:
        """
        Execute one conversation turn.
        
        Returns:
            True to continue, False to exit
        """
        e2e_start = time.perf_counter()
        stt_ttfb = 0
        stt_total = 0
        
        # Get user input
        if self.settings.text_input:
            print("\n⌨️  Type your message (or 'quit' to exit):")
            loop = asyncio.get_running_loop()
            user_text = await loop.run_in_executor(None, input, "> ")
            
            if user_text.lower().strip() in ('quit', 'exit', 'q'):
                print("👋 Goodbye!")
                return False
        else:
            print("\n🎙️ Speak now...")
            result = await self.stt.transcribe(
                self.audio_manager.audio_queue,
                bypass_vad=self.settings.bypass_vad
            )
            user_text = result.text
            stt_ttfb = result.ttfb
            stt_total = result.total_time
        
        # Skip empty input
        if not user_text.strip():
            if self.settings.text_input:
                print("❌ Empty input, try again...")
            else:
                print("🔇 No speech detected, listening again...")
            return True
        
        # Vision detection
        vision_latency = None
        if self.vision_manager.contains_trigger(user_text):
            print(f"👁️ Vision trigger detected: '{self.config.vision.trigger_word}'")
            vision_start = time.perf_counter()
            
            if self.movement_controller:
                self.movement_controller.queue_emotion(
                    self.config.vision.attention_emotion, priority=True
                )
                await asyncio.sleep(0.5)
            
            img_base64 = await asyncio.get_running_loop().run_in_executor(
                None, partial(self.vision_manager.capture_image, self.robot)
            )
            
            description = await self.vision_manager.describe_image(session, img_base64)
            vision_latency = time.perf_counter() - vision_start
            
            user_text_lower = user_text.lower().translate(str.maketrans('', '', string.punctuation))
            user_text = user_text_lower.replace(
                self.config.vision.trigger_word,
                f"{self.config.vision.trigger_word}. You are seeing the following: {description}"
            )
            print(f"📝 User (with vision): {user_text}")
        else:
            print(f"📝 User: {user_text}")
        
        self.conversation.append({"role": "user", "content": user_text})
        
        # LLM response
        llm_result = await self.llm.complete(
            session,
            self.conversation,
            self.settings.llm_model
        )
        reply = llm_result.text
        llm_latency = llm_result.latency
        
        self.conversation.append({"role": "assistant", "content": reply})
        print(f"🤖 Assistant ({self.llm.name}): {reply}")
        
        # Parse sentiment
        sentiment_match = re.match(r'^\[([^\]]+)\]\s*', reply)
        if sentiment_match:
            sentiment = sentiment_match.group(1).strip()
            tts_text = reply[sentiment_match.end():].strip()
            print(f"🎭 Sentiment: {sentiment}")
        else:
            sentiment = "neutral"
            tts_text = reply
            print(f"🎭 Sentiment: {sentiment} (default)")
        
        # Robot emotion
        if self.movement_controller:
            self.movement_controller.queue_emotion_for_sentiment(sentiment, priority=True)
            self.movement_controller.start_talking()
        
        # TTS
        interrupt_event = asyncio.Event()
        barge_in_queue = None if (self.settings.text_input or not self.settings.allow_interruption) else self.audio_manager.audio_queue
        
        tts_result = await self.tts.synthesize(
            tts_text,
            barge_in_queue,
            interrupt_event,
            self.settings.speaker_id,
            emotion=sentiment
        )
        
        # Stop talking animation
        if self.movement_controller:
            self.movement_controller.stop_talking()
        
        e2e = time.perf_counter() - e2e_start
        
        # Metrics
        if tts_result.was_interrupted:
            print("\n⏹️ Response was interrupted by user")
        
        print("\n📊 Latency metrics")
        print(f"STT TTFB      : {stt_ttfb:.3f}s")
        print(f"STT total     : {stt_total:.3f}s")
        if vision_latency is not None:
            print(f"Vision latency: {vision_latency:.3f}s")
        print(f"LLM backend   : {self.llm.name} ({self.settings.llm_model or self.llm.default_model})")
        print(f"LLM latency   : {llm_latency:.3f}s")
        print(f"TTS TTFB      : {tts_result.ttfb:.3f}s")
        print(f"TTS total     : {tts_result.total_time:.3f}s")
        
        return True
