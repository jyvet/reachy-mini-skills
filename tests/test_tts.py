"""Tests for Text-to-Speech (TTS) skills.

Test the TTS module including:
- TTSProvider base class
- TTSResult data class
- Provider-specific implementations (mocked)
- Real Cartesia API tests (when CARTESIA_API_KEY is set)

Run standalone:
    pytest tests/test_tts.py -v
    pytest tests/test_tts.py -v -k cartesia
    pytest tests/test_tts.py -v -m tts
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reachy_mini_skills.config import TTSConfig
from reachy_mini_skills.actuation.speech.tts_base import TTSResult, TTSProvider


# Check if Cartesia API key is available for real API tests
CARTESIA_API_KEY = os.environ.get("CARTESIA_API_KEY", "")
HAS_CARTESIA_API_KEY = bool(CARTESIA_API_KEY)


# =============================================================================
# TTSResult Tests
# =============================================================================

@pytest.mark.tts
class TestTTSResult:
    """Tests for TTSResult data class."""
    
    def test_tts_result_creation(self):
        """Test TTSResult can be created."""
        result = TTSResult(
            ttfb=0.1,
            total_time=2.5,
        )
        
        assert result.ttfb == 0.1
        assert result.total_time == 2.5
        assert result.was_interrupted is False
    
    def test_tts_result_with_interruption(self):
        """Test TTSResult with interruption flag."""
        result = TTSResult(
            ttfb=0.05,
            total_time=1.0,
            was_interrupted=True,
        )
        
        assert result.was_interrupted is True


# =============================================================================
# TTSConfig Tests
# =============================================================================

@pytest.mark.tts
class TestTTSConfig:
    """Tests for TTS configuration."""
    
    def test_default_config(self):
        """Test default TTS config values."""
        config = TTSConfig()
        
        assert config.cartesia_sample_rate == 22050
        assert config.cartesia_voice_id == "cbaf8084-f009-4838-a096-07ee2e6612b1"
    
    def test_config_update(self):
        """Test updating TTS config."""
        config = TTSConfig()
        updated = config.update(cartesia_sample_rate=44100)
        
        assert updated.cartesia_sample_rate == 44100
        # Original unchanged
        assert config.cartesia_sample_rate == 22050
    
    def test_unmute_config(self):
        """Test Unmute-specific config values."""
        config = TTSConfig()
        
        assert "ws://" in config.unmute_stream
        assert config.unmute_api_key == "public_token"


# =============================================================================
# Mock TTS Provider for Testing
# =============================================================================

class MockTTSProvider(TTSProvider):
    """Mock TTS provider for testing."""
    
    def __init__(self, config: TTSConfig, latency: float = 0.1):
        super().__init__(config)
        self.latency = latency
        self._call_count = 0
        self.synthesized_texts: list = []
    
    @property
    def name(self) -> str:
        return "mock"
    
    async def synthesize(
        self,
        text: str,
        audio_queue=None,
        interrupt_event=None,
        speaker_id=None,
        emotion=None,
    ) -> TTSResult:
        self._call_count += 1
        self.synthesized_texts.append(text)
        
        # Simulate synthesis time
        await asyncio.sleep(self.latency)
        
        # Check for interruption
        was_interrupted = False
        if interrupt_event and interrupt_event.is_set():
            was_interrupted = True
        
        return TTSResult(
            ttfb=self.latency * 0.5,
            total_time=self.latency + len(text) * 0.01,
            was_interrupted=was_interrupted,
        )


# =============================================================================
# TTSProvider Tests
# =============================================================================

@pytest.mark.tts
class TestTTSProvider:
    """Tests for TTS provider interface."""
    
    @pytest.mark.asyncio
    async def test_mock_provider_synthesize(self, tts_config):
        """Test mock provider synthesis."""
        provider = MockTTSProvider(tts_config)
        
        result = await provider.synthesize("Hello, world!")
        
        assert result.ttfb > 0
        assert result.total_time > result.ttfb
        assert result.was_interrupted is False
        assert provider._call_count == 1
        assert provider.synthesized_texts == ["Hello, world!"]
    
    @pytest.mark.asyncio
    async def test_provider_multiple_calls(self, tts_config):
        """Test provider handles multiple synthesis calls."""
        provider = MockTTSProvider(tts_config, latency=0.01)
        
        result1 = await provider.synthesize("First message.")
        result2 = await provider.synthesize("Second message.")
        
        assert provider._call_count == 2
        assert provider.synthesized_texts == ["First message.", "Second message."]
    
    @pytest.mark.asyncio
    async def test_provider_interruption(self, tts_config):
        """Test provider handles interruption."""
        provider = MockTTSProvider(tts_config, latency=0.01)
        
        interrupt_event = asyncio.Event()
        interrupt_event.set()  # Pre-set interruption
        
        result = await provider.synthesize(
            "This will be interrupted",
            interrupt_event=interrupt_event,
        )
        
        assert result.was_interrupted is True
    
    def test_provider_name(self, tts_config):
        """Test provider name property."""
        provider = MockTTSProvider(tts_config)
        assert provider.name == "mock"


# =============================================================================
# Audio Output Tests
# =============================================================================

@pytest.mark.tts
class TestAudioOutput:
    """Tests for audio output handling."""
    
    @pytest.mark.asyncio
    async def test_audio_queue_integration(self, tts_config):
        """Test TTS with audio queue for barge-in detection."""
        provider = MockTTSProvider(tts_config, latency=0.01)
        
        audio_queue = asyncio.Queue()
        
        result = await provider.synthesize(
            "Testing audio queue",
            audio_queue=audio_queue,
        )
        
        assert result is not None


# =============================================================================
# Voice/Emotion Tests
# =============================================================================

@pytest.mark.tts
class TestVoiceEmotions:
    """Tests for voice and emotion handling."""
    
    @pytest.mark.asyncio
    async def test_synthesis_with_emotion(self, tts_config):
        """Test synthesis with emotion parameter."""
        provider = MockTTSProvider(tts_config)
        
        result = await provider.synthesize(
            "I'm happy!",
            emotion="happy",
        )
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_synthesis_with_speaker(self, tts_config):
        """Test synthesis with speaker ID."""
        provider = MockTTSProvider(tts_config)
        
        result = await provider.synthesize(
            "Custom speaker",
            speaker_id=1,
        )
        
        assert result is not None


# =============================================================================
# Timing Tests
# =============================================================================

@pytest.mark.tts
class TestTTSTiming:
    """Tests for TTS timing measurements."""
    
    @pytest.mark.asyncio
    async def test_ttfb_measurement(self, tts_config):
        """Test time-to-first-byte measurement."""
        provider = MockTTSProvider(tts_config, latency=0.1)
        
        result = await provider.synthesize("Test timing")
        
        # TTFB should be positive and less than total time
        assert result.ttfb > 0
        assert result.ttfb < result.total_time
    
    @pytest.mark.asyncio
    async def test_longer_text_takes_more_time(self, tts_config):
        """Test that longer text takes more time."""
        provider = MockTTSProvider(tts_config, latency=0.01)
        
        short_result = await provider.synthesize("Hi")
        long_result = await provider.synthesize("This is a much longer sentence that should take more time to synthesize.")
        
        assert long_result.total_time > short_result.total_time


# =============================================================================
# Cartesia TTS Real API Tests
# =============================================================================

@pytest.mark.tts
@pytest.mark.skipif(not HAS_CARTESIA_API_KEY, reason="CARTESIA_API_KEY not set")
class TestCartesiaTTSRealAPI:
    """Real API tests for Cartesia TTS provider.
    
    These tests make actual API calls to Cartesia.
    They only run when CARTESIA_API_KEY environment variable is set.
    """
    
    @pytest.fixture
    def cartesia_config(self):
        """Create config with real Cartesia API key."""
        return TTSConfig(
            cartesia_api_key=CARTESIA_API_KEY,
            cartesia_sample_rate=22050,
            cartesia_voice_id="cbaf8084-f009-4838-a096-07ee2e6612b1",
        )
    
    @pytest.mark.asyncio
    async def test_cartesia_tts_simple_synthesis(self, cartesia_config):
        """Test Cartesia TTS can synthesize simple text."""
        from reachy_mini_skills.actuation.speech.tts_cartesia import CartesiaTTS
        
        provider = CartesiaTTS(cartesia_config)
        
        assert provider.name == "cartesia"
        
        # Test synthesis (audio will play if speakers available)
        result = await provider.synthesize("Hello, this is a test.")
        
        assert isinstance(result, TTSResult)
        assert result.ttfb >= 0
        assert result.total_time > 0
        assert result.was_interrupted is False
    
    @pytest.mark.asyncio
    async def test_cartesia_tts_with_emotion(self, cartesia_config):
        """Test Cartesia TTS with emotion parameter."""
        from reachy_mini_skills.actuation.speech.tts_cartesia import CartesiaTTS
        
        provider = CartesiaTTS(cartesia_config)
        
        result = await provider.synthesize(
            "I am so happy to see you!",
            emotion="happy",
        )
        
        assert isinstance(result, TTSResult)
        assert result.total_time > 0
    
    @pytest.mark.asyncio
    async def test_cartesia_tts_timing_metrics(self, cartesia_config):
        """Test that Cartesia TTS returns valid timing metrics."""
        from reachy_mini_skills.actuation.speech.tts_cartesia import CartesiaTTS
        
        provider = CartesiaTTS(cartesia_config)
        
        result = await provider.synthesize("Testing timing metrics.")
        
        # TTFB should be reasonable (less than 5 seconds)
        assert result.ttfb < 5.0
        # Total time should be greater than or equal to TTFB
        assert result.total_time >= result.ttfb
    
    @pytest.mark.asyncio
    async def test_cartesia_tts_create_function(self):
        """Test Cartesia TTS create() factory function."""
        from reachy_mini_skills.actuation.speech import tts_cartesia
        from reachy_mini_skills.config import Config
        
        config = Config()
        provider = tts_cartesia.create(config)
        
        assert provider.name == "cartesia"
        assert isinstance(provider.config, TTSConfig)
