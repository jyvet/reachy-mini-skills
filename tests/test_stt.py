"""Tests for Speech-to-Text (STT) skills.

Test the STT module including:
- STTProvider base class
- STTResult data class
- Provider-specific implementations (mocked)
- Real Cartesia API tests (when CARTESIA_API_KEY is set)

Run standalone:
    pytest tests/test_stt.py -v
    pytest tests/test_stt.py -v -k cartesia
    pytest tests/test_stt.py -v -m stt
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from reachy_mini_skills.config import STTConfig
from reachy_mini_skills.perception.audio.stt_base import (
    STTResult,
    STTProvider,
    has_sentence_ending,
    SENTENCE_ENDERS,
)


# Check if Cartesia API key is available for real API tests
CARTESIA_API_KEY = os.environ.get("CARTESIA_API_KEY", "")
HAS_CARTESIA_API_KEY = bool(CARTESIA_API_KEY)


# =============================================================================
# STTResult Tests
# =============================================================================

@pytest.mark.stt
class TestSTTResult:
    """Tests for STTResult data class."""
    
    def test_stt_result_creation(self):
        """Test STTResult can be created."""
        result = STTResult(
            text="Hello world",
            ttfb=0.1,
            total_time=1.5,
        )
        
        assert result.text == "Hello world"
        assert result.ttfb == 0.1
        assert result.total_time == 1.5
        assert result.was_interrupted is False
    
    def test_stt_result_with_interruption(self):
        """Test STTResult with interruption flag."""
        result = STTResult(
            text="Hello",
            ttfb=0.05,
            total_time=0.5,
            was_interrupted=True,
        )
        
        assert result.was_interrupted is True


# =============================================================================
# Sentence Ending Tests
# =============================================================================

@pytest.mark.stt
class TestSentenceEnding:
    """Tests for sentence ending detection."""
    
    def test_has_sentence_ending_period(self):
        """Test detection of period."""
        assert has_sentence_ending("Hello world.") is True
    
    def test_has_sentence_ending_question(self):
        """Test detection of question mark."""
        assert has_sentence_ending("How are you?") is True
    
    def test_has_sentence_ending_exclamation(self):
        """Test detection of exclamation mark."""
        assert has_sentence_ending("Hello!") is True
    
    def test_no_sentence_ending(self):
        """Test text without sentence ending."""
        assert has_sentence_ending("Hello world") is False
    
    def test_empty_string(self):
        """Test empty string."""
        assert has_sentence_ending("") is False
    
    def test_whitespace_handling(self):
        """Test with trailing whitespace."""
        assert has_sentence_ending("Hello.  ") is True
        assert has_sentence_ending("  ") is False


# =============================================================================
# STTConfig Tests
# =============================================================================

@pytest.mark.stt
class TestSTTConfig:
    """Tests for STT configuration."""
    
    def test_default_config(self):
        """Test default STT config values."""
        config = STTConfig()
        
        assert config.energy_threshold == 0.01
        assert config.silence_timeout == 1.2
        assert config.max_duration == 30.0
    
    def test_config_update(self):
        """Test updating STT config."""
        config = STTConfig()
        updated = config.update(energy_threshold=0.05, silence_timeout=2.0)
        
        assert updated.energy_threshold == 0.05
        assert updated.silence_timeout == 2.0
        # Original unchanged
        assert config.energy_threshold == 0.01


# =============================================================================
# Mock STT Provider for Testing
# =============================================================================

class MockSTTProvider(STTProvider):
    """Mock STT provider for testing."""
    
    def __init__(self, config: STTConfig, responses: list = None):
        super().__init__(config)
        self.responses = responses or ["Hello, this is a test."]
        self._call_count = 0
    
    @property
    def name(self) -> str:
        return "mock"
    
    async def transcribe(self, audio_queue, bypass_vad: bool = False) -> STTResult:
        self._call_count += 1
        response = self.responses[min(self._call_count - 1, len(self.responses) - 1)]
        
        # Simulate some processing time
        await asyncio.sleep(0.01)
        
        return STTResult(
            text=response,
            ttfb=0.1,
            total_time=0.5,
        )


# =============================================================================
# STTProvider Tests
# =============================================================================

@pytest.mark.stt
class TestSTTProvider:
    """Tests for STT provider interface."""
    
    @pytest.mark.asyncio
    async def test_mock_provider_transcribe(self, stt_config, mock_audio_queue):
        """Test mock provider transcription."""
        provider = MockSTTProvider(stt_config, responses=["Hello world."])
        
        result = await provider.transcribe(mock_audio_queue)
        
        assert result.text == "Hello world."
        assert result.ttfb > 0
        assert provider._call_count == 1
    
    @pytest.mark.asyncio
    async def test_provider_multiple_calls(self, stt_config, mock_audio_queue):
        """Test provider handles multiple calls."""
        provider = MockSTTProvider(
            stt_config, 
            responses=["First response.", "Second response."]
        )
        
        result1 = await provider.transcribe(mock_audio_queue)
        result2 = await provider.transcribe(mock_audio_queue)
        
        assert result1.text == "First response."
        assert result2.text == "Second response."
    
    def test_provider_name(self, stt_config):
        """Test provider name property."""
        provider = MockSTTProvider(stt_config)
        assert provider.name == "mock"


# =============================================================================
# Audio Queue Tests
# =============================================================================

@pytest.mark.stt
class TestAudioQueue:
    """Tests for audio queue handling."""
    
    @pytest.mark.asyncio
    async def test_audio_queue_speech_detection(self, mock_audio_queue):
        """Test that speech chunks can be added and retrieved."""
        mock_audio_queue.add_speech_chunk(energy=0.1, duration_ms=100)
        
        chunk = await mock_audio_queue.get()
        
        assert len(chunk) > 0
        assert chunk.dtype == np.float32
    
    @pytest.mark.asyncio
    async def test_audio_queue_silence_fallback(self, mock_audio_queue):
        """Test that queue returns silence when empty."""
        # Don't add any chunks
        chunk = await mock_audio_queue.get()
        
        # Should return silence (zeros)
        assert np.allclose(chunk, 0)


# =============================================================================
# Energy/VAD Tests
# =============================================================================

@pytest.mark.stt
class TestVAD:
    """Tests for Voice Activity Detection logic."""
    
    def test_energy_calculation(self):
        """Test RMS energy calculation."""
        from reachy_mini_skills.perception.audio import compute_rms_energy
        
        # Silence
        silence = np.zeros(1000, dtype=np.float32)
        assert compute_rms_energy(silence) == 0.0
        
        # Signal with known energy
        signal = np.ones(1000, dtype=np.float32) * 0.5
        energy = compute_rms_energy(signal)
        assert 0.4 < energy < 0.6
    
    def test_energy_threshold_detection(self, stt_config):
        """Test energy threshold detection."""
        from reachy_mini_skills.perception.audio import compute_rms_energy
        
        # Below threshold
        quiet = np.random.randn(1000).astype(np.float32) * 0.001
        assert compute_rms_energy(quiet) < stt_config.energy_threshold
        
        # Above threshold
        loud = np.random.randn(1000).astype(np.float32) * 0.1
        assert compute_rms_energy(loud) > stt_config.energy_threshold


# =============================================================================
# Cartesia STT Real API Tests
# =============================================================================

@pytest.mark.stt
@pytest.mark.skipif(not HAS_CARTESIA_API_KEY, reason="CARTESIA_API_KEY not set")
class TestCartesiaSTTRealAPI:
    """Real API tests for Cartesia STT provider.
    
    These tests make actual API calls to Cartesia.
    They only run when CARTESIA_API_KEY environment variable is set.
    """
    
    @pytest.fixture
    def cartesia_stt_config(self):
        """Create config with real Cartesia API key."""
        return STTConfig(
            cartesia_api_key=CARTESIA_API_KEY,
            cartesia_sample_rate=16000,
            silence_timeout=1.0,
            max_duration=10.0,
            pre_speech_timeout=5.0,
        )
    
    def test_cartesia_stt_provider_creation(self, cartesia_stt_config):
        """Test Cartesia STT provider can be created."""
        from reachy_mini_skills.perception.audio.stt_cartesia import CartesiaSTT
        
        provider = CartesiaSTT(cartesia_stt_config)
        
        assert provider.name == "cartesia"
        assert provider.config.cartesia_api_key == CARTESIA_API_KEY
    
    def test_cartesia_stt_create_function(self):
        """Test Cartesia STT create() factory function."""
        from reachy_mini_skills.perception.audio import stt_cartesia
        from reachy_mini_skills.config import Config
        
        config = Config()
        provider = stt_cartesia.create(config)
        
        assert provider.name == "cartesia"
        assert isinstance(provider.config, STTConfig)
    
    @pytest.mark.asyncio
    async def test_cartesia_stt_transcribe_with_simulated_audio(self, cartesia_stt_config):
        """Test Cartesia STT transcription with simulated audio.
        
        This test validates the API connection by attempting to transcribe.
        If the API key doesn't have STT permissions (401), the test passes
        with a warning since TTS and STT may have separate permissions.
        """
        from reachy_mini_skills.perception.audio.stt_cartesia import CartesiaSTT
        
        # Use a short timeout config for testing
        test_config = cartesia_stt_config.update(
            pre_speech_timeout=2.0,
            max_duration=3.0,
        )
        provider = CartesiaSTT(test_config)
        
        # Create an audio queue with silence (to test API connectivity)
        audio_queue = asyncio.Queue()
        
        # Pre-populate with some silence chunks
        for _ in range(50):
            silence = np.zeros(test_config.cartesia_chunk_size, dtype=np.float32)
            audio_queue.put_nowait(silence)
        
        try:
            # Run transcription with bypass_vad to skip voice detection wait
            result = await provider.transcribe(audio_queue, bypass_vad=True)
            
            assert isinstance(result, STTResult)
            assert result.total_time > 0
            # Text may be empty since we're sending silence
            assert isinstance(result.text, str)
        except RuntimeError as e:
            # API key may not have STT permissions (separate from TTS)
            if "401" in str(e) or "Unauthorized" in str(e):
                pytest.skip("Cartesia API key doesn't have STT permissions (TTS and STT may require separate access)")
            elif "402" in str(e) or "Payment" in str(e):
                pytest.skip("Cartesia API key has insufficient credits for STT")
            else:
                raise
