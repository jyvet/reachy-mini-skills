"""Tests for configuration module."""

import pytest
from reachy_mini_skills import Config, STTConfig, TTSConfig, LLMConfig


def test_default_config():
    """Test that default configuration can be created."""
    config = Config()
    assert config is not None
    assert config.stt is not None
    assert config.tts is not None
    assert config.llm is not None


def test_stt_config_update():
    """Test STT config update method."""
    config = STTConfig()
    updated = config.update(energy_threshold=0.05)
    assert updated.energy_threshold == 0.05
    assert config.energy_threshold == 0.01  # Original unchanged


def test_tts_config_update():
    """Test TTS config update method."""
    config = TTSConfig()
    updated = config.update(cartesia_sample_rate=44100)
    assert updated.cartesia_sample_rate == 44100
    assert config.cartesia_sample_rate == 22050  # Original unchanged


def test_llm_config_update():
    """Test LLM config update method."""
    config = LLMConfig()
    updated = config.update(cerebras_streaming=True)
    assert updated.cerebras_streaming is True
    assert config.cerebras_streaming is False  # Original unchanged
