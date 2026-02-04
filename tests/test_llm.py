"""Tests for Language Model (LLM) skills.

Test the LLM module including:
- LLMProvider base class
- LLMResult data class
- Provider-specific implementations (mocked)

Run standalone:
    pytest tests/test_llm.py -v
    pytest tests/test_llm.py -v -k cerebras
    pytest tests/test_llm.py -v -m llm
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reachy_mini_skills.config import LLMConfig
from reachy_mini_skills.llms.base import LLMResult, LLMProvider


# Check if Cerebras API key is available for integration tests
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY", "")
requires_cerebras_api_key = pytest.mark.skipif(
    not CEREBRAS_API_KEY,
    reason="CEREBRAS_API_KEY environment variable not set"
)


# =============================================================================
# LLMResult Tests
# =============================================================================

@pytest.mark.llm
class TestLLMResult:
    """Tests for LLMResult data class."""
    
    def test_llm_result_creation(self):
        """Test LLMResult can be created."""
        result = LLMResult(
            text="Hello, I'm an AI assistant.",
            latency=0.5,
        )
        
        assert result.text == "Hello, I'm an AI assistant."
        assert result.latency == 0.5
    
    def test_llm_result_empty_response(self):
        """Test LLMResult with empty response."""
        result = LLMResult(text="", latency=0.1)
        
        assert result.text == ""
        assert result.latency == 0.1


# =============================================================================
# LLMConfig Tests
# =============================================================================

@pytest.mark.llm
class TestLLMConfig:
    """Tests for LLM configuration."""
    
    def test_default_config(self):
        """Test default LLM config values."""
        config = LLMConfig()
        
        assert "localhost" in config.ollama_url
        assert config.cerebras_streaming is False
    
    def test_config_update(self):
        """Test updating LLM config."""
        config = LLMConfig()
        updated = config.update(cerebras_streaming=True)
        
        assert updated.cerebras_streaming is True
        # Original unchanged
        assert config.cerebras_streaming is False
    
    def test_ollama_config(self):
        """Test Ollama-specific config values."""
        config = LLMConfig()
        
        assert "/api/chat" in config.ollama_url
        assert config.ollama_default_model is not None


# =============================================================================
# Mock LLM Provider for Testing
# =============================================================================

class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""
    
    def __init__(self, config: LLMConfig, responses: list = None, latency: float = 0.1):
        super().__init__(config)
        self.responses = responses or ["I'm a helpful AI assistant."]
        self.latency = latency
        self._call_count = 0
        self.conversations_received: list = []
    
    @property
    def name(self) -> str:
        return "mock"
    
    @property
    def default_model(self) -> str:
        return "mock-model"
    
    async def complete(self, session, conversation, model=None) -> LLMResult:
        self._call_count += 1
        self.conversations_received.append(conversation)
        
        # Simulate LLM processing
        await asyncio.sleep(self.latency)
        
        response = self.responses[min(self._call_count - 1, len(self.responses) - 1)]
        
        return LLMResult(
            text=response,
            latency=self.latency,
        )


# =============================================================================
# LLMProvider Tests
# =============================================================================

@pytest.mark.llm
class TestLLMProvider:
    """Tests for LLM provider interface."""
    
    @pytest.mark.asyncio
    async def test_mock_provider_complete(self, llm_config, mock_http_session):
        """Test mock provider completion."""
        provider = MockLLMProvider(llm_config)
        
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ]
        
        result = await provider.complete(mock_http_session, conversation)
        
        assert result.text == "I'm a helpful AI assistant."
        assert result.latency > 0
        assert provider._call_count == 1
    
    @pytest.mark.asyncio
    async def test_provider_multiple_turns(self, llm_config, mock_http_session):
        """Test provider handles multi-turn conversations."""
        provider = MockLLMProvider(
            llm_config,
            responses=["First response.", "Second response.", "Third response."],
        )
        
        conversation = [{"role": "user", "content": "Hello"}]
        
        result1 = await provider.complete(mock_http_session, conversation)
        
        conversation.append({"role": "assistant", "content": result1.text})
        conversation.append({"role": "user", "content": "Tell me more"})
        
        result2 = await provider.complete(mock_http_session, conversation)
        
        assert result1.text == "First response."
        assert result2.text == "Second response."
        assert provider._call_count == 2
    
    def test_provider_name(self, llm_config):
        """Test provider name property."""
        provider = MockLLMProvider(llm_config)
        assert provider.name == "mock"
    
    def test_provider_default_model(self, llm_config):
        """Test provider default model property."""
        provider = MockLLMProvider(llm_config)
        assert provider.default_model == "mock-model"


# =============================================================================
# Conversation Tests
# =============================================================================

@pytest.mark.llm
class TestConversation:
    """Tests for conversation handling."""
    
    @pytest.mark.asyncio
    async def test_system_prompt(self, llm_config, mock_http_session):
        """Test that system prompts are passed correctly."""
        provider = MockLLMProvider(llm_config)
        
        system_prompt = "You are Reachy, a friendly robot."
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Who are you?"},
        ]
        
        await provider.complete(mock_http_session, conversation)
        
        received = provider.conversations_received[0]
        assert received[0]["role"] == "system"
        assert received[0]["content"] == system_prompt
    
    @pytest.mark.asyncio
    async def test_conversation_history(self, llm_config, mock_http_session):
        """Test that conversation history is preserved."""
        provider = MockLLMProvider(llm_config, responses=["Response 1", "Response 2"])
        
        conversation = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Question 1"},
        ]
        
        result1 = await provider.complete(mock_http_session, conversation)
        
        # Add to history
        conversation.append({"role": "assistant", "content": result1.text})
        conversation.append({"role": "user", "content": "Question 2"})
        
        await provider.complete(mock_http_session, conversation)
        
        # Second call should have full history
        received = provider.conversations_received[1]
        assert len(received) == 4


# =============================================================================
# Model Selection Tests
# =============================================================================

@pytest.mark.llm
class TestModelSelection:
    """Tests for model selection."""
    
    @pytest.mark.asyncio
    async def test_custom_model(self, llm_config, mock_http_session):
        """Test using a custom model."""
        provider = MockLLMProvider(llm_config)
        
        conversation = [{"role": "user", "content": "Hello"}]
        
        # Pass custom model (mock doesn't use it, but tests the interface)
        result = await provider.complete(
            mock_http_session, 
            conversation, 
            model="custom-model"
        )
        
        assert result is not None


# =============================================================================
# Error Handling Tests
# =============================================================================

@pytest.mark.llm
class TestLLMErrorHandling:
    """Tests for LLM error handling."""
    
    @pytest.mark.asyncio
    async def test_empty_conversation(self, llm_config, mock_http_session):
        """Test handling of empty conversation."""
        provider = MockLLMProvider(llm_config)
        
        # Empty conversation should still work
        result = await provider.complete(mock_http_session, [])
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_response_latency(self, llm_config, mock_http_session):
        """Test that latency is properly measured."""
        provider = MockLLMProvider(llm_config, latency=0.1)
        
        conversation = [{"role": "user", "content": "Hello"}]
        
        import time
        start = time.monotonic()
        result = await provider.complete(mock_http_session, conversation)
        elapsed = time.monotonic() - start
        
        # Should take at least the simulated latency
        assert elapsed >= 0.1
        assert result.latency >= 0.1


# =============================================================================
# Cerebras Provider Tests
# =============================================================================

@pytest.mark.llm
class TestCerebrasProvider:
    """Tests for Cerebras LLM provider."""
    
    def test_cerebras_provider_creation(self, llm_config):
        """Test CerebrasLLM can be instantiated."""
        from reachy_mini_skills.llms.cerebras import CerebrasLLM, CerebrasParams
        
        provider = CerebrasLLM(llm_config)
        
        assert provider.name == "cerebras"
        assert provider.default_model == llm_config.cerebras_default_model
    
    def test_cerebras_params_defaults(self):
        """Test CerebrasParams default values."""
        from reachy_mini_skills.llms.cerebras import CerebrasParams
        
        params = CerebrasParams()
        
        assert params.stream is None
        assert params.temperature == 0.7
        assert params.max_completion_tokens == 1024
        assert params.top_p == 1.0
    
    def test_cerebras_params_custom(self):
        """Test CerebrasParams with custom values."""
        from reachy_mini_skills.llms.cerebras import CerebrasParams
        
        params = CerebrasParams(
            stream=True,
            temperature=0.5,
            max_completion_tokens=2048,
            top_p=0.9,
        )
        
        assert params.stream is True
        assert params.temperature == 0.5
        assert params.max_completion_tokens == 2048
        assert params.top_p == 0.9
    
    def test_cerebras_streaming_config(self, llm_config):
        """Test Cerebras streaming configuration."""
        from reachy_mini_skills.llms.cerebras import CerebrasLLM, CerebrasParams
        
        # Default: use config value
        provider = CerebrasLLM(llm_config)
        assert provider.use_streaming == llm_config.cerebras_streaming
        
        # Override with params
        params = CerebrasParams(stream=True)
        provider_streaming = CerebrasLLM(llm_config, params)
        assert provider_streaming.use_streaming is True
        
        params_no_stream = CerebrasParams(stream=False)
        provider_no_streaming = CerebrasLLM(llm_config, params_no_stream)
        assert provider_no_streaming.use_streaming is False
    
    def test_cerebras_create_factory(self):
        """Test cerebras.create() factory function."""
        from reachy_mini_skills.llms import cerebras
        
        llm = cerebras.create(
            stream=True,
            temperature=0.5,
            max_completion_tokens=512,
        )
        
        assert llm.name == "cerebras"
        assert llm.params.stream is True
        assert llm.params.temperature == 0.5
        assert llm.params.max_completion_tokens == 512
    
    @requires_cerebras_api_key
    def test_cerebras_api_key_from_env(self):
        """Test that Cerebras API key is loaded from environment."""
        config = LLMConfig()
        assert config.cerebras_api_key == CEREBRAS_API_KEY
        assert len(config.cerebras_api_key) > 0
    
    @requires_cerebras_api_key
    @pytest.mark.asyncio
    async def test_cerebras_completion_integration(self):
        """Test Cerebras completion with real API (integration test)."""
        from reachy_mini_skills.llms.cerebras import CerebrasLLM, CerebrasParams
        
        config = LLMConfig()
        params = CerebrasParams(
            stream=False,
            max_completion_tokens=50,
            temperature=0.1,
        )
        provider = CerebrasLLM(config, params)
        
        conversation = [
            {"role": "user", "content": "Say 'hello' and nothing else."},
        ]
        
        # Session is not used by Cerebras provider (it creates its own client)
        result = await provider.complete(None, conversation)
        
        assert result is not None
        assert isinstance(result.text, str)
        assert len(result.text) > 0
        assert result.latency > 0
