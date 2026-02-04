"""API models and utilities for Reachy Mini applications.

Provides reusable Pydantic models and helper functions for building
FastAPI-based applications with the reachy_mini_skills library.
"""

import re
from typing import Any, Dict, List, Optional

import aiohttp
from pydantic import BaseModel


# ============================================================================
# Pydantic Request/Response Models
# ============================================================================

class ToggleState(BaseModel):
    """Toggle a boolean setting on/off."""
    enabled: bool


class ApiKeyUpdate(BaseModel):
    """Update an API key."""
    api_key: str


class TTSRequest(BaseModel):
    """Request TTS synthesis."""
    text: str


class LLMRequest(BaseModel):
    """Request LLM completion."""
    message: str
    clear_history: bool = False


class SystemPromptUpdate(BaseModel):
    """Update system prompt."""
    prompt: str


class ProviderUpdate(BaseModel):
    """Update a provider selection."""
    provider: str


class VoiceIdUpdate(BaseModel):
    """Update Cartesia voice ID."""
    voice_id: str


class SpeedUpdate(BaseModel):
    """Update TTS speed."""
    speed: str


class ModelUpdate(BaseModel):
    """Update model selection."""
    model: str


class VolumeUpdate(BaseModel):
    """Update volume level."""
    volume: int


class EmotionRequest(BaseModel):
    """Request emotion animation."""
    emotion: str = ""
    sentiment: str = ""


class KyutaiTTSConfigUpdate(BaseModel):
    """Update Kyutai TTS configuration."""
    stream_url: str
    voice: str


class KyutaiSTTConfigUpdate(BaseModel):
    """Update Kyutai STT configuration."""
    url: str


class WhisperSocketConfigUpdate(BaseModel):
    """Update WhisperSocket configuration."""
    url: str


class OllamaConfigUpdate(BaseModel):
    """Update Ollama configuration."""
    host: str
    port: int
    model: Optional[str] = None


class HuggingFaceConfigUpdate(BaseModel):
    """Update Hugging Face configuration."""
    model: Optional[str] = None
    api_url: Optional[str] = None


class VADConfigUpdate(BaseModel):
    """Update VAD configuration."""
    energy_threshold: Optional[float] = None
    energy_frames_required: Optional[int] = None
    pre_speech_timeout: Optional[float] = None
    silence_timeout: Optional[float] = None
    silence_timeout_with_punct: Optional[float] = None
    max_duration: Optional[float] = None


class AudioOutputUpdate(BaseModel):
    """Update audio output configuration."""
    mode: str  # "reachy", "system", "custom"
    speaker_id: Optional[int] = None


class AudioInputUpdate(BaseModel):
    """Update audio input configuration."""
    mode: str  # "reachy", "system", "custom"
    mic_id: Optional[int] = None


class VisionConfigUpdate(BaseModel):
    """Update Ollama vision configuration."""
    host: str
    port: int
    model: Optional[str] = None
    trigger_word: Optional[str] = None


class VisionHuggingFaceConfigUpdate(BaseModel):
    """Update Hugging Face vision configuration."""
    model: str
    trigger_word: Optional[str] = None


# ============================================================================
# Audio Device Helpers
# ============================================================================

def list_audio_speakers() -> List[Dict[str, Any]]:
    """List available audio output devices.
    
    Returns:
        List of speaker dictionaries with id, name, channels, sample_rate
    """
    import sounddevice as sd
    devices = sd.query_devices()
    speakers = []
    for i, dev in enumerate(devices):
        if dev['max_output_channels'] > 0:
            speakers.append({
                "id": i,
                "name": dev['name'],
                "channels": dev['max_output_channels'],
                "sample_rate": int(dev['default_samplerate'])
            })
    return speakers


def list_audio_microphones() -> List[Dict[str, Any]]:
    """List available audio input devices.
    
    Returns:
        List of microphone dictionaries with id, name, channels, sample_rate
    """
    import sounddevice as sd
    devices = sd.query_devices()
    microphones = []
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            microphones.append({
                "id": i,
                "name": dev['name'],
                "channels": dev['max_input_channels'],
                "sample_rate": int(dev['default_samplerate'])
            })
    return microphones


# ============================================================================
# Model Fetching Helpers
# ============================================================================

async def fetch_huggingface_llm_models(
    current_model: str = "",
    min_params_b: int = 70,
    exclude_keywords: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Fetch available LLM models from Hugging Face with Cerebras inference provider.
    
    Args:
        current_model: Currently selected model ID
        min_params_b: Minimum parameter size in billions
        exclude_keywords: Keywords to exclude (default: code-related models)
        
    Returns:
        Dict with success, models list, current_model, and optional error
    """
    if exclude_keywords is None:
        exclude_keywords = ["coder", "code", "codegen", "starcoder", "codellama", 
                          "deepseek-coder", "glm", "thinking"]
    
    try:
        url = "https://huggingface.co/api/models?inference_provider=cerebras&limit=50&pipeline_tag=text-generation"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    models = []
                    for m in data:
                        model_id = m.get("id", "")
                        model_name_lower = model_id.lower()
                        
                        # Skip excluded keywords
                        if any(kw in model_name_lower for kw in exclude_keywords):
                            continue
                        
                        # Check parameter size
                        param_match = re.search(r'(\d+)b', model_name_lower)
                        if param_match:
                            param_size = int(param_match.group(1))
                            if param_size < min_params_b:
                                continue
                        
                        models.append({
                            "id": f"{model_id}:cerebras",
                            "name": model_id.split("/")[-1] if "/" in model_id else model_id,
                            "full_name": model_id,
                            "likes": m.get("likes", 0),
                            "downloads": m.get("downloads", 0)
                        })
                    return {"success": True, "models": models, "current_model": current_model}
                else:
                    return {"success": False, "error": f"HTTP {response.status}", "models": []}
    except Exception as e:
        return {"success": False, "error": str(e), "models": []}


async def fetch_huggingface_vl_models(current_model: str = "") -> Dict[str, Any]:
    """Fetch available Vision-Language models from Hugging Face.
    
    Args:
        current_model: Currently selected model ID
        
    Returns:
        Dict with success, models list, current_model, and optional error
    """
    try:
        url = "https://huggingface.co/api/models?inference_provider=novita&limit=50&pipeline_tag=image-text-to-text"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    models = []
                    for m in data:
                        model_id = m.get("id", "")
                        # Only include models with "VL" in the name
                        if "VL" not in model_id.upper():
                            continue
                        models.append({
                            "id": model_id,
                            "name": model_id.split("/")[-1] if "/" in model_id else model_id,
                            "full_name": model_id,
                            "likes": m.get("likes", 0),
                            "downloads": m.get("downloads", 0)
                        })
                    return {"success": True, "models": models, "current_model": current_model}
                else:
                    return {"success": False, "error": f"HTTP {response.status}", "models": []}
    except Exception as e:
        return {"success": False, "error": str(e), "models": []}


def fetch_cerebras_models(api_key: str, current_model: str = "") -> Dict[str, Any]:
    """Fetch available models from Cerebras.
    
    Args:
        api_key: Cerebras API key
        current_model: Currently selected model ID
        
    Returns:
        Dict with success, models list, current_model, and optional error
    """
    if not api_key:
        return {"success": False, "error": "Cerebras API key not set", "models": []}
    
    try:
        from cerebras.cloud.sdk import Cerebras
        client = Cerebras(api_key=api_key)
        models_response = client.models.list()
        models = [{"id": m.id, "owned_by": m.owned_by} for m in models_response.data]
        return {"success": True, "models": models, "current_model": current_model}
    except Exception as e:
        return {"success": False, "error": str(e), "models": []}


# ============================================================================
# API Key Validation
# ============================================================================

def validate_provider_api_keys(
    stt_provider: str,
    tts_provider: str,
    llm_provider: str,
    cartesia_api_key: str = "",
    cerebras_api_key: str = "",
    huggingface_api_key: str = "",
) -> Optional[str]:
    """Validate that required API keys are set for selected providers.
    
    Args:
        stt_provider: Selected STT provider
        tts_provider: Selected TTS provider
        llm_provider: Selected LLM provider
        cartesia_api_key: Cartesia API key
        cerebras_api_key: Cerebras API key
        huggingface_api_key: Hugging Face API key
        
    Returns:
        Error message if validation fails, None if all required keys are set
    """
    if stt_provider == "cartesia" and not cartesia_api_key:
        return "Cartesia API key not set (required for Cartesia STT)"
    if tts_provider == "cartesia" and not cartesia_api_key:
        return "Cartesia API key not set (required for Cartesia TTS)"
    if llm_provider == "cerebras" and not cerebras_api_key:
        return "Cerebras API key not set"
    if llm_provider == "huggingface" and not huggingface_api_key:
        return "Hugging Face API key not set"
    return None


def get_api_key_preview(api_key: str, preview_length: int = 8) -> str:
    """Get a safe preview of an API key.
    
    Args:
        api_key: The API key
        preview_length: Number of characters to show
        
    Returns:
        Preview string like "abc12345..." or empty string
    """
    if api_key and len(api_key) > preview_length:
        return f"{api_key[:preview_length]}..."
    return ""
