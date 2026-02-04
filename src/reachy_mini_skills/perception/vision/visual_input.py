"""Vision utilities for Reachy Mini Library."""

import base64
import time
from typing import Optional

import aiohttp
import cv2

from ...config import VisionConfig, LLMConfig


class VisionManager:
    """Manages vision-related tasks (camera capture, image description)."""
    
    def __init__(self, vision_config: VisionConfig, llm_config: LLMConfig):
        self.config = vision_config
        self.llm_config = llm_config
    
    def capture_image(self, robot) -> Optional[str]:
        """
        Capture an image from the robot's camera and return it as base64.
        
        Args:
            robot: ReachyMini robot instance
            
        Returns:
            Base64-encoded JPEG image string, or None on failure
        """
        if robot is None:
            print("❌ Robot not available for camera capture")
            return None
        
        try:
            frame = robot.media.get_frame()
            
            if frame is None:
                print("❌ Failed to capture image from robot")
                return None
            
            _, buffer = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            print("📷 Image captured from robot camera!")
            return img_base64
        except Exception as e:
            print(f"❌ Failed to capture image: {e}")
            return None
    
    async def describe_image(
        self,
        session: aiohttp.ClientSession,
        img_base64: str,
        prompt: str = "Describe what you see in this image in a concise way. Focus on the main subjects and actions."
    ) -> str:
        """
        Send an image to vision model and get a description.
        Uses the configured provider (huggingface or ollama).
        
        Args:
            session: aiohttp session
            img_base64: Base64-encoded image
            prompt: Description prompt
            
        Returns:
            Image description string
        """
        if img_base64 is None:
            return "I tried to look but could not access the camera."
        
        provider = getattr(self.config, 'provider', 'ollama')
        
        if provider == "huggingface":
            return await self._describe_image_huggingface(session, img_base64, prompt)
        else:
            return await self._describe_image_ollama(session, img_base64, prompt)
    
    async def _describe_image_huggingface(
        self,
        session: aiohttp.ClientSession,
        img_base64: str,
        prompt: str
    ) -> str:
        """Send image to Hugging Face Inference API for description."""
        start = time.perf_counter()
        
        api_key = self.llm_config.huggingface_api_key
        if not api_key:
            print("❌ Hugging Face API key not set")
            return "I tried to look but the Hugging Face API key is not configured."
        
        model = getattr(self.config, 'huggingface_model', 'Qwen/Qwen2.5-VL-7B-Instruct')
        # Use the Hugging Face router API (OpenAI-compatible endpoint)
        url = "https://router.huggingface.co/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Format for OpenAI-compatible vision API
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "max_tokens": 256
        }
        
        try:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # Extract response from OpenAI-compatible format
                    if "choices" in data and len(data["choices"]) > 0:
                        description = data["choices"][0]["message"]["content"]
                    elif isinstance(data, list) and len(data) > 0:
                        description = data[0].get("generated_text", str(data[0]))
                    elif isinstance(data, dict):
                        description = data.get("generated_text", str(data))
                    else:
                        description = str(data)
                    
                    latency = time.perf_counter() - start
                    print(f"👁️ Vision description via HF ({latency:.2f}s): {description}")
                    return description
                else:
                    error_text = await resp.text()
                    print(f"❌ Hugging Face Vision API error ({resp.status}): {error_text}")
                    return f"I tried to look but there was an error: {resp.status}"
        except Exception as e:
            print(f"❌ Hugging Face Vision model error: {e}")
            return "I tried to look but there was an error processing the image."
    
    async def _describe_image_ollama(
        self,
        session: aiohttp.ClientSession,
        img_base64: str,
        prompt: str
    ) -> str:
        """Send image to Ollama for description."""
        start = time.perf_counter()
        
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [img_base64]
                }
            ],
            "stream": False,
        }
        
        try:
            async with session.post(self.llm_config.ollama_url, json=payload) as resp:
                data = await resp.json()
            
            description = data["message"]["content"]
            latency = time.perf_counter() - start
            print(f"👁️ Vision description via Ollama ({latency:.2f}s): {description}")
            return description
        except Exception as e:
            print(f"❌ Ollama Vision model error: {e}")
            return "I tried to look but there was an error processing the image."
    
    def contains_trigger(self, text: str) -> bool:
        """Check if text contains the vision trigger word."""
        import string
        text_lower = text.lower().translate(str.maketrans('', '', string.punctuation))
        return self.config.trigger_word in text_lower
