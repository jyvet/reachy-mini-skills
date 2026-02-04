"""Ollama Vision Language model.

Usage:
    from reachy_mini_skills.vision import vl_ollama
    
    vl = vl_ollama.create()
    img_base64 = vl.capture_image(robot)
    description = await vl.describe_image(session, img_base64)
"""

from .visual_input import VisionManager
from ...config import Config

__all__ = ["VisionManager", "create"]


def create(config: Config = None) -> VisionManager:
    """Create an Ollama Vision Language instance."""
    if config is None:
        config = Config()
    return VisionManager(config.vision, config.llm)
