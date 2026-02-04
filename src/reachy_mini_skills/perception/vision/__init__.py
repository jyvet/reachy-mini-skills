"""Vision perception module - Vision Language models.

Usage:
    from reachy_mini_skills.perception.vision import vl_ollama
    
    vl = vl_ollama.create()
    img = vl.capture_image(robot)
    description = await vl.describe_image(session, img)
"""

from .visual_input import VisionManager
from . import vl_ollama

__all__ = [
    "VisionManager",
    "vl_ollama",
]
