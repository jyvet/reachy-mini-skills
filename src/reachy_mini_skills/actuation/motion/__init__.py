"""Motion module - Robot movement, emotions, and tracking.

Usage:
    from reachy_mini_skills.actuation.motion import MotionController
    from reachy_mini_skills.actuation.motion import load_emotions
    from reachy_mini_skills.actuation.motion import FaceTrackingWorker
    from reachy_mini_skills.actuation.motion import parse_emotion_tags, get_animations_for_tag
"""

from .controller import MotionController, SimpleController, BreathingMove, TalkingMove, TransitionToNeutralMove
from .emotions import load_emotions, get_emotion_for_sentiment, parse_emotion_tags, get_animations_for_tag, convert_emotion_tags_for_cartesia
from .face_tracking import FaceTrackingWorker

__all__ = [
    # Primary exports
    "MotionController",
    "FaceTrackingWorker",
    "BreathingMove",
    "TalkingMove",
    "TransitionToNeutralMove",
    "load_emotions",
    "get_emotion_for_sentiment",
    "parse_emotion_tags",
    "get_animations_for_tag",
    "convert_emotion_tags_for_cartesia",
    # Backwards compatibility
    "SimpleController",  # Alias for MotionController
]
