"""Actuation module - Robot motion, emotions, and tracking.

Usage:
    from reachy_mini_skills.actuation import motion
    from reachy_mini_skills.actuation.motion import (
        MotionController,
        BreathingMove,
        TalkingMove,
        TransitionToNeutralMove,
        FaceTrackingWorker,
        load_emotions,
        get_emotion_for_sentiment,
    )
"""

from .motion import (
    MotionController,
    SimpleController,  # Backwards compatibility alias
    BreathingMove,
    TalkingMove,
    TransitionToNeutralMove,
    FaceTrackingWorker,
    load_emotions,
    get_emotion_for_sentiment,
)

__all__ = [
    "MotionController",
    "SimpleController",
    "BreathingMove",
    "TalkingMove",
    "TransitionToNeutralMove",
    "FaceTrackingWorker",
    "load_emotions",
    "get_emotion_for_sentiment",
]