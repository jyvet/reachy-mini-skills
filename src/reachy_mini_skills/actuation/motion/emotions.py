"""Emotion loading and mapping for Reachy Mini.

Usage:
    from reachy_mini_skills.actuation.motion import emotions
    
    moves, names = emotions.load_emotions(config)
    emotion = emotions.get_emotion_for_sentiment("happy", names)
    
    # Parse emotion tags from text
    emotions_list, clean_text = emotions.parse_emotion_tags("[happy] Hello there!")
    # emotions_list = ["happy"], clean_text = "Hello there!"
"""

import re
import random
from typing import Dict, List, Optional, Tuple

from reachy_mini.motion.recorded_move import RecordedMoves

from ...config import RobotConfig, DEFAULT_SENTIMENT_TO_EMOTION

__all__ = ["load_emotions", "get_emotion_for_sentiment", "parse_emotion_tags", "get_animations_for_tag", "convert_emotion_tags_for_cartesia"]


def load_emotions(config: Optional[RobotConfig] = None) -> Tuple[RecordedMoves, List[str]]:
    """
    Load all recorded emotion moves from Hugging Face.
    
    Args:
        config: Optional robot config, uses default if not provided
        
    Returns:
        Tuple of (RecordedMoves, list of emotion names)
    """
    if config is None:
        config = RobotConfig()
    
    moves = RecordedMoves(config.emotion_library)
    emotion_names = moves.list_moves()
    return moves, emotion_names


def get_emotion_for_sentiment(
    sentiment: str,
    emotion_names: List[str],
    sentiment_mapping: Optional[Dict[str, List[str]]] = None
) -> Optional[str]:
    """
    Get a random emotion name for the given sentiment.
    
    Args:
        sentiment: Sentiment string (e.g., "happy", "sad")
        emotion_names: List of available emotion names
        sentiment_mapping: Optional custom mapping, uses default if not provided
        
    Returns:
        Random emotion name from available options, or None
    """
    if sentiment_mapping is None:
        sentiment_mapping = DEFAULT_SENTIMENT_TO_EMOTION
    
    possible_actions = sentiment_mapping.get(sentiment.lower(), ["attentive1"])
    available_actions = [a for a in possible_actions if a in emotion_names]
    
    if not available_actions:
        available_actions = ["attentive1"] if "attentive1" in emotion_names else []
    
    if not available_actions:
        return None
    
    return random.choice(available_actions)


def parse_emotion_tags(text: str) -> Tuple[List[str], str]:
    """
    Parse emotion tags from text in the format [tag] or [ tag ] (with optional spaces).
    
    Args:
        text: Text containing emotion tags like "[happy]" or "[ excited ]"
        
    Returns:
        Tuple of (list of emotion tags found, cleaned text with tags removed)
        
    Example:
        >>> parse_emotion_tags("[happy] Hello there! [excited]")
        (['happy', 'excited'], 'Hello there!')
    """
    # Match [tag] with optional whitespace inside brackets
    pattern = r'\[\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\]'
    
    # Find all tags
    tags = re.findall(pattern, text)
    # Normalize to lowercase
    tags = [tag.lower() for tag in tags]
    
    # Remove tags from text
    clean_text = re.sub(pattern, '', text)
    # Clean up extra whitespace
    clean_text = ' '.join(clean_text.split()).strip()
    
    return tags, clean_text


def convert_emotion_tags_for_cartesia(text: str) -> str:
    """
    Convert emotion tags from [tag] format to Cartesia TTS format <emotion value="tag" />.
    
    Args:
        text: Text containing emotion tags like "[happy]" or "[ excited ]"
        
    Returns:
        Text with emotion tags converted to Cartesia format
        
    Example:
        >>> convert_emotion_tags_for_cartesia("[happy] Hello there! [excited]")
        '<emotion value="happy" /> Hello there! <emotion value="excited" />'
    """
    # Match [tag] with optional whitespace inside brackets
    pattern = r'\[\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\]'
    
    def replace_tag(match):
        tag = match.group(1).lower()
        return f'<emotion value="{tag}" />'
    
    # Replace all tags with Cartesia format
    return re.sub(pattern, replace_tag, text)


def get_animations_for_tag(
    tag: str,
    emotion_names: List[str],
    sentiment_mapping: Optional[Dict[str, List[str]]] = None
) -> Optional[str]:
    """
    Get a random animation name for a given emotion tag.
    
    This is an alias for get_emotion_for_sentiment that makes it clearer
    when working with parsed tags from text.
    
    Args:
        tag: Emotion tag (e.g., "happy", "sad", "excited")
        emotion_names: List of available animation names
        sentiment_mapping: Optional custom mapping, uses default if not provided
        
    Returns:
        Random animation name from available options, or None if tag not recognized
    """
    return get_emotion_for_sentiment(tag, emotion_names, sentiment_mapping)
