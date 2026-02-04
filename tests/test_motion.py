"""Tests for motion/movement skills.

Test the motion module including:
- BreathingMove animation
- TalkingMove animation  
- MovementController
- FaceTrackingWorker
- Emotion loading and mapping

Run standalone:
    pytest tests/test_motion.py -v
    pytest tests/test_motion.py -v -k breathing
    pytest tests/test_motion.py -v -m motion
"""

import asyncio
import time
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from reachy_mini_skills.config import RobotConfig, DEFAULT_SENTIMENT_TO_EMOTION


# =============================================================================
# BreathingMove Tests
# =============================================================================

@pytest.mark.motion
class TestBreathingMove:
    """Tests for the BreathingMove animation."""
    
    def test_breathing_move_initialization(self, robot_config, mock_robot):
        """Test BreathingMove can be initialized."""
        from reachy_mini_skills.actuation.motion import BreathingMove
        
        start_pose = mock_robot.get_current_head_pose()
        _, start_antennas = mock_robot.get_current_joint_positions()
        
        move = BreathingMove(
            interpolation_start_pose=start_pose,
            interpolation_start_antennas=start_antennas,
            config=robot_config,
        )
        
        assert move.duration == float("inf")
        assert move.config == robot_config
    
    def test_breathing_interpolation_phase(self, robot_config, mock_robot):
        """Test breathing move during interpolation phase."""
        from reachy_mini_skills.actuation.motion import BreathingMove
        
        start_pose = mock_robot.get_current_head_pose()
        _, start_antennas = mock_robot.get_current_joint_positions()
        
        move = BreathingMove(
            interpolation_start_pose=start_pose,
            interpolation_start_antennas=start_antennas,
            config=robot_config,
            interpolation_duration=1.0,
        )
        
        # Test at t=0 (start of interpolation)
        head, antennas, body_yaw = move.evaluate(0.0)
        assert head is not None
        assert antennas is not None
        assert body_yaw == 0.0
        
        # Test at t=0.5 (midpoint)
        head, antennas, body_yaw = move.evaluate(0.5)
        assert head is not None
    
    def test_breathing_continuous_phase(self, robot_config, mock_robot):
        """Test breathing move after interpolation (continuous breathing)."""
        from reachy_mini_skills.actuation.motion import BreathingMove
        
        start_pose = mock_robot.get_current_head_pose()
        _, start_antennas = mock_robot.get_current_joint_positions()
        
        move = BreathingMove(
            interpolation_start_pose=start_pose,
            interpolation_start_antennas=start_antennas,
            config=robot_config,
            interpolation_duration=0.5,
        )
        
        # Test breathing phase (after interpolation)
        results = []
        for t in [1.0, 1.5, 2.0, 2.5]:
            head, antennas, body_yaw = move.evaluate(t)
            results.append((head, antennas))
        
        # Antennas should oscillate (different values at different times)
        antenna_values = [r[1][0] for r in results]
        assert len(set(antenna_values)) > 1, "Antennas should vary during breathing"


# =============================================================================
# TalkingMove Tests
# =============================================================================

@pytest.mark.motion
class TestTalkingMove:
    """Tests for the TalkingMove animation."""
    
    def test_talking_move_initialization(self, robot_config, mock_robot):
        """Test TalkingMove can be initialized."""
        from reachy_mini_skills.actuation.motion import TalkingMove
        
        start_pose = mock_robot.get_current_head_pose()
        _, start_antennas = mock_robot.get_current_joint_positions()
        
        move = TalkingMove(
            interpolation_start_pose=start_pose,
            interpolation_start_antennas=start_antennas,
            config=robot_config,
        )
        
        assert move.duration == float("inf")
    
    def test_talking_produces_head_movement(self, robot_config, mock_robot):
        """Test that talking animation produces varied head movements."""
        from reachy_mini_skills.actuation.motion import TalkingMove
        
        start_pose = mock_robot.get_current_head_pose()
        _, start_antennas = mock_robot.get_current_joint_positions()
        
        move = TalkingMove(
            interpolation_start_pose=start_pose,
            interpolation_start_antennas=start_antennas,
            config=robot_config,
            interpolation_duration=0.1,
        )
        
        # Sample positions during talking
        positions = []
        for t in np.linspace(0.2, 2.0, 10):
            head, antennas, _ = move.evaluate(t)
            positions.append(antennas[0])
        
        # Should have variation in positions
        assert max(positions) != min(positions), "Talking should produce movement"


# =============================================================================
# Emotion Tests
# =============================================================================

@pytest.mark.motion
class TestEmotions:
    """Tests for emotion loading and mapping."""
    
    def test_get_emotion_for_sentiment(self):
        """Test emotion selection for sentiment."""
        from reachy_mini_skills.actuation.motion import get_emotion_for_sentiment
        
        emotion_names = ["happy1", "sad1", "attentive1", "dance1", "yes1"]
        
        # Test with default mapping
        emotion = get_emotion_for_sentiment("happy", emotion_names)
        assert emotion is not None
        assert emotion in emotion_names
    
    def test_get_emotion_fallback(self):
        """Test emotion fallback for unknown sentiment."""
        from reachy_mini_skills.actuation.motion import get_emotion_for_sentiment
        
        emotion_names = ["attentive1", "happy1"]
        
        # Unknown sentiment should fall back
        emotion = get_emotion_for_sentiment("unknown_sentiment", emotion_names)
        assert emotion == "attentive1"
    
    def test_get_emotion_custom_mapping(self):
        """Test emotion selection with custom mapping."""
        from reachy_mini_skills.actuation.motion import get_emotion_for_sentiment
        
        emotion_names = ["custom1", "custom2"]
        custom_mapping = {"test": ["custom1", "custom2"]}
        
        emotion = get_emotion_for_sentiment(
            "test", 
            emotion_names, 
            sentiment_mapping=custom_mapping
        )
        assert emotion in ["custom1", "custom2"]
    
    def test_get_emotion_returns_none_if_no_match(self):
        """Test that None is returned when no emotions available."""
        from reachy_mini_skills.actuation.motion import get_emotion_for_sentiment
        
        # Empty list with no fallback
        emotion = get_emotion_for_sentiment("happy", [])
        assert emotion is None


# =============================================================================
# MovementController Tests
# =============================================================================

@pytest.mark.motion
class TestMovementController:
    """Tests for the MovementController."""
    
    def test_controller_initialization(self, robot_config, mock_robot, mock_moves):
        """Test MovementController can be initialized."""
        from reachy_mini_skills.actuation.motion import MovementController
        
        controller = MovementController(
            robot=mock_robot,
            moves=mock_moves,
            emotion_names=mock_moves.list_moves(),
            config=robot_config,
        )
        
        assert controller.robot == mock_robot
        assert controller.config == robot_config
    
    def test_controller_start_stop(self, robot_config, mock_robot, mock_moves):
        """Test starting and stopping the controller."""
        from reachy_mini_skills.actuation.motion import MovementController
        
        controller = MovementController(
            robot=mock_robot,
            moves=mock_moves,
            emotion_names=mock_moves.list_moves(),
            config=robot_config,
        )
        
        controller.start()
        time.sleep(0.1)  # Let thread start
        
        controller.stop()
        # Should not raise
    
    def test_controller_queue_emotion(self, robot_config, mock_robot, mock_moves):
        """Test queuing an emotion."""
        from reachy_mini_skills.actuation.motion import MovementController
        
        controller = MovementController(
            robot=mock_robot,
            moves=mock_moves,
            emotion_names=mock_moves.list_moves(),
            config=robot_config,
        )
        
        controller.start()
        time.sleep(0.1)
        
        controller.queue_emotion("happy1")
        time.sleep(0.2)  # Let it process
        
        controller.stop()
        
        # Should have recorded some targets
        assert len(mock_robot._targets) > 0
    
    def test_controller_talking_mode(self, robot_config, mock_robot, mock_moves):
        """Test talking animation mode."""
        from reachy_mini_skills.actuation.motion import MovementController
        
        controller = MovementController(
            robot=mock_robot,
            moves=mock_moves,
            emotion_names=mock_moves.list_moves(),
            config=robot_config,
        )
        
        controller.start()
        time.sleep(0.1)
        
        controller.start_talking()
        time.sleep(0.3)  # Let talking animation start
        
        controller.stop_talking()
        time.sleep(0.1)
        
        controller.stop()


# =============================================================================
# FaceTrackingWorker Tests
# =============================================================================

@pytest.mark.motion
class TestFaceTrackingWorker:
    """Tests for the FaceTrackingWorker."""
    
    def test_face_tracker_initialization(self, robot_config, mock_robot):
        """Test FaceTrackingWorker can be initialized."""
        from reachy_mini_skills.actuation.motion import FaceTrackingWorker
        
        tracker = FaceTrackingWorker(
            robot=mock_robot,
            config=robot_config,
        )
        
        assert tracker.robot == mock_robot
        assert tracker.config == robot_config
    
    def test_face_tracker_start_stop(self, robot_config, mock_robot):
        """Test starting and stopping the face tracker."""
        from reachy_mini_skills.actuation.motion import FaceTrackingWorker
        
        tracker = FaceTrackingWorker(
            robot=mock_robot,
            config=robot_config,
        )
        
        tracker.start()
        time.sleep(0.1)
        
        offsets = tracker.get_face_tracking_offsets()
        assert len(offsets) == 6
        
        tracker.stop()
    
    def test_face_tracker_offsets_decay(self, robot_config, mock_robot):
        """Test that offsets decay when no face is detected."""
        from reachy_mini_skills.actuation.motion import FaceTrackingWorker
        
        # Robot returns blank frames (no face)
        mock_robot.media.get_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        
        tracker = FaceTrackingWorker(
            robot=mock_robot,
            config=robot_config,
        )
        
        tracker.start()
        time.sleep(0.3)
        
        offsets = tracker.get_face_tracking_offsets()
        # With no face, offsets should be near zero
        assert all(abs(o) < 0.1 for o in offsets)
        
        tracker.stop()


# =============================================================================
# Integration-style Tests
# =============================================================================

@pytest.mark.motion
@pytest.mark.integration
class TestMotionIntegration:
    """Integration tests for motion components working together."""
    
    def test_controller_with_face_tracking(self, robot_config, mock_robot, mock_moves):
        """Test MovementController with FaceTrackingWorker."""
        from reachy_mini_skills.actuation.motion import (
            MovementController,
            FaceTrackingWorker,
        )
        
        face_tracker = FaceTrackingWorker(
            robot=mock_robot,
            config=robot_config,
        )
        
        controller = MovementController(
            robot=mock_robot,
            moves=mock_moves,
            emotion_names=mock_moves.list_moves(),
            config=robot_config,
            face_tracker=face_tracker,
        )
        
        face_tracker.start()
        controller.start()
        time.sleep(0.2)
        
        controller.stop()
        face_tracker.stop()
