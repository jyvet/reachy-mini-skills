"""Motion controller for Reachy Mini.

Unified controller that handles face tracking, breathing, talking animations,
and optional emotion playback. Designed for easy use in apps with toggle properties.

Usage (basic - face tracking + breathing):
    from reachy_mini_skills.actuation.motion import MotionController
    
    controller = MotionController(robot)
    controller.start()
    
    # Toggle features via properties
    controller.face_tracking_enabled = False
    controller.breathing_enabled = True
    
    controller.stop()

Usage (with emotions):
    from reachy_mini_skills.actuation.motion import MotionController, load_emotions
    
    moves, emotion_names = load_emotions(robot, "emotion_library_name")
    controller = MotionController(robot, moves=moves, emotion_names=emotion_names)
    controller.start()
    
    controller.queue_emotion("happy1")
    controller.start_talking()
    controller.stop_talking()
    
    controller.stop()
"""

import queue
import random
import threading
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
from reachy_mini.utils import create_head_pose
from reachy_mini.utils.interpolation import linear_pose_interpolation, compose_world_offset

from ...config import RobotConfig, DEFAULT_SENTIMENT_TO_EMOTION
from .face_tracking import FaceTrackingWorker

__all__ = ["MotionController", "BreathingMove", "TalkingMove", "TransitionToNeutralMove", "SimpleController"]


# =============================================================================
# Animation Classes
# =============================================================================

def _smooth_step(t: float) -> float:
    """Smoothstep easing function for smooth start and end (zero velocity at boundaries)."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class BreathingMove:
    """Breathing move with interpolation to neutral and then continuous breathing patterns."""

    def __init__(
        self,
        interpolation_start_pose,
        interpolation_start_antennas,
        config: RobotConfig,
        interpolation_duration: float = 1.0,
        config_provider=None,
    ):
        self.interpolation_start_pose = interpolation_start_pose
        self.interpolation_start_antennas = np.array(interpolation_start_antennas)
        self.interpolation_duration = interpolation_duration
        self._static_config = config
        self._config_provider = config_provider  # Callable that returns current config

        self.neutral_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
        self.neutral_antennas = np.array([0.0, 0.0])
        
        # Breathing cycle state for hold-at-top behavior
        self._cycle_start_time = None  # When current cycle started
        self._current_hold_duration = 0.0  # Random hold duration for current cycle (1-5s)
        self._in_hold_phase = False  # Whether we're currently holding at top
    
    @property
    def config(self) -> RobotConfig:
        """Get current config - from provider if available, otherwise static."""
        if self._config_provider is not None:
            return self._config_provider()
        return self._static_config

    @property
    def duration(self) -> float:
        return float("inf")

    def evaluate(self, t: float):
        if t < self.interpolation_duration:
            # Use smoothstep for smooth acceleration/deceleration (zero velocity at start and end)
            interpolation_t = _smooth_step(t / self.interpolation_duration)
            head_pose = linear_pose_interpolation(
                self.interpolation_start_pose, self.neutral_head_pose, interpolation_t,
            )
            antennas_interp = (
                1 - interpolation_t
            ) * self.interpolation_start_antennas + interpolation_t * self.neutral_antennas
            antennas = antennas_interp.astype(np.float64)
        else:
            breathing_time = t - self.interpolation_duration
            
            # Smooth amplitude ramp-up over first breathing cycle to avoid jerk
            base_period = 1.0 / self.config.breathing_frequency
            ramp_duration = base_period
            amplitude_factor = min(1.0, breathing_time / ramp_duration) if ramp_duration > 0 else 1.0
            # Use smoothstep for the ramp to ensure zero acceleration at boundaries
            amplitude_factor = _smooth_step(amplitude_factor)
            
            # Initialize cycle tracking on first call
            if self._cycle_start_time is None:
                self._cycle_start_time = breathing_time
                self._current_hold_duration = random.uniform(2.0, 8.0)
            
            # Calculate time within current cycle
            cycle_time = breathing_time - self._cycle_start_time
            
            # Breathing phases (starting at top position):
            # 1. Hold at top: random 2-8 seconds
            # 2. Exhale (fall): very fast, takes ~10% of base period
            # 3. Inhale (rise): slower, takes ~90% of base period (no hold at bottom)
            exhale_duration = base_period * 0.1
            inhale_duration = base_period * 0.9
            total_cycle_duration = self._current_hold_duration + exhale_duration + inhale_duration
            
            if cycle_time >= total_cycle_duration:
                # Start new cycle with new random hold duration
                # Carry over excess time to avoid any pause
                excess_time = cycle_time - total_cycle_duration
                self._cycle_start_time = breathing_time - excess_time
                self._current_hold_duration = random.uniform(2.0, 8.0)
                cycle_time = excess_time
            
            # Determine which phase we're in and calculate wave progress
            if cycle_time < self._current_hold_duration:
                # Hold at top phase - stay at peak (start position)
                smooth_wave = 1.0
            elif cycle_time < self._current_hold_duration + exhale_duration:
                # Exhale phase - fast LINEAR fall (no slowdown at bottom)
                exhale_time = cycle_time - self._current_hold_duration
                phase_progress = min(1.0, exhale_time / exhale_duration)
                # Linear mapping from top to bottom: +1 to -1
                smooth_wave = 1.0 - 2.0 * phase_progress
            else:
                # Inhale phase - slow smooth rise
                inhale_time = cycle_time - self._current_hold_duration - exhale_duration
                phase_progress = inhale_time / inhale_duration
                # Use smoothstep for smooth acceleration at start
                smooth_progress = _smooth_step(phase_progress)
                # Map 0->1 to bottom->top: -1 to +1
                smooth_wave = -1.0 + 2.0 * smooth_progress
            
            # Translate the wave upwards so both positions are higher
            # Wave goes from -1 to +1, adding 0.5 shifts it to -0.5 to +1.5
            smooth_wave += 0.5
            
            # Head breathing (z movement) - only if enabled
            if self.config.head_breathing_enabled:
                z_offset = amplitude_factor * self.config.breathing_z_amplitude * smooth_wave
            else:
                z_offset = 0.0
            head_pose = create_head_pose(x=0, y=0, z=z_offset, roll=0, pitch=0, yaw=0, degrees=True, mm=False)
            
            # Antenna breathing - only if enabled
            if self.config.antenna_breathing_enabled:
                # Use excited frequency if in excited mode, otherwise normal frequency
                current_antenna_frequency = (
                    self.config.antenna_excited_frequency 
                    if self.config.antenna_excited_mode 
                    else self.config.antenna_frequency
                )
                
                # Same smooth ramp for antenna sway
                antenna_ramp_duration = 1.0 / current_antenna_frequency
                antenna_amplitude_factor = min(1.0, breathing_time / antenna_ramp_duration) if antenna_ramp_duration > 0 else 1.0
                antenna_amplitude_factor = _smooth_step(antenna_amplitude_factor)
                
                # Use raised cosine for antenna as well
                antenna_phase = 2 * np.pi * current_antenna_frequency * breathing_time
                antenna_smooth_wave = -np.cos(antenna_phase)
                
                antenna_sway = antenna_amplitude_factor * np.deg2rad(self.config.antenna_sway_amplitude_deg) * antenna_smooth_wave
                antennas = np.array([antenna_sway, -antenna_sway], dtype=np.float64)
            else:
                antennas = np.array([0.0, 0.0], dtype=np.float64)

        return (head_pose, antennas, 0.0)


class TalkingMove:
    """Talking animation with small head movements to simulate speaking.
    
    Returns oscillation OFFSETS (not absolute poses) that should be composed
    on top of the base pose (e.g., face tracking). The offsets ramp up smoothly
    when starting and can be ramped down by calling start_ramp_down().
    """

    def __init__(
        self,
        config: RobotConfig,
        ramp_duration: float = 0.3,
    ):
        self.config = config
        self.ramp_duration = ramp_duration
        self._amplitude_factor = 0.0  # Current amplitude (0 to 1)
        self._ramping_down = False
        self._start_time = None

    @property
    def duration(self) -> float:
        return float("inf")
    
    def start_ramp_down(self):
        """Start ramping down the amplitude for smooth stop."""
        self._ramping_down = True
    
    def is_finished(self) -> bool:
        """Return True if ramp down is complete."""
        return self._ramping_down and self._amplitude_factor <= 0.001

    def evaluate(self, t: float):
        """Return (head_offset_pose, antenna_offsets, body_yaw_offset)."""
        if self._start_time is None:
            self._start_time = t
        
        talking_time = t - self._start_time
        
        # Ramp amplitude up or down
        if self._ramping_down:
            # Ramp down
            self._amplitude_factor = max(0.0, self._amplitude_factor - 0.05)
        else:
            # Ramp up over ramp_duration
            target = min(1.0, talking_time / self.ramp_duration) if self.ramp_duration > 0 else 1.0
            target = _smooth_step(target)
            # Smooth toward target
            self._amplitude_factor += 0.1 * (target - self._amplitude_factor)

        amplitude_factor = self._amplitude_factor
        
        pitch_amplitude = amplitude_factor * np.deg2rad(self.config.talking_pitch_amplitude_deg)
        yaw_amplitude = amplitude_factor * np.deg2rad(self.config.talking_yaw_amplitude_deg)
        
        pitch_offset = pitch_amplitude * np.sin(
            2 * np.pi * self.config.talking_pitch_frequency * talking_time
        )
        yaw_offset = yaw_amplitude * np.sin(
            2 * np.pi * self.config.talking_yaw_frequency * talking_time
        )
        z_offset = amplitude_factor * self.config.talking_z_amplitude * np.sin(
            2 * np.pi * self.config.talking_pitch_frequency * 0.5 * talking_time
        )

        head_offset = create_head_pose(
            x=0, y=0, z=z_offset,
            roll=0, pitch=pitch_offset, yaw=yaw_offset,
            degrees=False, mm=False
        )

        antenna_amplitude = amplitude_factor * np.deg2rad(self.config.talking_antenna_amplitude_deg)
        antenna_offset = antenna_amplitude * np.sin(
            2 * np.pi * self.config.talking_antenna_frequency * talking_time
        )
        antenna_offsets = np.array([
            antenna_offset,
            -antenna_offset * 0.7 + 0.3 * antenna_amplitude * np.sin(
                2 * np.pi * self.config.talking_antenna_frequency * 1.3 * talking_time
            )
        ], dtype=np.float64)

        return (head_offset, antenna_offsets, 0.0)


class TransitionToNeutralMove:
    """Smooth transition from current position to neutral pose."""

    def __init__(
        self,
        start_pose,
        start_antennas,
        duration: float = 0.4,
    ):
        self.start_pose = start_pose
        self.start_antennas = np.array(start_antennas)
        self._duration = duration
        self.neutral_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
        self.neutral_antennas = np.array([0.0, 0.0])

    @property
    def duration(self) -> float:
        return self._duration

    def evaluate(self, t: float):
        if t >= self._duration:
            return (self.neutral_head_pose, self.neutral_antennas, 0.0)
        
        # Use smoothstep for smooth acceleration/deceleration
        interpolation_t = _smooth_step(t / self._duration)
        head_pose = linear_pose_interpolation(
            self.start_pose, self.neutral_head_pose, interpolation_t,
        )
        antennas_interp = (
            1 - interpolation_t
        ) * self.start_antennas + interpolation_t * self.neutral_antennas
        antennas = antennas_interp.astype(np.float64)
        
        return (head_pose, antennas, 0.0)


# =============================================================================
# Motion Controller
# =============================================================================

class MotionController:
    """
    Unified controller for Reachy Mini movements.
    
    Combines face tracking, breathing animation, talking animation,
    and optional emotion playback in a simple, toggleable interface.
    
    Features (all toggleable via properties):
    - Face tracking: Robot follows detected faces
    - Breathing: Subtle head movement and antenna sway when idle
    - Talking: Animation while TTS is active
    - Emotions: Queue and play emotion animations (optional)
    """
    
    def __init__(
        self,
        robot,
        config: Optional[RobotConfig] = None,
        moves: Optional[Dict] = None,
        emotion_names: Optional[List[str]] = None,
        sentiment_mapping: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Initialize the controller.
        
        Args:
            robot: ReachyMini robot instance
            config: Optional RobotConfig for customization
            moves: Optional dict of emotion moves (from load_emotions)
            emotion_names: Optional list of available emotion names
            sentiment_mapping: Optional custom sentiment to emotion mapping
        """
        self.robot = robot
        self.config = config or RobotConfig()
        self.moves = moves or {}
        self.emotion_names = emotion_names or []
        self.sentiment_mapping = sentiment_mapping or DEFAULT_SENTIMENT_TO_EMOTION
        
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._command_queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        
        # Toggleable features
        self._face_tracking_enabled = True
        self._breathing_enabled = True
        self._body_rotation_enabled = True
        
        # Face tracker
        self._face_tracker: Optional[FaceTrackingWorker] = None
        
        # Animation state
        self._current_move: Optional[Tuple[str, object]] = None
        self._move_start_time: Optional[float] = None
        self._move_queue: deque = deque()
        self._breathing_active = False
        self._talking_active = False
        self._talking_requested = False
        self._talking_move: Optional[TalkingMove] = None  # Separate from main move, composes on top
        self._last_activity_time: float = 0.0
        self._idle_inactivity_delay = 0.5
        self._start_time: float = 0.0
        
        # Last known state
        self._last_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
        self._last_antennas = (0.0, 0.0)
        self._last_body_yaw = 0.0
        # Pre-face-tracking pose (for smooth transitions)
        self._last_base_head_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
        self._last_base_antennas = np.array([0.0, 0.0])
        # Smoothed output pose (prevents jerky movement when animations stop)
        self._smoothed_output_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
        self._smoothed_output_antennas = np.array([0.0, 0.0])
        self._output_smoothing = 0.15  # Lower = smoother transitions when no animation
        
        # Output smoothing to prevent jitter
        self._smoothed_face_offsets = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self._face_offset_smoothing = 0.3  # How fast to track face offsets (higher = more responsive)
        
        # Store body yaw when talking starts to keep it fixed during talking
        self._talking_body_yaw = 0.0
        
        # Store body yaw when emotion starts to freeze body position
        self._emotion_frozen_body_yaw = 0.0
        
        self._target_period = 1.0 / self.config.control_loop_frequency_hz
    
    # =========================================================================
    # Toggle Properties
    # =========================================================================
    
    @property
    def face_tracking_enabled(self) -> bool:
        """Whether face tracking is enabled."""
        with self._lock:
            return self._face_tracking_enabled
    
    @face_tracking_enabled.setter
    def face_tracking_enabled(self, value: bool):
        """Enable or disable face tracking."""
        with self._lock:
            if self._face_tracking_enabled != value:
                self._face_tracking_enabled = value
                if self._face_tracker is not None:
                    if value:
                        self._face_tracker.start()
                    else:
                        self._face_tracker.stop()
                print(f"[Controller] Face tracking = {value}")
    
    @property
    def breathing_enabled(self) -> bool:
        """Whether breathing animation is enabled."""
        with self._lock:
            return self._breathing_enabled
    
    @breathing_enabled.setter
    def breathing_enabled(self, value: bool):
        """Enable or disable breathing animation."""
        with self._lock:
            self._breathing_enabled = value
            print(f"[Controller] Breathing = {value}")
    
    @property
    def head_breathing_enabled(self) -> bool:
        """Whether head breathing (z movement) is enabled."""
        with self._lock:
            return self.config.head_breathing_enabled
    
    @head_breathing_enabled.setter
    def head_breathing_enabled(self, value: bool):
        """Enable or disable head breathing (z movement)."""
        with self._lock:
            self.config = self.config.update(head_breathing_enabled=value)
            print(f"[Controller] Head breathing = {value}")
    
    @property
    def antenna_breathing_enabled(self) -> bool:
        """Whether antenna breathing (sway) is enabled."""
        with self._lock:
            return self.config.antenna_breathing_enabled
    
    @antenna_breathing_enabled.setter
    def antenna_breathing_enabled(self, value: bool):
        """Enable or disable antenna breathing (sway)."""
        with self._lock:
            self.config = self.config.update(antenna_breathing_enabled=value)
            print(f"[Controller] Antenna breathing = {value}")
    
    @property
    def antenna_excited_mode(self) -> bool:
        """Whether antenna excited mode (faster oscillation) is enabled."""
        with self._lock:
            return self.config.antenna_excited_mode
    
    @antenna_excited_mode.setter
    def antenna_excited_mode(self, value: bool):
        """Enable or disable antenna excited mode (faster oscillation)."""
        with self._lock:
            self.config = self.config.update(antenna_excited_mode=value)
            print(f"[Controller] Antenna excited mode = {value}")
    
    @property
    def body_rotation_enabled(self) -> bool:
        """Whether body rotation for extended tracking is enabled."""
        with self._lock:
            return self._body_rotation_enabled
    
    @body_rotation_enabled.setter
    def body_rotation_enabled(self, value: bool):
        """Enable or disable body rotation for face tracking."""
        with self._lock:
            self._body_rotation_enabled = value
            # Also update the config for the face tracker
            self.config = self.config.update(body_rotation_enabled=value)
            print(f"[Controller] Body rotation = {value}")
    
    def get_state(self) -> dict:
        """Get current state of all toggles and status."""
        with self._lock:
            return {
                "face_tracking_enabled": self._face_tracking_enabled,
                "breathing_enabled": self._breathing_enabled,
                "head_breathing_enabled": self.config.head_breathing_enabled,
                "antenna_breathing_enabled": self.config.antenna_breathing_enabled,
                "antenna_excited_mode": self.config.antenna_excited_mode,
                "body_rotation_enabled": self._body_rotation_enabled,
                "talking_active": self._talking_active,
                "breathing_active": self._breathing_active,
                "has_emotions": len(self.emotion_names) > 0,
            }
    
    def get_annotated_frame(self):
        """
        Get the current camera frame with face tracking annotations.
        
        Returns:
            Tuple of (frame, face_info) from the face tracker, or (None, {}) if unavailable.
        """
        if self._face_tracker is not None:
            return self._face_tracker.get_annotated_frame()
        return None, {"detected": False, "bbox": None, "center": None, "offsets": {"pitch": 0, "yaw": 0}}
    
    def get_raw_frame(self):
        """
        Get the current raw camera frame without annotations.
        
        Returns:
            The raw frame or None if unavailable.
        """
        if self._face_tracker is not None:
            return self._face_tracker.get_raw_frame()
        return None
    
    # =========================================================================
    # Emotion and Talking Control
    # =========================================================================
    
    def queue_emotion(self, emotion_name: str, priority: bool = False):
        """
        Queue an emotion to play.
        
        Args:
            emotion_name: Name of the emotion to play
            priority: If True, clear queue and play immediately
        """
        self._command_queue.put(("queue_emotion", (emotion_name, priority)))
    
    def queue_emotion_for_sentiment(self, sentiment: str, priority: bool = False):
        """
        Queue a random emotion for the given sentiment.
        
        Args:
            sentiment: Sentiment name (e.g., "happy", "sad", "curious")
            priority: If True, clear queue and play immediately
        """
        emotion_name = self._get_emotion_for_sentiment(sentiment)
        if emotion_name:
            self.queue_emotion(emotion_name, priority)
    
    def start_talking(self):
        """Start talking animation (call when TTS starts)."""
        self._command_queue.put(("start_talking", None))
    
    def stop_talking(self):
        """Stop talking animation (call when TTS ends)."""
        self._command_queue.put(("stop_talking", None))
    
    def _get_emotion_for_sentiment(self, sentiment: str) -> Optional[str]:
        """Get a random emotion name for the given sentiment."""
        possible_actions = self.sentiment_mapping.get(sentiment.lower(), ["attentive1"])
        available_actions = [a for a in possible_actions if a in self.emotion_names]
        
        if not available_actions:
            available_actions = ["attentive1"] if "attentive1" in self.emotion_names else []
        
        if not available_actions:
            return None
        
        return random.choice(available_actions)
    
    # =========================================================================
    # Lifecycle
    # =========================================================================
    
    def start(self):
        """Start the controller."""
        if self._thread is not None and self._thread.is_alive():
            print("[Controller] Already running")
            return
        
        print("[Controller] Starting...")
        
        # Initialize face tracker
        self._face_tracker = FaceTrackingWorker(self.robot, self.config)
        if self._face_tracking_enabled:
            self._face_tracker.start()
        
        # Start control loop
        self._stop_event.clear()
        self._start_time = time.monotonic()
        self._last_activity_time = self._start_time
        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._thread.start()
        
        print("[Controller] Started")
    
    def stop(self):
        """Stop the controller and return robot to neutral position."""
        if self._thread is None or not self._thread.is_alive():
            return
        
        print("[Controller] Stopping...")
        
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._thread = None
        
        # Stop face tracker
        if self._face_tracker is not None:
            self._face_tracker.stop()
            self._face_tracker = None
        
        # Return to neutral position
        try:
            neutral_head = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            self.robot.goto_target(
                head=neutral_head,
                antennas=[0.0, 0.0],
                duration=1.0,
            )
        except Exception as e:
            print(f"[Controller] Failed to reset to neutral: {e}")
        
        print("[Controller] Stopped")
    
    # =========================================================================
    # Internal Control Loop
    # =========================================================================
    
    def _mark_activity(self):
        """Mark that activity occurred (resets idle timer)."""
        self._last_activity_time = time.monotonic()
    
    def _process_commands(self):
        """Process any pending commands from the queue."""
        while True:
            try:
                cmd, payload = self._command_queue.get_nowait()
            except queue.Empty:
                break
            
            if cmd == "queue_emotion":
                emotion_name, priority = payload
                if emotion_name not in self.emotion_names:
                    print(f"[Controller] Emotion '{emotion_name}' not found")
                    continue
                
                move = self.moves.get(emotion_name)
                if move is None:
                    continue
                
                if priority:
                    self._move_queue.clear()
                    self._current_move = None
                    self._move_start_time = None
                    self._breathing_active = False
                    self._talking_active = False
                
                self._move_queue.append((emotion_name, move))
                self._mark_activity()
                print(f"[Controller] Queued emotion: {emotion_name}")
            
            elif cmd == "start_talking":
                self._talking_requested = True
                # Store current body yaw to keep it fixed during talking
                if self._face_tracker is not None:
                    self._talking_body_yaw = self._face_tracker.get_body_yaw()
                else:
                    self._talking_body_yaw = 0.0
                self._mark_activity()
                print("[Controller] Talking requested")
            
            elif cmd == "stop_talking":
                self._talking_requested = False
                if self._talking_active or self._talking_move is not None:
                    print("[Controller] Stopping talking")
                    self._talking_move = None  # Clear the talking move
                    self._talking_active = False
                    self._mark_activity()
    
    def _get_current_robot_state(self) -> Tuple:
        """Get current robot pose and antennas."""
        try:
            _, current_antennas = self.robot.get_current_joint_positions()
            current_head_pose = self.robot.get_current_head_pose()
            return current_head_pose, current_antennas
        except Exception:
            return create_head_pose(0, 0, 0, 0, 0, 0, degrees=True), [0.0, 0.0]
    
    def _manage_idle_animations(self, current_time: float):
        """Manage breathing and talking animations when idle."""
        with self._lock:
            breathing_enabled = self._breathing_enabled
        
        # Stop breathing animation if breathing is disabled
        if self._breathing_active and not breathing_enabled:
            print("[Controller] Breathing disabled, stopping breathing animation")
            self._current_move = None
            self._move_start_time = None
            self._breathing_active = False
            self._mark_activity()
        
        # Handle talking request - talking composes ON TOP of other animations/face tracking
        if self._talking_requested and not self._talking_active:
            # Store current body yaw to keep it fixed during talking
            if self._face_tracker is not None:
                self._talking_body_yaw = self._face_tracker.get_body_yaw()
            else:
                self._talking_body_yaw = 0.0
            # Start talking animation (produces offsets, not absolute poses)
            self._talking_move = TalkingMove(
                config=self.config,
                ramp_duration=0.3,
            )
            self._talking_active = True
            print("[Controller] Starting talking animation")
        
        # Handle talking stop request - ramp down instead of abrupt stop
        if not self._talking_requested and self._talking_active:
            if self._talking_move is not None:
                self._talking_move.start_ramp_down()
        
        # Interrupt breathing for queued emotions (but not talking - it composes on top)
        if self._breathing_active and self._move_queue:
            print("[Controller] Interrupting breathing for emotion")
            self._current_move = None
            self._move_start_time = None
            self._breathing_active = False
        
        # Check if we should start breathing animation (only when idle and not talking)
        if (
            self._current_move is None
            and not self._move_queue
            and not self._breathing_active
            and not self._talking_active
        ):
            idle_for = current_time - self._last_activity_time
            if idle_for >= self._idle_inactivity_delay:
                current_head_pose, current_antennas = self._get_current_robot_state()
                
                if breathing_enabled:
                    # Start breathing animation
                    breathing_move = BreathingMove(
                        interpolation_start_pose=current_head_pose,
                        interpolation_start_antennas=current_antennas,
                        config=self.config,
                        interpolation_duration=1.0,
                        config_provider=lambda: self.config,
                    )
                    self._current_move = ("breathing", breathing_move)
                    self._move_start_time = current_time
                    self._breathing_active = True
    
    def _manage_move_queue(self, current_time: float):
        """Manage emotion move queue."""
        # Check if current move finished (non-infinite duration moves)
        if self._current_move is not None and self._move_start_time is not None:
            name, move = self._current_move
            if name not in ("breathing", "talking", "transition"):
                elapsed = current_time - self._move_start_time
                if hasattr(move, 'duration') and elapsed >= move.duration:
                    # Emotion finished - queue a smooth transition to neutral
                    # Use the last BASE pose (before face tracking) for smooth transition
                    # This ensures continuity since face tracking will still be applied on top
                    try:
                        start_pose = self._last_base_head_pose
                        start_antennas = self._last_base_antennas
                        # Create transition move from last base pose to neutral
                        transition = TransitionToNeutralMove(
                            start_pose=start_pose,
                            start_antennas=start_antennas,
                            duration=0.8,  # Smooth 0.8s transition
                        )
                        self._current_move = ("transition", transition)
                        self._move_start_time = current_time
                        print(f"[Controller] Emotion '{name}' finished, transitioning to neutral")
                    except Exception as e:
                        print(f"[Controller] Error creating transition: {e}")
                        self._current_move = None
                        self._move_start_time = None
            elif name == "transition":
                # Check if transition finished
                elapsed = current_time - self._move_start_time
                if hasattr(move, 'duration') and elapsed >= move.duration:
                    self._current_move = None
                    self._move_start_time = None
                    # Mark activity when transition completes for proper idle timing
                    self._mark_activity()
                    # Resume face tracking now that emotion+transition is complete
                    if self._face_tracker is not None:
                        self._face_tracker.resume()
        
        # Start next queued move
        if self._current_move is None and self._move_queue:
            emotion_name, move = self._move_queue.popleft()
            self._current_move = (emotion_name, move)
            self._move_start_time = current_time
            self._breathing_active = False
            self._talking_active = False
            # Store current body yaw to freeze position during emotion
            self._emotion_frozen_body_yaw = self._last_body_yaw
            # Pause face tracking during emotion playback
            if self._face_tracker is not None:
                self._face_tracker.pause()
            print(f"[Controller] Playing emotion: {emotion_name}")
    
    def _get_face_tracking_offsets(self) -> Tuple[float, float, float, float, float, float]:
        """Get face tracking offsets if enabled, with additional smoothing."""
        with self._lock:
            face_tracking = self._face_tracking_enabled
            body_rotation = self._body_rotation_enabled
            talking_active = self._talking_active
        
        # Get raw offsets (zeros if tracking disabled)
        if face_tracking and self._face_tracker is not None:
            raw_offsets = self._face_tracker.get_face_tracking_offsets()
        else:
            raw_offsets = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        smoothing = self._face_offset_smoothing
        
        # Apply smoothing - but when body rotation is enabled (and not talking), 
        # use raw yaw directly so head follows body instantly
        for i in range(6):
            if body_rotation and not talking_active and i == 5:  # index 5 is yaw
                # NO smoothing for head yaw when body tracking is enabled and not talking
                # Head must follow body rotation immediately
                self._smoothed_face_offsets[i] = raw_offsets[i]
            else:
                # Apply smoothing
                self._smoothed_face_offsets[i] += smoothing * (raw_offsets[i] - self._smoothed_face_offsets[i])
            # Snap to zero if very small to avoid micro-movements
            if abs(self._smoothed_face_offsets[i]) < 0.0005:
                self._smoothed_face_offsets[i] = 0.0
        
        return (
            self._smoothed_face_offsets[0],
            self._smoothed_face_offsets[1],
            self._smoothed_face_offsets[2],
            self._smoothed_face_offsets[3],
            self._smoothed_face_offsets[4],
            self._smoothed_face_offsets[5],
        )
    
    def _apply_face_tracking(self, head_pose):
        """Apply face tracking offsets to a head pose."""
        offsets = self._get_face_tracking_offsets()
        
        if all(o == 0.0 for o in offsets):
            return head_pose
        
        face_offset = create_head_pose(
            x=offsets[0], y=offsets[1], z=offsets[2],
            roll=offsets[3], pitch=offsets[4], yaw=offsets[5],
            degrees=False, mm=False,
        )
        
        return compose_world_offset(head_pose, face_offset, reorthonormalize=True)
    
    def _evaluate_and_send(self, current_time: float):
        """Evaluate current move and send to robot."""
        head_pose = None
        antennas = None
        is_emotion_or_transition = False
        
        if self._current_move is not None and self._move_start_time is not None:
            name, move = self._current_move
            elapsed = current_time - self._move_start_time
            
            # Check if this is an emotion or transition (not breathing/talking)
            is_emotion_or_transition = name not in ("breathing", "talking")
            
            try:
                result = move.evaluate(elapsed)
                if result is not None:
                    if len(result) == 3:
                        head_pose, antennas, _ = result
                    else:
                        head_pose, antennas = result[:2]
            except Exception as e:
                print(f"[Controller] Move evaluation error: {e}")
        
        # Default to neutral if no move - apply smoothing to prevent jerks
        if head_pose is None:
            target_pose = create_head_pose(0, 0, 0, 0, 0, 0, degrees=True)
            target_antennas = np.array([0.0, 0.0])
            # Smoothly interpolate toward neutral
            # Position smoothing (translation part of 4x4 matrix)
            self._smoothed_output_pose[:3, 3] += self._output_smoothing * (
                target_pose[:3, 3] - self._smoothed_output_pose[:3, 3]
            )
            # Rotation smoothing (blend rotation matrices via weighted average)
            # For small angles this is approximately correct
            self._smoothed_output_pose[:3, :3] += self._output_smoothing * (
                target_pose[:3, :3] - self._smoothed_output_pose[:3, :3]
            )
            # Antenna smoothing
            self._smoothed_output_antennas += self._output_smoothing * (
                target_antennas - self._smoothed_output_antennas
            )
            head_pose = self._smoothed_output_pose.copy()
            antennas = self._smoothed_output_antennas.copy()
        else:
            # When animation is active, track the output for smooth transitions later
            self._smoothed_output_pose = head_pose.copy()
            self._smoothed_output_antennas = np.array(antennas, dtype=np.float64)
        
        if antennas is None:
            antennas = np.array([0.0, 0.0])
        
        # During emotion/transition: PAUSE all other animations
        # - No face tracking
        # - Body frozen at position when emotion started
        # - Just play the emotion animation as-is
        if is_emotion_or_transition:
            # Reset face tracking offsets so there's no jump when resuming
            for i in range(6):
                self._smoothed_face_offsets[i] = 0.0
            
            # Use frozen body yaw (keep body at position when emotion started)
            body_yaw = self._emotion_frozen_body_yaw
            
            # Send emotion pose directly to robot (no additional modifications)
            antennas_tuple = (float(antennas[0]), float(antennas[1]))
            
            try:
                self.robot.set_target(
                    head=head_pose,
                    antennas=antennas_tuple,
                    body_yaw=body_yaw,
                )
                self._last_head_pose = head_pose
                self._last_antennas = antennas_tuple
                self._last_body_yaw = body_yaw
            except Exception as e:
                print(f"[Controller] Robot error: {e}")
            return
        
        # Normal mode (breathing/idle): Apply face tracking and body rotation
        # Store pre-face-tracking pose for transitions
        self._last_base_head_pose = head_pose
        self._last_base_antennas = tuple(float(a) for a in antennas)
        
        # Apply face tracking
        head_pose = self._apply_face_tracking(head_pose)
        
        # Apply talking animation offsets on top of face-tracked pose
        if self._talking_move is not None:
            try:
                talking_result = self._talking_move.evaluate(current_time)
                if talking_result is not None:
                    talking_offset, antenna_offsets, _ = talking_result
                    # Compose talking offset on top of current pose
                    head_pose = compose_world_offset(head_pose, talking_offset, reorthonormalize=True)
                    # Add antenna offsets
                    antennas = np.array(antennas) + antenna_offsets
                # Check if ramp down finished after evaluate (amplitude decreases during evaluate)
                if self._talking_move.is_finished():
                    print("[Controller] Talking animation finished")
                    self._talking_move = None
                    self._talking_active = False
            except Exception as e:
                print(f"[Controller] Talking animation error: {e}")
        
        # Get body yaw from face tracker if body rotation is enabled
        body_yaw = 0.0
        with self._lock:
            body_rotation_enabled = self._body_rotation_enabled
        if body_rotation_enabled and self._face_tracker is not None:
            body_yaw = self._face_tracker.get_body_yaw()
        
        # Convert antennas to tuple
        antennas_tuple = (float(antennas[0]), float(antennas[1]))
        
        # Send to robot
        try:
            self.robot.set_target(
                head=head_pose,
                antennas=antennas_tuple,
                body_yaw=body_yaw,
            )
            self._last_head_pose = head_pose
            self._last_antennas = antennas_tuple
            self._last_body_yaw = body_yaw
        except Exception as e:
            print(f"[Controller] Robot error: {e}")
    
    def _control_loop(self):
        """Main control loop."""
        print("[Controller] Control loop started")
        
        while not self._stop_event.is_set():
            loop_start = time.monotonic()
            
            self._process_commands()
            self._manage_idle_animations(loop_start)
            self._manage_move_queue(loop_start)
            self._evaluate_and_send(loop_start)
            
            # Maintain loop frequency
            elapsed = time.monotonic() - loop_start
            sleep_time = max(0.0, self._target_period - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        print("[Controller] Control loop stopped")


# Backwards compatibility alias
SimpleController = MotionController
