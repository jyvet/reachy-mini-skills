"""Face tracking worker for Reachy Mini.

Usage:
    from reachy_mini_skills.actuation.motion import face_tracking
    
    tracker = face_tracking.FaceTrackingWorker(robot, config)
    tracker.start()
    offsets = tracker.get_face_tracking_offsets()
    tracker.stop()
    
    # Get annotated frame for visualization
    frame, face_info = tracker.get_annotated_frame()
"""

import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

from ...config import RobotConfig

__all__ = ["FaceTrackingWorker"]


class FaceTrackingWorker:
    """
    Background thread that captures frames from robot camera,
    detects faces, and produces head tracking offsets.
    
    Uses MediaPipe for robust face detection if available,
    falls back to Haar cascade otherwise.
    """
    
    def __init__(self, robot, config: RobotConfig = None):
        self.robot = robot
        self.config = config or RobotConfig()
        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        
        # Pause flag - when True, tracking continues but doesn't update values
        self._paused = False
        
        self._offsets: Tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        self._smoothed_yaw = 0.0
        self._smoothed_pitch = 0.0
        
        # Body rotation tracking
        self._body_yaw = 0.0
        self._target_body_yaw = 0.0
        
        # PID controller state for yaw (body rotation)
        self._pid_integral_yaw = 0.0
        self._pid_last_error_yaw = 0.0
        self._pid_last_time = None
        
        # PID controller state for pitch (head)
        self._pid_integral_pitch = 0.0
        self._pid_last_error_pitch = 0.0
        
        # Flag to initialize PID on first face detection (prevents derivative spike)
        self._pid_needs_init = True
        # Ramp-up counter for smooth start (0.0 to 1.0 over ~0.5 seconds)
        self._tracking_ramp = 0.0
        self._tracking_ramp_speed = 0.05  # How fast to ramp up (per frame)
        
        # Store last frame and detection info for visualization
        self._last_frame: Optional[np.ndarray] = None
        self._last_face_bbox: Optional[Tuple[int, int, int, int]] = None
        self._last_face_center: Optional[Tuple[int, int]] = None
        self._tracking_active = False
        
        # Face persistence - track the same face for a minimum time before switching
        self._tracked_face_center: Optional[Tuple[float, float]] = None  # Center of currently tracked face
        self._tracked_face_area: float = 0.0  # Area of currently tracked face
        self._tracked_face_start_time: float = 0.0  # When we started tracking this face
        self._min_face_tracking_duration: float = 10.0  # Minimum seconds to track same face
        self._face_match_threshold: float = 0.3  # Max relative distance to consider same face
        
        # Initialize face detector (MediaPipe preferred, Haar cascade fallback)
        self._use_mediapipe = MEDIAPIPE_AVAILABLE
        if self._use_mediapipe:
            # Use new MediaPipe Tasks API (0.10.x+)
            base_options = mp_tasks.BaseOptions(
                model_asset_path=self._get_mediapipe_model_path()
            )
            options = mp_vision.FaceDetectorOptions(
                base_options=base_options,
                min_detection_confidence=0.5,
            )
            self._face_detector = mp_vision.FaceDetector.create_from_options(options)
            print("[FaceTracking] Using MediaPipe for face detection")
        else:
            self._face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            print("[FaceTracking] Using Haar cascade for face detection (MediaPipe not available)")
        
        self._target_period = 1.0 / self.config.face_tracking_fps
        self._last_face_time = 0.0
        self._face_timeout = getattr(self.config, 'face_tracking_hold_time', 10.0)
    
    def _get_mediapipe_model_path(self) -> str:
        """Get the path to the MediaPipe face detection model.
        
        Downloads the model if not already present.
        """
        import os
        import urllib.request
        
        # Store model in user's cache directory
        cache_dir = os.path.expanduser("~/.cache/mediapipe/models")
        os.makedirs(cache_dir, exist_ok=True)
        
        model_path = os.path.join(cache_dir, "blaze_face_short_range.tflite")
        
        if not os.path.exists(model_path):
            print("[FaceTracking] Downloading MediaPipe face detection model...")
            model_url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
            urllib.request.urlretrieve(model_url, model_path)
            print(f"[FaceTracking] Model downloaded to {model_path}")
        
        return model_path
    
    def start(self):
        """Start the face tracking thread."""
        if self._thread is not None and self._thread.is_alive():
            print("[FaceTracking] Already running")
            return
        
        # Reset PID init flag so first detection initializes properly
        self._pid_needs_init = True
        self._tracking_ramp = 0.0
        
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self._thread.start()
        print("[FaceTracking] Started")
    
    def stop(self):
        """Stop the face tracking thread."""
        if self._thread is None or not self._thread.is_alive():
            return
        
        print("[FaceTracking] Stopping...")
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._thread = None
        print("[FaceTracking] Stopped")
    
    def get_face_tracking_offsets(self) -> Tuple[float, float, float, float, float, float]:
        """Get current face tracking offsets (thread-safe)."""
        with self._lock:
            return self._offsets
    
    def get_body_yaw(self) -> float:
        """Get current body yaw for rotation tracking (thread-safe)."""
        with self._lock:
            return self._body_yaw
    
    def reset_tracking_state(self):
        """
        Reset tracking state to prevent jumps after talking ends.
        
        This resets the smoothed values and PID state so face tracking
        starts fresh without accumulated drift causing sudden movements.
        """
        with self._lock:
            # Reset smoothed values
            self._smoothed_yaw = 0.0
            self._smoothed_pitch = 0.0
            
            # Reset offsets
            self._offsets = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            
            # Reset ramp so tracking starts smoothly again
            self._tracking_ramp = 0.0
            
            # Reset tracked face persistence
            self._tracked_face_center = None
            self._tracked_face_area = 0.0
            
        # Set flag to reinitialize PID on next detection (prevents derivative spike)
        self._pid_needs_init = True
            
        print("[FaceTracking] Tracking state reset")
    
    def pause(self):
        """
        Pause face tracking updates.
        
        The tracking thread continues running but won't update offsets or body yaw.
        This prevents accumulation during emotion playback.
        """
        with self._lock:
            self._paused = True
        print("[FaceTracking] Paused")
    
    def resume(self):
        """
        Resume face tracking updates.
        
        Also resets tracking state to prevent jumps from accumulated values.
        """
        with self._lock:
            self._paused = False
            # Reset body yaw to prevent jump
            self._body_yaw = 0.0
            self._target_body_yaw = 0.0
            # Reset head tracking offsets
            self._smoothed_yaw = 0.0
            self._smoothed_pitch = 0.0
            self._offsets = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            self._tracking_ramp = 0.0
            # Reset tracked face persistence
            self._tracked_face_center = None
            self._tracked_face_area = 0.0
        self._pid_needs_init = True
        print("[FaceTracking] Resumed")
    
    @property
    def is_paused(self) -> bool:
        """Check if face tracking is paused."""
        with self._lock:
            return self._paused
    
    def get_annotated_frame(self) -> Tuple[Optional[np.ndarray], dict]:
        """
        Get the last frame with face detection annotations drawn.
        
        Returns:
            Tuple of (annotated_frame, face_info) where:
            - annotated_frame: Frame with detection overlay, or None if no frame available
            - face_info: Dict with 'detected', 'bbox', 'center', 'offsets' keys
        """
        with self._lock:
            frame = self._last_frame.copy() if self._last_frame is not None else None
            bbox = self._last_face_bbox
            center = self._last_face_center
            tracking = self._tracking_active
            offsets = self._offsets
        
        face_info = {
            "detected": tracking,
            "bbox": bbox,
            "center": center,
            "offsets": {
                "pitch": offsets[4],
                "yaw": offsets[5],
            }
        }
        
        if frame is not None:
            h, w = frame.shape[:2]
            
            # Draw frame center crosshair
            cv2.line(frame, (w // 2 - 20, h // 2), (w // 2 + 20, h // 2), (100, 100, 100), 1)
            cv2.line(frame, (w // 2, h // 2 - 20), (w // 2, h // 2 + 20), (100, 100, 100), 1)
            
            if tracking and bbox is not None:
                x, y, bw, bh = bbox
                # Draw face bounding box (green when tracking)
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                
                # Draw face center
                if center is not None:
                    cv2.circle(frame, center, 5, (0, 255, 0), -1)
                    # Draw line from center to face
                    cv2.line(frame, (w // 2, h // 2), center, (0, 255, 0), 1)
                
                # Display tracking info
                cv2.putText(frame, f"Yaw: {np.rad2deg(offsets[5]):.1f}°", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Pitch: {np.rad2deg(offsets[4]):.1f}°", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, "TRACKING", (10, h - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                # No face detected
                cv2.putText(frame, "No face detected", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return frame, face_info
    
    def get_raw_frame(self) -> Optional[np.ndarray]:
        """Get the last raw frame without annotations (thread-safe)."""
        with self._lock:
            return self._last_frame.copy() if self._last_frame is not None else None

    def _select_face_with_persistence(self, faces: list, frame_w: int, frame_h: int) -> Optional[tuple]:
        """
        Select which face to track with persistence logic.
        
        - If no face is currently being tracked, pick the largest (closest) face
        - If a face is being tracked and min duration hasn't elapsed, try to find the same face
        - If min duration has elapsed, switch to largest face if it's significantly larger
        
        Args:
            faces: List of tuples (x, y, w, h, center_x, center_y, area)
            frame_w: Frame width for normalization
            frame_h: Frame height for normalization
            
        Returns:
            Selected face tuple or None
        """
        if not faces:
            return None
        
        current_time = time.monotonic()
        
        # Find the largest face (closest to robot)
        largest_face = max(faces, key=lambda f: f[6])  # f[6] is area
        
        # If no face is being tracked, start tracking the largest one
        if self._tracked_face_center is None:
            self._tracked_face_center = (largest_face[4], largest_face[5])  # center_x, center_y
            self._tracked_face_area = largest_face[6]
            self._tracked_face_start_time = current_time
            print(f"[FaceTracking] Started tracking new face (area: {largest_face[6]:.0f})")
            return largest_face
        
        # Calculate time elapsed since we started tracking the current face
        tracking_duration = current_time - self._tracked_face_start_time
        
        # Try to find the currently tracked face based on position similarity
        best_match = None
        best_match_distance = float('inf')
        
        for face in faces:
            center_x, center_y = face[4], face[5]
            # Normalize distance by frame dimensions
            dist_x = (center_x - self._tracked_face_center[0]) / frame_w
            dist_y = (center_y - self._tracked_face_center[1]) / frame_h
            distance = (dist_x ** 2 + dist_y ** 2) ** 0.5
            
            if distance < best_match_distance:
                best_match_distance = distance
                best_match = face
        
        # Check if best match is close enough to be considered the same face
        is_same_face = best_match_distance < self._face_match_threshold
        
        if tracking_duration < self._min_face_tracking_duration:
            # Still in minimum tracking period - stick with current face if found
            if is_same_face and best_match is not None:
                # Update tracked face position
                self._tracked_face_center = (best_match[4], best_match[5])
                self._tracked_face_area = best_match[6]
                return best_match
            elif best_match is not None:
                # Current face not found but we're in lock period
                # Use best match anyway to avoid jarring switches
                self._tracked_face_center = (best_match[4], best_match[5])
                self._tracked_face_area = best_match[6]
                return best_match
        else:
            # Minimum tracking time elapsed - can switch to a better face
            # Switch to largest face if it's significantly larger (50% bigger)
            if largest_face[6] > self._tracked_face_area * 1.5:
                if largest_face != best_match or not is_same_face:
                    print(f"[FaceTracking] Switching to larger face (area: {largest_face[6]:.0f} vs {self._tracked_face_area:.0f})")
                self._tracked_face_center = (largest_face[4], largest_face[5])
                self._tracked_face_area = largest_face[6]
                self._tracked_face_start_time = current_time  # Reset tracking timer
                return largest_face
            elif is_same_face and best_match is not None:
                # Keep tracking the same face
                self._tracked_face_center = (best_match[4], best_match[5])
                self._tracked_face_area = best_match[6]
                return best_match
            else:
                # Current face lost, switch to largest
                print(f"[FaceTracking] Lost tracked face, switching to largest (area: {largest_face[6]:.0f})")
                self._tracked_face_center = (largest_face[4], largest_face[5])
                self._tracked_face_area = largest_face[6]
                self._tracked_face_start_time = current_time
                return largest_face
        
        return largest_face

    def _tracking_loop(self):
        """Main face tracking loop."""
        print("[FaceTracking] Loop started")
        
        while not self._stop_event.is_set():
            loop_start = time.monotonic()
            
            # Check if paused - skip processing but keep loop running
            with self._lock:
                paused = self._paused
            
            if paused:
                # When paused, just sleep and continue
                time.sleep(self._target_period)
                continue
            
            try:
                frame = self.robot.media.get_frame()
                
                if frame is not None:
                    # Store frame for visualization
                    with self._lock:
                        self._last_frame = frame.copy()
                    
                    self._process_frame(frame)
                else:
                    self._decay_offsets()
                    
            except Exception as e:
                print(f"[FaceTracking] Error: {e}")
                self._decay_offsets()
            
            elapsed = time.monotonic() - loop_start
            sleep_time = max(0.0, self._target_period - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        print("[FaceTracking] Loop stopped")
    
    def _process_frame(self, frame):
        """Process a frame to detect face and update offsets."""
        frame_h, frame_w = frame.shape[:2]
        face_detected = False
        face_center_x = 0
        face_center_y = 0
        bbox = None
        
        if self._use_mediapipe:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Create MediaPipe Image from numpy array
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self._face_detector.detect(mp_image)
            
            if results.detections:
                # Convert all detections to absolute coordinates
                valid_faces = []
                for detection in results.detections:
                    bbox_data = detection.bounding_box
                    x = bbox_data.origin_x
                    y = bbox_data.origin_y
                    w = bbox_data.width
                    h = bbox_data.height
                    
                    # Clamp values to frame bounds
                    x = max(0, min(x, frame_w - 1))
                    y = max(0, min(y, frame_h - 1))
                    w = min(w, frame_w - x)
                    h = min(h, frame_h - y)
                    
                    if w > 20 and h > 20:  # Minimum face size
                        center_x = x + w / 2
                        center_y = y + h / 2
                        area = w * h
                        valid_faces.append((x, y, w, h, center_x, center_y, area))
                
                if valid_faces:
                    # Select face using persistence logic
                    selected_face = self._select_face_with_persistence(valid_faces, frame_w, frame_h)
                    if selected_face:
                        x, y, w, h, face_center_x, face_center_y, area = selected_face
                        bbox = (x, y, w, h)
                        face_detected = True
        else:
            # Fallback to Haar cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = self._face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if len(faces) > 0:
                # Convert to same format as MediaPipe for consistency
                valid_faces = []
                for (x, y, w, h) in faces:
                    center_x = x + w / 2
                    center_y = y + h / 2
                    area = w * h
                    valid_faces.append((x, y, w, h, center_x, center_y, area))
                
                # Select face using persistence logic
                selected_face = self._select_face_with_persistence(valid_faces, frame_w, frame_h)
                if selected_face:
                    x, y, w, h, face_center_x, face_center_y, area = selected_face
                    bbox = (x, y, w, h)
                    face_detected = True
        
        max_yaw = np.deg2rad(self.config.face_tracking_max_yaw_deg)
        max_pitch = np.deg2rad(self.config.face_tracking_max_pitch_deg)
        
        if face_detected:
            frame_center_x = frame_w / 2
            frame_center_y = frame_h / 2
            
            # Calculate normalized offset (-1 to 1) - this is our error signal
            error_x = (face_center_x - frame_center_x) / (frame_w / 2)
            error_y = (face_center_y - frame_center_y) / (frame_h / 2)
            
            # Apply deadzone to ignore small movements and reduce jitter
            deadzone = getattr(self.config, 'face_tracking_deadzone', 0.05)
            if abs(error_x) < deadzone:
                error_x = 0.0
            if abs(error_y) < deadzone:
                error_y = 0.0
            
            # Initialize PID state on first detection to prevent derivative spike
            if self._pid_needs_init:
                self._pid_last_error_yaw = error_x
                self._pid_last_error_pitch = error_y
                self._pid_integral_yaw = 0.0
                self._pid_integral_pitch = 0.0
                self._pid_last_time = None
                self._tracking_ramp = 0.0  # Start ramp from zero
                self._pid_needs_init = False
                print("[FaceTracking] PID initialized for new face detection")
            
            # Ramp up tracking smoothly to avoid jumps
            self._tracking_ramp = min(1.0, self._tracking_ramp + self._tracking_ramp_speed)
            
            # Get PID gains from config
            kp = getattr(self.config, 'face_tracking_pid_kp', 0.6)
            ki = getattr(self.config, 'face_tracking_pid_ki', 0.02)
            kd = getattr(self.config, 'face_tracking_pid_kd', 0.15)
            integral_limit = getattr(self.config, 'face_tracking_pid_integral_limit', 0.3)
            
            # Calculate dt for PID
            current_time = time.monotonic()
            if self._pid_last_time is None:
                dt = self._target_period
            else:
                dt = current_time - self._pid_last_time
            self._pid_last_time = current_time
            dt = max(dt, 0.001)  # Prevent division by zero
            
            # PID for pitch (head up/down)
            self._pid_integral_pitch += error_y * dt
            self._pid_integral_pitch = np.clip(self._pid_integral_pitch, -integral_limit, integral_limit)
            derivative_pitch = (error_y - self._pid_last_error_pitch) / dt
            self._pid_last_error_pitch = error_y
            
            pid_output_pitch = kp * error_y + ki * self._pid_integral_pitch + kd * derivative_pitch
            target_pitch = np.clip(pid_output_pitch * max_pitch, -max_pitch, max_pitch)
            
            if self.config.body_rotation_enabled:
                max_body_yaw = np.deg2rad(self.config.body_rotation_max_deg)
                
                # PID for yaw (body rotation)
                self._pid_integral_yaw += error_x * dt
                self._pid_integral_yaw = np.clip(self._pid_integral_yaw, -integral_limit, integral_limit)
                derivative_yaw = (error_x - self._pid_last_error_yaw) / dt
                self._pid_last_error_yaw = error_x
                
                pid_output_yaw = kp * error_x + ki * self._pid_integral_yaw + kd * derivative_yaw
                
                # Body rotates opposite to face offset (face on right -> rotate right -> negative yaw)
                target_body_yaw = -np.clip(pid_output_yaw * max_body_yaw, -max_body_yaw, max_body_yaw)
                self._target_body_yaw = target_body_yaw
                
                # Smooth body movement with low-pass filter
                body_smoothing = getattr(self.config, 'body_rotation_smoothing', 0.15)
                self._body_yaw += body_smoothing * (self._target_body_yaw - self._body_yaw)
                
                # HEAD: Follow body yaw to stay aligned with body rotation
                target_yaw = self._body_yaw
            else:
                # Body rotation disabled - head does all tracking with PID
                self._pid_integral_yaw += error_x * dt
                self._pid_integral_yaw = np.clip(self._pid_integral_yaw, -integral_limit, integral_limit)
                derivative_yaw = (error_x - self._pid_last_error_yaw) / dt
                self._pid_last_error_yaw = error_x
                
                pid_output_yaw = kp * error_x + ki * self._pid_integral_yaw + kd * derivative_yaw
                target_yaw = -np.clip(pid_output_yaw * max_yaw, -max_yaw, max_yaw)
            
            # Apply smoothing for final output
            smoothing = self.config.face_tracking_smoothing
            self._smoothed_pitch += smoothing * (target_pitch - self._smoothed_pitch)
            self._smoothed_yaw += smoothing * (target_yaw - self._smoothed_yaw)
            
            # Apply ramp to prevent jumps when tracking starts
            ramped_pitch = self._smoothed_pitch * self._tracking_ramp
            ramped_yaw = self._smoothed_yaw * self._tracking_ramp
            
            self._last_face_time = time.monotonic()
            
            with self._lock:
                self._offsets = (
                    0.0, 0.0, 0.0, 0.0,
                    ramped_pitch,
                    ramped_yaw,
                )
                self._last_face_bbox = bbox
                self._last_face_center = (int(face_center_x), int(face_center_y))
                self._tracking_active = True
        else:
            time_since_face = time.monotonic() - self._last_face_time
            if time_since_face > self._face_timeout:
                self._decay_offsets()
                # Reset PID init flag so next detection starts fresh
                self._pid_needs_init = True
                # Reset tracked face so we start fresh on next detection
                self._tracked_face_center = None
                self._tracked_face_area = 0.0
            
            with self._lock:
                self._last_face_bbox = None
                self._last_face_center = None
                self._tracking_active = False
    
    def _decay_offsets(self):
        """Gradually decay offsets towards zero and reset PID state."""
        self._smoothed_yaw *= self.config.face_tracking_decay
        self._smoothed_pitch *= self.config.face_tracking_decay
        
        # Also decay body yaw when no face detected
        self._target_body_yaw *= 0.95
        self._body_yaw += 0.1 * (self._target_body_yaw - self._body_yaw)
        
        # Decay PID integral terms to prevent windup
        self._pid_integral_yaw *= 0.9
        self._pid_integral_pitch *= 0.9
        
        if abs(self._smoothed_yaw) < 0.001:
            self._smoothed_yaw = 0.0
        if abs(self._smoothed_pitch) < 0.001:
            self._smoothed_pitch = 0.0
        if abs(self._body_yaw) < 0.001:
            self._body_yaw = 0.0
            self._target_body_yaw = 0.0
            # Reset tracked face when fully decayed
            self._tracked_face_center = None
            self._tracked_face_area = 0.0
        if abs(self._pid_integral_yaw) < 0.001:
            self._pid_integral_yaw = 0.0
            self._pid_last_error_yaw = 0.0
        if abs(self._pid_integral_pitch) < 0.001:
            self._pid_integral_pitch = 0.0
            self._pid_last_error_pitch = 0.0
        
        with self._lock:
            self._offsets = (
                0.0, 0.0, 0.0, 0.0,
                self._smoothed_pitch,
                self._smoothed_yaw,
            )
