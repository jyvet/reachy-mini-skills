"""Shared test fixtures and configuration for reachy_mini_skills tests.

This module provides:
- Mock classes for external dependencies (robot, audio, network)
- Pytest fixtures for common test setups
- Markers for categorizing tests by skill type
"""

import asyncio
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from reachy_mini_skills.config import (
    STTConfig,
    TTSConfig,
    LLMConfig,
    RobotConfig,
    AudioConfig,
    VisionConfig,
)


# =============================================================================
# Mock Classes
# =============================================================================

class MockHeadPose:
    """Mock head pose matrix."""
    def __init__(self, x=0, y=0, z=0, roll=0, pitch=0, yaw=0):
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        # Create a simple 4x4 identity-like matrix
        self._matrix = np.eye(4)
        self._matrix[0, 3] = x
        self._matrix[1, 3] = y
        self._matrix[2, 3] = z
    
    def __array__(self):
        return self._matrix


class MockRobot:
    """Mock robot for testing motion/actuation skills."""
    
    def __init__(self):
        self._head_pose = MockHeadPose()
        self._antennas = (0.0, 0.0)
        self._body_yaw = 0.0
        self._targets: List[Dict] = []
        self._frames: List[np.ndarray] = []
        
        # Media mock for vision
        self.media = MagicMock()
        self.media.get_frame = MagicMock(return_value=self._generate_mock_frame())
    
    def _generate_mock_frame(self, width=640, height=480) -> np.ndarray:
        """Generate a mock camera frame."""
        return np.zeros((height, width, 3), dtype=np.uint8)
    
    def get_current_head_pose(self) -> MockHeadPose:
        return self._head_pose
    
    def get_current_joint_positions(self) -> Tuple[Any, Tuple[float, float]]:
        return (None, self._antennas)
    
    def set_target(self, head=None, antennas=None, body_yaw=None):
        """Record target for verification."""
        self._targets.append({
            "head": head,
            "antennas": antennas,
            "body_yaw": body_yaw,
            "timestamp": time.monotonic()
        })
        if head is not None:
            self._head_pose = head
        if antennas is not None:
            self._antennas = antennas
        if body_yaw is not None:
            self._body_yaw = body_yaw
    
    def goto_target(self, head=None, antennas=None, duration=1.0, body_yaw=None):
        """Mock goto with duration."""
        self.set_target(head=head, antennas=antennas, body_yaw=body_yaw)
    
    def add_frame_with_face(self, x=320, y=240, w=100, h=100):
        """Add a mock frame with a 'face' (white rectangle) for face tracking tests."""
        frame = self._generate_mock_frame()
        # Draw white rectangle as "face"
        frame[y:y+h, x:x+w] = 255
        self.media.get_frame.return_value = frame


class MockAudioQueue:
    """Mock audio queue for STT testing."""
    
    def __init__(self, chunks: Optional[List[np.ndarray]] = None):
        self._queue = asyncio.Queue()
        self._chunks = chunks or []
        self._index = 0
    
    async def get(self) -> np.ndarray:
        if self._index < len(self._chunks):
            chunk = self._chunks[self._index]
            self._index += 1
            return chunk
        # Return silence
        return np.zeros(1600, dtype=np.float32)
    
    def put_nowait(self, item):
        self._queue.put_nowait(item)
    
    async def put(self, item):
        await self._queue.put(item)
    
    def add_speech_chunk(self, energy: float = 0.1, duration_ms: int = 100):
        """Add a chunk with speech-like energy."""
        samples = int(16000 * duration_ms / 1000)
        chunk = np.random.randn(samples).astype(np.float32) * energy
        self._chunks.append(chunk)


class MockWebSocket:
    """Mock WebSocket for network-based providers."""
    
    def __init__(self, responses: Optional[List[str]] = None):
        self.responses = responses or []
        self._response_index = 0
        self.sent_messages: List[Any] = []
        self.closed = False
    
    async def send(self, message):
        self.sent_messages.append(message)
    
    async def recv(self) -> str:
        if self._response_index < len(self.responses):
            resp = self.responses[self._response_index]
            self._response_index += 1
            return resp
        raise StopIteration("No more responses")
    
    async def close(self):
        self.closed = True
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        await self.close()


class MockHTTPSession:
    """Mock aiohttp session for HTTP-based providers."""
    
    def __init__(self, responses: Optional[List[Dict]] = None):
        self.responses = responses or []
        self._response_index = 0
        self.requests: List[Dict] = []
    
    def post(self, url, **kwargs):
        self.requests.append({"method": "POST", "url": url, **kwargs})
        return self._create_response_context()
    
    def get(self, url, **kwargs):
        self.requests.append({"method": "GET", "url": url, **kwargs})
        return self._create_response_context()
    
    def _create_response_context(self):
        response = MagicMock()
        if self._response_index < len(self.responses):
            data = self.responses[self._response_index]
            self._response_index += 1
        else:
            data = {}
        
        response.status = data.get("status", 200)
        response.json = AsyncMock(return_value=data.get("json", {}))
        response.text = AsyncMock(return_value=data.get("text", ""))
        response.__aenter__ = AsyncMock(return_value=response)
        response.__aexit__ = AsyncMock(return_value=None)
        return response


class MockRecordedMoves:
    """Mock RecordedMoves for emotion testing."""
    
    def __init__(self, emotions: Optional[List[str]] = None):
        self._emotions = emotions or ["happy1", "sad1", "attentive1", "dance1"]
        self._moves = {name: MockMove(name) for name in self._emotions}
    
    def list_moves(self) -> List[str]:
        return self._emotions
    
    def get(self, name: str):
        return self._moves.get(name)


class MockMove:
    """Mock move for emotion testing."""
    
    def __init__(self, name: str, duration: float = 2.0):
        self.name = name
        self.duration = duration
    
    def evaluate(self, t: float) -> Tuple[MockHeadPose, Tuple[float, float], float]:
        """Return a simple pose based on time."""
        return (MockHeadPose(), (0.0, 0.0), 0.0)


# =============================================================================
# Pytest Fixtures
# =============================================================================

@pytest.fixture
def stt_config() -> STTConfig:
    """Default STT configuration for tests."""
    return STTConfig(
        energy_threshold=0.01,
        silence_timeout=0.5,  # Shorter for tests
        max_duration=5.0,     # Shorter for tests
    )


@pytest.fixture
def tts_config() -> TTSConfig:
    """Default TTS configuration for tests."""
    return TTSConfig()


@pytest.fixture
def llm_config() -> LLMConfig:
    """Default LLM configuration for tests."""
    return LLMConfig()


@pytest.fixture
def robot_config() -> RobotConfig:
    """Default robot configuration for tests."""
    return RobotConfig(
        control_loop_frequency_hz=50.0,  # Faster for tests
        face_tracking_fps=10,            # Faster for tests
    )


@pytest.fixture
def audio_config() -> AudioConfig:
    """Default audio configuration for tests."""
    return AudioConfig()


@pytest.fixture
def vision_config() -> VisionConfig:
    """Default vision configuration for tests."""
    return VisionConfig()


@pytest.fixture
def mock_robot() -> MockRobot:
    """Provide a mock robot instance."""
    return MockRobot()


@pytest.fixture
def mock_audio_queue() -> MockAudioQueue:
    """Provide a mock audio queue."""
    return MockAudioQueue()


@pytest.fixture
def mock_http_session() -> MockHTTPSession:
    """Provide a mock HTTP session."""
    return MockHTTPSession()


@pytest.fixture
def mock_moves() -> MockRecordedMoves:
    """Provide mock recorded moves."""
    return MockRecordedMoves()


# =============================================================================
# Pytest Markers
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "stt: Speech-to-text skill tests")
    config.addinivalue_line("markers", "tts: Text-to-speech skill tests")
    config.addinivalue_line("markers", "llm: Language model skill tests")
    config.addinivalue_line("markers", "motion: Motion/movement skill tests")
    config.addinivalue_line("markers", "vision: Vision skill tests")
    config.addinivalue_line("markers", "integration: Integration tests (may need real services)")
    config.addinivalue_line("markers", "slow: Slow tests")
