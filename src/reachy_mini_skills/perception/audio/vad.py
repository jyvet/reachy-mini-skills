"""Voice Activity Detection (VAD) for Reachy Mini Library.

This module provides VAD implementations used by STT providers.

Usage:
    from reachy_mini_skills.perception.audio.vad import EnergyVAD, WhisperSocketVAD
    
    # Energy-based VAD (used by Deepgram, Cartesia, Unmute)
    vad = EnergyVAD(config)
    vad.add_audio(audio_data)
    if vad.voice_detected:
        buffered_audio = vad.get_buffered_audio()
    
    # WhisperSocket-style VAD (accumulates sentences)
    vad = WhisperSocketVAD(config)
    sentence_complete, audio_data = vad.process(audio_chunk)
"""

import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Deque, List, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from ...config import STTConfig

__all__ = [
    "VADBase",
    "EnergyVAD", 
    "WhisperSocketVAD",
    "WhisperVAD",  # Backwards compatibility alias
    "VADResult",
    "compute_rms_energy",
]


def compute_rms_energy(audio_data: np.ndarray) -> float:
    """Compute RMS energy of an audio chunk.
    
    Args:
        audio_data: Audio samples as numpy array (float32 or int16).
        
    Returns:
        RMS energy value.
    """
    return float(np.sqrt(np.mean(audio_data ** 2)))


@dataclass
class VADResult:
    """Result from VAD processing."""
    voice_detected: bool
    speech_ended: bool
    audio_data: Optional[Union[np.ndarray, List[np.ndarray]]] = None


class VADBase(ABC):
    """Base class for Voice Activity Detection."""
    
    def __init__(self, config: "STTConfig"):
        self.config = config
        self._voice_detected = False
        self._start_time = time.perf_counter()
    
    @property
    def voice_detected(self) -> bool:
        """Whether voice has been detected."""
        return self._voice_detected
    
    @abstractmethod
    def process(self, audio_data: np.ndarray) -> VADResult:
        """Process an audio chunk and return VAD result.
        
        Args:
            audio_data: Audio samples as numpy array.
            
        Returns:
            VADResult with detection status.
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset VAD state."""
        pass
    
    def check_timeout(self) -> bool:
        """Check if pre-speech timeout has been reached.
        
        Returns:
            True if timeout reached without voice detection.
        """
        if not self._voice_detected:
            return time.perf_counter() - self._start_time > self.config.pre_speech_timeout
        return False


class EnergyVAD(VADBase):
    """Energy-based Voice Activity Detection.
    
    Used by streaming STT providers (Deepgram, Cartesia, Unmute) that need
    to detect voice before starting the transcription stream.
    
    Features:
    - RMS energy threshold detection
    - Consecutive frames requirement to avoid false triggers
    - Audio buffering to capture pre-voice audio
    - Activity tracking for silence detection
    """
    
    def __init__(self, config: "STTConfig"):
        super().__init__(config)
        # Circular buffer for pre-voice audio
        buffer_size = int(config.buffer_seconds * 24000 / 1920)
        self._audio_buffer: Deque[np.ndarray] = deque(maxlen=buffer_size)
        self._consecutive_energy_frames = 0
        self._last_activity_time = time.perf_counter()
    
    @property
    def last_activity_time(self) -> float:
        """Time of last voice activity."""
        return self._last_activity_time
    
    def process(self, audio_data: np.ndarray) -> VADResult:
        """Process audio chunk for voice detection.
        
        Args:
            audio_data: Audio samples as numpy array.
            
        Returns:
            VADResult indicating if voice was detected.
        """
        energy = compute_rms_energy(audio_data)
        
        if energy > self.config.energy_threshold:
            self._consecutive_energy_frames += 1
            if self._consecutive_energy_frames >= self.config.energy_frames_required:
                if not self._voice_detected:
                    self._voice_detected = True
                self._last_activity_time = time.perf_counter()
        else:
            self._consecutive_energy_frames = max(0, self._consecutive_energy_frames - 1)
        
        # Always buffer audio (useful for capturing pre-voice context)
        self._audio_buffer.append(audio_data)
        
        return VADResult(
            voice_detected=self._voice_detected,
            speech_ended=False,
            audio_data=audio_data
        )
    
    def update_activity(self, audio_data: np.ndarray) -> bool:
        """Update activity tracking during streaming (after voice detected).
        
        Args:
            audio_data: Audio samples as numpy array.
            
        Returns:
            True if activity detected (energy above half threshold).
        """
        energy = compute_rms_energy(audio_data)
        if energy > self.config.energy_threshold * 0.5:
            self._last_activity_time = time.perf_counter()
            return True
        return False
    
    def get_buffered_audio(self) -> List[np.ndarray]:
        """Get all buffered audio chunks.
        
        Returns:
            List of buffered audio chunks.
        """
        return list(self._audio_buffer)
    
    def reset(self) -> None:
        """Reset VAD state."""
        self._voice_detected = False
        self._consecutive_energy_frames = 0
        self._audio_buffer.clear()
        self._start_time = time.perf_counter()
        self._last_activity_time = time.perf_counter()


class WhisperSocketVAD(VADBase):
    """VAD that accumulates complete sentences before triggering send.
    
    Used by WhisperSocket STT which processes complete audio segments rather
    than streaming. Detects speech onset, accumulates audio, and detects
    natural sentence endings based on silence.
    
    Features:
    - RMS threshold-based speech detection
    - Adaptive silence threshold based on speech duration
    - Natural pause detection based on energy drop
    - Pre-speech buffer for capturing word onsets
    - Sentence accumulation with automatic end detection
    """
    
    def __init__(self, config: "STTConfig"):
        super().__init__(config)
        self._is_speaking = False
        self._speech_chunks = 0
        self._silence_chunks = 0
        self._consecutive_silence = 0
        self._pre_buffer: Deque[np.ndarray] = deque(maxlen=config.whispersocket_pre_speech_buffer)
        self._sentence_buffer: List[np.ndarray] = []
        self._total_speech_chunks = 0
        self._energy_history: Deque[float] = deque(maxlen=10)
        self._peak_energy = 0.0
    
    @property
    def is_speaking(self) -> bool:
        """Whether speech is currently active."""
        return self._is_speaking
    
    def _get_adaptive_silence_threshold(self) -> int:
        """Get silence threshold based on speech duration.
        
        Longer utterances get longer silence thresholds to avoid
        cutting off mid-sentence.
        """
        speech_duration_secs = self._total_speech_chunks * 0.08
        
        if speech_duration_secs < 1.0:
            return self.config.whispersocket_silence_chunks_short
        elif speech_duration_secs < 3.0:
            return int(
                self.config.whispersocket_silence_chunks_short + 
                (self.config.whispersocket_silence_chunks - self.config.whispersocket_silence_chunks_short) * 
                (speech_duration_secs - 1.0) / 2.0
            )
        else:
            return self.config.whispersocket_silence_chunks
    
    def _is_natural_pause(self, current_energy: float) -> bool:
        """Detect if current frame represents a natural pause.
        
        A natural pause is detected when energy drops significantly
        from the peak energy observed during speech.
        """
        if len(self._energy_history) < 3:
            return False
        
        if self._peak_energy > 0:
            energy_ratio = current_energy / self._peak_energy
            if energy_ratio < 0.1:
                return True
        
        return False
    
    def _convert_to_int16(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Convert audio to int16 format."""
        if audio_chunk.dtype == np.float32 or audio_chunk.dtype == np.float64:
            return (audio_chunk * 32767).astype(np.int16)
        return audio_chunk.astype(np.int16)
    
    def process(self, audio_chunk: np.ndarray) -> VADResult:
        """Process audio chunk and detect sentence completion.
        
        Args:
            audio_chunk: Audio samples as numpy array.
            
        Returns:
            VADResult with speech_ended=True when a complete sentence
            is detected, along with the accumulated audio data.
        """
        audio_int16 = self._convert_to_int16(audio_chunk)
        
        rms = np.sqrt(np.mean(np.square(audio_int16.astype(np.float32))))
        is_speech = rms > self.config.whispersocket_vad_threshold
        
        self._energy_history.append(rms)
        if is_speech and rms > self._peak_energy:
            self._peak_energy = rms
        
        if is_speech:
            self._consecutive_silence = 0
            self._silence_chunks = 0
            self._speech_chunks += 1
            self._total_speech_chunks += 1
            
            if not self._is_speaking and self._speech_chunks >= 1:
                self._is_speaking = True
                self._voice_detected = True
                
                # Transfer pre-buffer to sentence buffer
                for pre_chunk in self._pre_buffer:
                    self._sentence_buffer.append(pre_chunk)
                self._pre_buffer.clear()
            
            if self._is_speaking:
                self._sentence_buffer.append(audio_int16)
            else:
                self._pre_buffer.append(audio_int16)
            
            return VADResult(
                voice_detected=self._voice_detected,
                speech_ended=False,
                audio_data=None
            )
        else:
            # Silence frame
            self._consecutive_silence += 1
            self._speech_chunks = 0
            
            if self._is_speaking:
                self._sentence_buffer.append(audio_int16)
                
                if self._consecutive_silence >= self.config.whispersocket_min_silence_for_count:
                    self._silence_chunks += 1
                
                silence_threshold = self._get_adaptive_silence_threshold()
                
                if self._is_natural_pause(rms):
                    silence_threshold = min(silence_threshold, self.config.whispersocket_silence_chunks_short)
                
                if self._silence_chunks >= silence_threshold:
                    # Sentence complete
                    self._is_speaking = False
                    self._silence_chunks = 0
                    self._consecutive_silence = 0
                    self._total_speech_chunks = 0
                    self._peak_energy = 0
                    self._energy_history.clear()
                    
                    complete_audio = self._sentence_buffer.copy()
                    self._sentence_buffer = []
                    
                    return VADResult(
                        voice_detected=self._voice_detected,
                        speech_ended=True,
                        audio_data=complete_audio
                    )
                else:
                    return VADResult(
                        voice_detected=self._voice_detected,
                        speech_ended=False,
                        audio_data=None
                    )
            else:
                self._pre_buffer.append(audio_int16)
                return VADResult(
                    voice_detected=self._voice_detected,
                    speech_ended=False,
                    audio_data=None
                )
    
    def get_sentence_buffer(self) -> List[np.ndarray]:
        """Get current sentence buffer contents.
        
        Returns:
            List of audio chunks in the sentence buffer.
        """
        return self._sentence_buffer.copy()
    
    def reset(self) -> None:
        """Reset VAD state."""
        self._voice_detected = False
        self._is_speaking = False
        self._speech_chunks = 0
        self._silence_chunks = 0
        self._consecutive_silence = 0
        self._pre_buffer.clear()
        self._sentence_buffer = []
        self._total_speech_chunks = 0
        self._energy_history.clear()
        self._peak_energy = 0.0
        self._start_time = time.perf_counter()


# Backwards compatibility alias
WhisperVAD = WhisperSocketVAD
