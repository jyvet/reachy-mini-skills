"""Audio I/O utilities for Reachy Mini Library."""

import asyncio
import numpy as np
import sounddevice as sd
from typing import List, Optional

from ...config import AudioConfig


def compute_rms_energy(audio_data: np.ndarray) -> float:
    """Compute RMS energy of an audio chunk."""
    return float(np.sqrt(np.mean(audio_data ** 2)))


def list_speakers() -> List[int]:
    """
    List available audio output devices and their IDs.
    
    Returns:
        List of available speaker device IDs.
    """
    print("\n🔊 Available audio outputs:")
    devices = sd.query_devices()
    available = []
    for i, dev in enumerate(devices):
        if dev['max_output_channels'] > 0:
            print(f"  Speaker {i}: {dev['name']} ({dev['max_output_channels']} ch, {int(dev['default_samplerate'])} Hz)")
            available.append(i)
    
    if not available:
        print("  No audio outputs found!")
    
    return available


def list_microphones() -> List[int]:
    """
    List available audio input devices and their IDs.
    
    Returns:
        List of available microphone device IDs.
    """
    print("\n🎤 Available audio inputs:")
    devices = sd.query_devices()
    available = []
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            print(f"  Microphone {i}: {dev['name']} ({dev['max_input_channels']} ch, {int(dev['default_samplerate'])} Hz)")
            available.append(i)
    
    if not available:
        print("  No audio inputs found!")
    
    return available


class AudioManager:
    """Manages audio input streaming."""
    
    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self._stream_task: Optional[asyncio.Task] = None
        self._audio_queue: Optional[asyncio.Queue] = None
    
    @property
    def audio_queue(self) -> asyncio.Queue:
        """Get the audio queue, creating if necessary."""
        if self._audio_queue is None:
            self._audio_queue = asyncio.Queue()
        return self._audio_queue
    
    async def start_microphone_stream(self) -> asyncio.Queue:
        """Start microphone streaming and return the audio queue."""
        if self._stream_task is not None:
            return self.audio_queue
        
        # Drain any stale audio from previous sessions
        self.drain_queue()
        
        self._stream_task = asyncio.create_task(self._microphone_stream())
        return self.audio_queue
    
    async def stop_microphone_stream(self):
        """Stop the microphone stream."""
        if self._stream_task is not None:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
            self._stream_task = None
    
    async def _microphone_stream(self):
        """Continuously capture audio from the configured microphone."""
        loop = asyncio.get_running_loop()
        queue = self.audio_queue

        def callback(indata, frames, time_info, status):
            loop.call_soon_threadsafe(
                queue.put_nowait, indata[:, 0].astype(np.float32).copy()
            )

        with sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.config.block_size,
            device=self.config.input_device_id,  # None = system default
            callback=callback,
        ):
            while True:
                await asyncio.sleep(0.1)
    
    def drain_queue(self):
        """Drain any stale audio from the queue."""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break


class ReachyAudioManager:
    """Manages audio input streaming from Reachy Mini's microphone.
    
    Uses reachy_mini.media API:
        - start_recording(): Start recording from robot's microphone
        - get_audio_sample(): Get audio samples (returns np.ndarray[float32] or None)
        - stop_recording(): Stop recording
        
    Audio is captured at 16000 Hz sample rate.
    """
    
    # Reachy Mini audio operates at 16000 Hz
    REACHY_SAMPLE_RATE = 16000
    
    def __init__(self, reachy_media, config: Optional[AudioConfig] = None):
        """Initialize ReachyAudioManager.
        
        Args:
            reachy_media: Reachy Mini media manager (reachy_mini.media)
            config: Optional audio configuration
        """
        self.reachy_media = reachy_media
        self.config = config or AudioConfig()
        self._stream_task: Optional[asyncio.Task] = None
        self._audio_queue: Optional[asyncio.Queue] = None
        self._recording = False
    
    @property
    def audio_queue(self) -> asyncio.Queue:
        """Get the audio queue, creating if necessary."""
        if self._audio_queue is None:
            self._audio_queue = asyncio.Queue()
        return self._audio_queue
    
    async def start_microphone_stream(self) -> asyncio.Queue:
        """Start microphone streaming from Reachy Mini and return the audio queue."""
        if self._stream_task is not None:
            return self.audio_queue
        
        # Drain any stale audio from previous sessions
        self.drain_queue()
        
        self._stream_task = asyncio.create_task(self._microphone_stream())
        return self.audio_queue
    
    async def stop_microphone_stream(self):
        """Stop the microphone stream."""
        if self._stream_task is not None:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
            self._stream_task = None
    
    async def _microphone_stream(self):
        """Continuously capture audio from Reachy Mini's microphone."""
        queue = self.audio_queue
        
        try:
            # Start recording from Reachy Mini
            self.reachy_media.start_recording()
            self._recording = True
            print("[ReachyAudioManager] Started recording from Reachy Mini microphone")
            
            while True:
                # Get audio sample from Reachy Mini
                # get_audio_sample() returns np.ndarray[float32] or None
                sample = self.reachy_media.get_audio_sample()
                
                if sample is not None:
                    # Put audio samples in queue
                    queue.put_nowait(sample.astype(np.float32))
                else:
                    # No data available, wait a bit
                    await asyncio.sleep(0.01)
                
                # Small sleep to prevent busy loop
                await asyncio.sleep(0.001)
                
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[ReachyAudioManager] Error in microphone stream: {e}")
        finally:
            # Stop recording
            if self._recording:
                try:
                    self.reachy_media.stop_recording()
                    self._recording = False
                    print("[ReachyAudioManager] Stopped recording from Reachy Mini microphone")
                except Exception as e:
                    print(f"[ReachyAudioManager] Error stopping recording: {e}")
    
    def drain_queue(self):
        """Drain any stale audio from the queue."""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
