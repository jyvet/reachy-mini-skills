"""Unmute TTS provider.

Usage:
    from reachy_mini_skills.speech import tts_unmute
    
    tts = tts_unmute.create()
    result = await tts.synthesize("Hello world!")
"""

import asyncio
import time
from urllib.parse import urlencode
from typing import Optional, Any, TYPE_CHECKING

import msgpack
import numpy as np
import sounddevice as sd
import websockets

from .tts_base import TTSProvider, TTSResult
from ...config import TTSConfig
from ...perception.audio import compute_rms_energy

if TYPE_CHECKING:
    from ...config import Config

__all__ = ["UnmuteTTS", "TTSResult", "create"]


class UnmuteTTS(TTSProvider):
    """Unmute streaming TTS provider."""
    
    @property
    def name(self) -> str:
        return "unmute"
    
    async def synthesize(
        self,
        text: str,
        audio_queue: Optional[asyncio.Queue] = None,
        interrupt_event: Optional[asyncio.Event] = None,
        speaker_id: Optional[int] = None,
        emotion: Optional[str] = None,
        reachy_media: Optional[Any] = None
    ) -> TTSResult:
        """Synthesize and play speech using Unmute TTS.
        
        Args:
            text: Text to synthesize
            audio_queue: Optional queue for barge-in detection
            interrupt_event: Optional event to signal interruption
            speaker_id: Optional speaker device ID for sounddevice output
            emotion: Optional emotion (not used by Unmute)
            reachy_media: Optional Reachy Mini media manager (reachy_mini.media)
                          If provided, audio will be played through Reachy Mini's speaker
                          using push_audio_sample(). Requires resampling to 16000 Hz.
        """
        try:
            import scipy.signal
            HAS_SCIPY = True
        except ImportError:
            HAS_SCIPY = False
        
        start = time.perf_counter()
        ttfb = None
        interrupted = False
        
        # Unmute TTS operates at 24000 Hz, Reachy Mini at 16000 Hz
        UNMUTE_SAMPLE_RATE = 24000
        REACHY_SAMPLE_RATE = 16000
        use_reachy = reachy_media is not None

        params = {"voice": self.config.unmute_voice, "format": "PcmMessagePack"}
        uri = f"{self.config.unmute_stream}/api/tts_streaming?{urlencode(params)}"
        headers = {"kyutai-api-key": self.config.unmute_api_key}

        output_queue = asyncio.Queue()
        stop_playback = asyncio.Event()
        
        # For Reachy playback tracking
        reachy_playing = False

        async with websockets.connect(uri, additional_headers=headers) as websocket:
            async def send_text():
                try:
                    for word in text.split():
                        if stop_playback.is_set():
                            return
                        await websocket.send(msgpack.packb({"type": "Text", "text": word}))
                    if not stop_playback.is_set():
                        await websocket.send(msgpack.packb({"type": "Eos"}))
                except websockets.ConnectionClosed:
                    pass

            async def receive_audio():
                nonlocal ttfb
                try:
                    async for message_bytes in websocket:
                        if stop_playback.is_set():
                            break
                        msg = msgpack.unpackb(message_bytes)
                        if msg["type"] == "Audio":
                            if ttfb is None:
                                ttfb = time.perf_counter() - start
                            pcm = np.array(msg["pcm"]).astype(np.float32)
                            await output_queue.put(pcm)
                except websockets.ConnectionClosed:
                    pass
                await output_queue.put(None)

            async def play_audio():
                nonlocal reachy_playing, interrupted
                should_exit = False
                sample_rate = UNMUTE_SAMPLE_RATE
                block_size = 1920
                
                # Track samples for calculating playback duration (for Reachy)
                total_samples_pushed = 0
                reachy_playback_start = None

                if use_reachy:
                    # Use Reachy Mini media for audio output
                    try:
                        reachy_media.start_playing()
                        reachy_playing = True
                    except Exception as e:
                        print(f"[UnmuteTTS] Error starting Reachy playback: {e}")
                        return
                    
                    try:
                        while not should_exit and not stop_playback.is_set():
                            try:
                                pcm_data = await asyncio.wait_for(output_queue.get(), timeout=0.1)
                                if pcm_data is None:
                                    should_exit = True
                                    break
                                
                                # Resample from 24000 Hz to 16000 Hz for Reachy
                                if HAS_SCIPY:
                                    num_samples = int(len(pcm_data) * REACHY_SAMPLE_RATE / UNMUTE_SAMPLE_RATE)
                                    resampled = scipy.signal.resample(pcm_data, num_samples).astype(np.float32)
                                else:
                                    # Simple decimation if scipy not available
                                    ratio = UNMUTE_SAMPLE_RATE / REACHY_SAMPLE_RATE
                                    indices = np.arange(0, len(pcm_data), ratio).astype(int)
                                    indices = indices[indices < len(pcm_data)]
                                    resampled = pcm_data[indices].astype(np.float32)
                                
                                # Track total samples pushed for calculating playback duration
                                total_samples_pushed += len(resampled)
                                
                                # Track when we first started pushing audio
                                if reachy_playback_start is None:
                                    reachy_playback_start = time.perf_counter()
                                
                                # Push to Reachy media as float32 (same as CartesiaTTS)
                                reachy_media.push_audio_sample(resampled)
                                
                            except asyncio.TimeoutError:
                                continue
                        
                        # Wait for audio to finish playing
                        # push_audio_sample() is non-blocking, so we need to wait for playback
                        if total_samples_pushed > 0 and reachy_playback_start and not stop_playback.is_set():
                            playback_duration = total_samples_pushed / REACHY_SAMPLE_RATE
                            # Calculate how much time has elapsed since playback started
                            elapsed_since_start = time.perf_counter() - reachy_playback_start
                            # Wait for remaining time (plus buffer for audio system latency)
                            remaining_time = playback_duration - elapsed_since_start + 0.3
                            
                            if remaining_time > 0:
                                print(f"[UnmuteTTS] Audio duration: {playback_duration:.2f}s, elapsed: {elapsed_since_start:.2f}s, waiting: {remaining_time:.2f}s")
                                # Wait in small increments to allow interrupt
                                wait_start = time.perf_counter()
                                while time.perf_counter() - wait_start < remaining_time:
                                    if stop_playback.is_set():
                                        interrupted = True
                                        break
                                    await asyncio.sleep(0.05)
                    finally:
                        if reachy_playing:
                            try:
                                reachy_media.stop_playing()
                                reachy_playing = False
                            except:
                                pass
                else:
                    # Use sounddevice for regular audio output
                    def audio_callback(outdata, _a, _b, _c):
                        nonlocal should_exit
                        if stop_playback.is_set():
                            should_exit = True
                            outdata[:] = 0
                            return
                        try:
                            pcm_data = output_queue.get_nowait()
                            if pcm_data is not None:
                                outdata[:, 0] = pcm_data
                            else:
                                should_exit = True
                                outdata[:] = 0
                        except asyncio.QueueEmpty:
                            outdata[:] = 0

                    with sd.OutputStream(
                        samplerate=sample_rate,
                        blocksize=block_size,
                        channels=1,
                        callback=audio_callback,
                        device=speaker_id,
                    ):
                        while not should_exit and not stop_playback.is_set():
                            await asyncio.sleep(0.05)

            async def barge_in_monitor():
                nonlocal interrupted
                if audio_queue is None or interrupt_event is None:
                    while not stop_playback.is_set():
                        await asyncio.sleep(0.1)
                    return
                
                consecutive_energy_frames = 0
                energy_threshold = 0.01
                frames_required = 3
                
                while not stop_playback.is_set():
                    try:
                        audio_data = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                        energy = compute_rms_energy(audio_data)
                        
                        if energy > energy_threshold * 1.5:
                            consecutive_energy_frames += 1
                            if consecutive_energy_frames >= frames_required * 2:
                                pass  # Barge-in disabled
                        else:
                            consecutive_energy_frames = max(0, consecutive_energy_frames - 1)
                    except asyncio.TimeoutError:
                        continue

            send_task = asyncio.create_task(send_text())
            receive_task = asyncio.create_task(receive_audio())
            play_task = asyncio.create_task(play_audio())
            barge_in_task = asyncio.create_task(barge_in_monitor())
            
            done, pending = await asyncio.wait(
                [play_task, barge_in_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            stop_playback.set()
            for task in [send_task, receive_task, play_task, barge_in_task]:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        total = time.perf_counter() - start
        return TTSResult(
            ttfb=ttfb if ttfb else 0,
            total_time=total,
            was_interrupted=interrupted
        )


def create(config: "Config" = None) -> UnmuteTTS:
    """Create an Unmute TTS instance."""
    from ...config import Config
    if config is None:
        config = Config()
    return UnmuteTTS(config.tts)
