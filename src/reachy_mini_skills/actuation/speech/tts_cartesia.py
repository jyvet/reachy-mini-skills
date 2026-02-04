"""Cartesia TTS provider.

Usage:
    from reachy_mini_skills.speech import tts_cartesia
    
    tts = tts_cartesia.create()
    result = await tts.synthesize("Hello world!")
"""

import asyncio
import threading
import time
from typing import Optional, Callable, Any, TYPE_CHECKING

import numpy as np
import pyaudio

from .tts_base import TTSProvider, TTSResult
from ...config import TTSConfig
from ...perception.audio import compute_rms_energy

if TYPE_CHECKING:
    from ...config import Config

__all__ = ["CartesiaTTS", "TTSResult", "create"]


class CartesiaTTS(TTSProvider):
    """Cartesia streaming TTS provider."""
    
    @property
    def name(self) -> str:
        return "cartesia"
    
    async def synthesize(
        self,
        text: str,
        audio_queue: Optional[asyncio.Queue] = None,
        interrupt_event: Optional[asyncio.Event] = None,
        speaker_id: Optional[int] = None,
        emotion: Optional[str] = None,
        reachy_media: Optional[Any] = None
    ) -> TTSResult:
        """Synthesize and play speech using Cartesia TTS.
        
        Args:
            text: Text to synthesize
            audio_queue: Optional queue for barge-in detection
            interrupt_event: Optional event to signal interruption
            speaker_id: Optional speaker device ID for pyaudio output
            emotion: Optional emotion for TTS
            reachy_media: Optional Reachy Mini media manager (reachy_mini.media)
                          If provided, audio will be played through Reachy Mini's speaker
                          using push_audio_sample(). Requires resampling to 16000 Hz.
        """
        from cartesia import Cartesia
        try:
            import scipy.signal
            HAS_SCIPY = True
        except ImportError:
            HAS_SCIPY = False
        
        start = time.perf_counter()
        ttfb = None
        interrupted = False
        stop_playback = threading.Event()
        
        loop = asyncio.get_running_loop()
        
        # Reachy Mini audio operates at 16000 Hz
        REACHY_SAMPLE_RATE = 16000
        
        def run_cartesia_tts():
            nonlocal ttfb, interrupted
            
            client = Cartesia(api_key=self.config.cartesia_api_key)
            
            if emotion:
                transcript = f'<emotion value="{emotion}"/>{text}'
            else:
                transcript = text
            
            # For Reachy Mini media, we need to handle audio differently
            use_reachy = reachy_media is not None
            
            p = None
            stream = None
            reachy_playing = False
            total_samples_pushed = 0  # Track samples for calculating playback duration
            reachy_playback_start = None  # Track when we started pushing audio
            
            if not use_reachy:
                p = pyaudio.PyAudio()
            
            try:
                # Start Reachy audio output if using Reachy Mini
                if use_reachy:
                    try:
                        reachy_media.start_playing()
                        reachy_playing = True
                    except Exception as e:
                        print(f"   (Reachy Mini start_playing error: {e})")
                        return
                
                ws = client.tts.websocket()
                
                for output in ws.send(
                    model_id="sonic-3",
                    transcript=transcript,
                    voice={"mode": "id", "id": self.config.cartesia_voice_id, "__experimental_controls": {"speed": self.config.cartesia_speed}},
                    stream=True,
                    output_format={
                        "container": "raw",
                        "encoding": "pcm_f32le",
                        "sample_rate": self.config.cartesia_sample_rate,
                        "Max_silence_duration_secs": self.config.cartesia_silence_detect,      
                    },
                ):
                    if stop_playback.is_set():
                        interrupted = True
                        # Stop Reachy audio immediately on interrupt during streaming
                        if use_reachy and reachy_playing:
                            try:
                                print("   (Stopping Reachy Mini audio due to interrupt during streaming)")
                                reachy_media.stop_playing()
                                reachy_playing = False
                            except Exception as e:
                                print(f"   (Reachy Mini stop_playing error: {e})")
                        break
                    
                    buffer = output.audio
                    
                    if ttfb is None:
                        ttfb = time.perf_counter() - start
                    
                    if use_reachy:
                        # Convert bytes to float32 array
                        audio_array = np.frombuffer(buffer, dtype=np.float32)
                        
                        # Resample from Cartesia sample rate to Reachy sample rate (16000 Hz)
                        if self.config.cartesia_sample_rate != REACHY_SAMPLE_RATE:
                            if HAS_SCIPY:
                                # Use scipy for high-quality resampling
                                num_samples = int(len(audio_array) * REACHY_SAMPLE_RATE / self.config.cartesia_sample_rate)
                                audio_array = scipy.signal.resample(audio_array, num_samples).astype(np.float32)
                            else:
                                # Simple decimation fallback (lower quality)
                                ratio = self.config.cartesia_sample_rate / REACHY_SAMPLE_RATE
                                indices = np.arange(0, len(audio_array), ratio).astype(int)
                                indices = indices[indices < len(audio_array)]
                                audio_array = audio_array[indices]
                        
                        # Track total samples pushed for calculating playback duration
                        total_samples_pushed += len(audio_array)
                        
                        # Track when we first started pushing audio
                        if reachy_playback_start is None:
                            reachy_playback_start = time.perf_counter()
                        
                        # Push audio samples to Reachy Mini (streaming, non-blocking)
                        try:
                            reachy_media.push_audio_sample(audio_array)
                        except Exception as e:
                            print(f"   (Reachy Mini push_audio_sample error: {e})")
                    else:
                        # Use pyaudio for direct output (blocking)
                        if not stream:
                            stream = p.open(
                                format=pyaudio.paFloat32,
                                channels=1,
                                rate=self.config.cartesia_sample_rate,
                                output=True,
                                output_device_index=speaker_id
                            )
                        stream.write(buffer)
                
                ws.close()
                
                # For Reachy Mini: wait for audio to finish playing
                # push_audio_sample() is non-blocking, so we need to wait for playback
                if use_reachy and total_samples_pushed > 0 and reachy_playback_start and not interrupted:
                    playback_duration = total_samples_pushed / REACHY_SAMPLE_RATE
                    # Calculate how much time has elapsed since playback started
                    elapsed_since_start = time.perf_counter() - reachy_playback_start
                    # Wait for remaining time (plus buffer for audio system latency)
                    remaining_time = playback_duration - elapsed_since_start + 0.3
                    
                    if remaining_time > 0:
                        print(f"   (Audio duration: {playback_duration:.2f}s, elapsed: {elapsed_since_start:.2f}s, waiting: {remaining_time:.2f}s)")
                        # Wait in small increments to allow interrupt
                        wait_start = time.perf_counter()
                        while time.perf_counter() - wait_start < remaining_time:
                            if stop_playback.is_set():
                                interrupted = True
                                # Stop Reachy audio immediately on interrupt
                                if reachy_playing:
                                    try:
                                        print("   (Stopping Reachy Mini audio due to interrupt)")
                                        reachy_media.stop_playing()
                                        reachy_playing = False  # Mark as stopped so finally block doesn't call again
                                    except Exception as e:
                                        print(f"   (Reachy Mini stop_playing error: {e})")
                                break
                            time.sleep(0.05)
                    else:
                        print(f"   (Audio duration: {playback_duration:.2f}s, already played)")
                
            except Exception as e:
                print(f"   (Cartesia TTS error: {e})")
            finally:
                if stream:
                    stream.stop_stream()
                    stream.close()
                if p:
                    p.terminate()
                if use_reachy and reachy_playing:
                    try:
                        reachy_media.stop_playing()
                    except Exception as e:
                        print(f"   (Reachy Mini stop_playing error: {e})")
        
        tts_task = loop.run_in_executor(None, run_cartesia_tts)
        
        async def barge_in_monitor():
            nonlocal interrupted
            if interrupt_event is None:
                print("   (barge_in_monitor: skipping, interrupt_event is None)", flush=True)
                return
            
            print("   (barge_in_monitor: active, monitoring for interrupt)", flush=True)
            # Monitor for external interrupt signal
            while not stop_playback.is_set():
                if interrupt_event.is_set():
                    print("   (TTS interrupted by stop signal)")
                    stop_playback.set()
                    interrupted = True
                    break
                await asyncio.sleep(0.05)
        
        barge_in_task = asyncio.create_task(barge_in_monitor())
        
        try:
            await tts_task
        finally:
            stop_playback.set()
            barge_in_task.cancel()
            try:
                await barge_in_task
            except asyncio.CancelledError:
                pass
        
        total = time.perf_counter() - start
        return TTSResult(
            ttfb=ttfb if ttfb else 0,
            total_time=total,
            was_interrupted=interrupted
        )


def create(config: "Config" = None) -> CartesiaTTS:
    """Create a Cartesia TTS instance."""
    from ...config import Config
    if config is None:
        config = Config()
    return CartesiaTTS(config.tts)
