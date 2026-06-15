"""Cartesia STT provider.

Usage:
    from reachy_mini_skills.perception.audio import stt_cartesia
    
    stt = stt_cartesia.create()
    result = await stt.transcribe(audio_queue)
"""

import asyncio
import threading
import time
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import resample

from .stt_base import STTProvider, STTResult, has_sentence_ending
from .vad import EnergyVAD
from ...config import STTConfig

if TYPE_CHECKING:
    from ...config import Config

__all__ = ["CartesiaSTT", "STTResult", "create"]


class CartesiaSTT(STTProvider):
    """Cartesia streaming STT provider."""

    # AudioManager captures system microphone audio at 24 kHz by default.
    # ReachyAudioManager already captures at 16 kHz, matching Cartesia's STT default.
    DEFAULT_INPUT_SAMPLE_RATE = 24000
    
    @property
    def name(self) -> str:
        return "cartesia"
    
    async def transcribe(
        self,
        audio_queue: asyncio.Queue,
        bypass_vad: bool = False
    ) -> STTResult:
        """Transcribe audio using Cartesia STT."""
        from cartesia import Cartesia
        
        start = time.perf_counter()
        ttfb = None
        transcript_text = ""
        speech_started = False
        voice_detected = bypass_vad
        stop_receiving = threading.Event()
        last_word_time = None
        
        vad = EnergyVAD(self.config)
        
        loop = asyncio.get_running_loop()

        def to_cartesia_pcm(audio_data: np.ndarray) -> bytes:
            """Convert float32 microphone audio to Cartesia's configured PCM stream."""
            audio_array = np.asarray(audio_data, dtype=np.float32)
            input_sample_rate = self.config.input_sample_rate or self.DEFAULT_INPUT_SAMPLE_RATE

            if input_sample_rate != self.config.cartesia_sample_rate and len(audio_array) > 0:
                num_samples = int(len(audio_array) * self.config.cartesia_sample_rate / input_sample_rate)
                audio_array = resample(audio_array, max(1, num_samples)).astype(np.float32)

            return (np.clip(audio_array, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        
        def run_cartesia_stt():
            nonlocal ttfb, transcript_text, speech_started, voice_detected
            nonlocal last_word_time, start
            
            client = Cartesia(api_key=self.config.cartesia_api_key)
            if not hasattr(client.stt, "manual_finalize"):
                raise RuntimeError(
                    "Cartesia STT ink2 requires cartesia>=3.2.0. "
                    "Upgrade the app environment so reachy-mini-skills installs the newer SDK."
                )
            
            ws_context = client.stt.manual_finalize.websocket(
                model="ink-2",
                encoding="pcm_s16le",
                sample_rate=self.config.cartesia_sample_rate,
            )
            ws = ws_context.enter() if hasattr(ws_context, "enter") else ws_context.__enter__()

            def send_audio_bytes(audio_bytes: bytes) -> None:
                if hasattr(ws, "send_raw"):
                    ws.send_raw(audio_bytes)
                else:
                    ws.send(audio_bytes)

            def receive_events():
                if hasattr(ws, "receive"):
                    return ws.receive()
                return ws

            def event_value(event, key, default=None):
                if isinstance(event, dict):
                    return event.get(key, default)
                return getattr(event, key, default)
            
            try:
                # Drain queue
                while not audio_queue.empty():
                    try:
                        audio_queue.get_nowait()
                    except:
                        break
                
                # Phase 1: Wait for voice
                if not bypass_vad:
                    print("   (waiting for voice...)", end="", flush=True)
                    
                    while not stop_receiving.is_set() and not voice_detected:
                        try:
                            audio_data = audio_queue.get_nowait()
                        except:
                            time.sleep(0.01)
                            continue
                        
                        # Use VAD to process audio
                        result = vad.process(audio_data)
                        
                        if result.voice_detected:
                            voice_detected = True
                            print("\r   (voice detected!)      ", flush=True)
                        
                        if time.perf_counter() - start > self.config.pre_speech_timeout:
                            print("\r   (no voice detected - timeout)    ")
                            stop_receiving.set()
                            try:
                                ws.send("close")
                                ws.close()
                            except:
                                pass
                            return  # Will return STTResult with empty text
                else:
                    print("   (VAD bypassed, streaming directly to STT...)", flush=True)
                
                # Phase 2: Send buffered audio
                if voice_detected:
                    for buffered_audio in vad.get_buffered_audio():
                        if stop_receiving.is_set():
                            break
                        send_audio_bytes(to_cartesia_pcm(buffered_audio))
                
                # Phase 3: Continue streaming
                def send_audio():
                    while not stop_receiving.is_set():
                        try:
                            audio_data = audio_queue.get_nowait()
                        except:
                            time.sleep(0.01)
                            continue
                        
                        vad.update_activity(audio_data)
                        
                        try:
                            send_audio_bytes(to_cartesia_pcm(audio_data))
                        except:
                            break
                    
                    try:
                        ws.send("finalize")
                        ws.send("close")
                    except:
                        pass
                
                def timeout_monitor():
                    nonlocal last_word_time, transcript_text
                    while not stop_receiving.is_set():
                        time.sleep(0.1)
                        current_time = time.perf_counter()
                        elapsed = current_time - start
                        
                        if elapsed > self.config.max_duration:
                            print("\n   (max duration reached)")
                            stop_receiving.set()
                            break
                        
                        if speech_started and last_word_time:
                            time_since_last_word = current_time - last_word_time
                            
                            if has_sentence_ending(transcript_text):
                                if time_since_last_word > self.config.silence_timeout_with_punct:
                                    stop_receiving.set()
                                    break
                            elif time_since_last_word > self.config.silence_timeout:
                                stop_receiving.set()
                                break
                        
                        if voice_detected and not speech_started:
                            time_since_voice = current_time - vad.last_activity_time
                            if time_since_voice > self.config.silence_timeout * 2:
                                print("\n   (voice detected but no words recognized)")
                                stop_receiving.set()
                                break
                
                send_thread = threading.Thread(target=send_audio, daemon=True)
                monitor_thread = threading.Thread(target=timeout_monitor, daemon=True)
                send_thread.start()
                monitor_thread.start()
                
                for result in receive_events():
                    if stop_receiving.is_set():
                        break
                    
                    result_type = event_value(result, "type")
                    if result_type == 'transcript':
                        if ttfb is None:
                            ttfb = time.perf_counter() - start
                        
                        text = event_value(result, "text", "")
                        if text:
                            print(text, end=" ", flush=True)
                            speech_started = True
                            last_word_time = time.perf_counter()
                        
                        if event_value(result, "is_final", False):
                            transcript_text += text
                            stop_receiving.set()
                            break
                    
                    elif result_type == 'done':
                        stop_receiving.set()
                        break

                    elif result_type == 'error':
                        message = event_value(result, "message", "unknown error")
                        print(f"\n   (Cartesia STT error: {message})")
                        stop_receiving.set()
                        break
                
                stop_receiving.set()
                send_thread.join(timeout=1.0)
                monitor_thread.join(timeout=1.0)
                
            except Exception as e:
                print(f"\n   (Cartesia STT error: {e})")
            finally:
                try:
                    if hasattr(ws_context, "__exit__"):
                        ws_context.__exit__(None, None, None)
                    else:
                        ws.close()
                except:
                    pass
        
        await loop.run_in_executor(None, run_cartesia_stt)
        
        total = time.perf_counter() - start
        print()
        
        return STTResult(
            text=transcript_text,
            ttfb=ttfb if ttfb else 0,
            total_time=total
        )


def create(config: "Config" = None) -> CartesiaSTT:
    """Create a Cartesia STT instance."""
    from ...config import Config
    if config is None:
        config = Config()
    return CartesiaSTT(config.stt)
