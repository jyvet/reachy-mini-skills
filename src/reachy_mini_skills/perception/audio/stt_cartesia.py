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

from .stt_base import STTProvider, STTResult, has_sentence_ending
from .vad import EnergyVAD
from ...config import STTConfig

if TYPE_CHECKING:
    from ...config import Config

__all__ = ["CartesiaSTT", "STTResult", "create"]


class CartesiaSTT(STTProvider):
    """Cartesia streaming STT provider."""
    
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
        
        def run_cartesia_stt():
            nonlocal ttfb, transcript_text, speech_started, voice_detected
            nonlocal last_word_time, start
            
            client = Cartesia(api_key=self.config.cartesia_api_key)
            
            ws = client.stt.websocket(
                model="ink-whisper",
                language="en",
                encoding="pcm_s16le",
                sample_rate=self.config.cartesia_sample_rate,
                min_volume=0.1,
                max_silence_duration_secs=self.config.silence_timeout,
            )
            
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
                                ws.send("done")
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
                        pcm_int16 = (buffered_audio * 32767).astype(np.int16)
                        if 24000 != self.config.cartesia_sample_rate:
                            ratio = 24000 / self.config.cartesia_sample_rate
                            indices = np.arange(0, len(pcm_int16), ratio).astype(int)
                            pcm_int16 = pcm_int16[indices]
                        ws.send(pcm_int16.tobytes())
                
                # Phase 3: Continue streaming
                def send_audio():
                    while not stop_receiving.is_set():
                        try:
                            audio_data = audio_queue.get_nowait()
                        except:
                            time.sleep(0.01)
                            continue
                        
                        vad.update_activity(audio_data)
                        
                        pcm_int16 = (audio_data * 32767).astype(np.int16)
                        if 24000 != self.config.cartesia_sample_rate:
                            ratio = 24000 / self.config.cartesia_sample_rate
                            indices = np.arange(0, len(pcm_int16), ratio).astype(int)
                            pcm_int16 = pcm_int16[indices]
                        
                        try:
                            ws.send(pcm_int16.tobytes())
                        except:
                            break
                    
                    try:
                        ws.send("finalize")
                        ws.send("done")
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
                
                for result in ws.receive():
                    if stop_receiving.is_set():
                        break
                    
                    if result['type'] == 'transcript':
                        if ttfb is None:
                            ttfb = time.perf_counter() - start
                        
                        text = result.get('text', '')
                        if text:
                            print(text, end=" ", flush=True)
                            transcript_text = text
                            speech_started = True
                            last_word_time = time.perf_counter()
                        
                        if result.get('is_final', False):
                            stop_receiving.set()
                            break
                    
                    elif result['type'] == 'done':
                        stop_receiving.set()
                        break
                
                stop_receiving.set()
                send_thread.join(timeout=1.0)
                monitor_thread.join(timeout=1.0)
                
            except Exception as e:
                print(f"\n   (Cartesia STT error: {e})")
            finally:
                try:
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
