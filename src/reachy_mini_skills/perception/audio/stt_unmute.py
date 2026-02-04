"""Unmute STT provider.

Usage:
    from reachy_mini_skills.perception.audio import stt_unmute
    
    stt = stt_unmute.create()
    result = await stt.transcribe(audio_queue)
"""

import asyncio
import time
from typing import TYPE_CHECKING

import msgpack
import numpy as np
import websockets

from .stt_base import STTProvider, STTResult, has_sentence_ending
from .vad import EnergyVAD
from ...config import STTConfig

if TYPE_CHECKING:
    from ...config import Config

__all__ = ["UnmuteSTT", "STTResult", "create"]


class UnmuteSTT(STTProvider):
    """Unmute streaming STT provider."""
    
    @property
    def name(self) -> str:
        return "unmute"
    
    async def transcribe(
        self,
        audio_queue: asyncio.Queue,
        bypass_vad: bool = False
    ) -> STTResult:
        """Transcribe audio using Unmute STT."""
        transcript_words = []
        start = time.perf_counter()
        ttfb = None
        speech_started = False
        voice_detected = bypass_vad
        stop_receiving = asyncio.Event()
        last_word_time = None
        
        vad = EnergyVAD(self.config)

        url = f"{self.config.unmute_ws}/api/asr-streaming"
        headers = {"kyutai-api-key": "public_token"}

        async def sender(websocket):
            nonlocal voice_detected
            
            # Drain queue first
            while not audio_queue.empty():
                try:
                    audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            # Phase 1: Wait for voice activity
            if not bypass_vad:
                print("   (waiting for voice...)", end="", flush=True)
                
                while not stop_receiving.is_set() and not voice_detected:
                    try:
                        audio_data = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                        result = vad.process(audio_data)
                        
                        if result.voice_detected:
                            voice_detected = True
                            print("\r   (voice detected!)      ", flush=True)
                        
                        if vad.check_timeout():
                            print("\r   (no voice detected)    ")
                            stop_receiving.set()
                            return
                            
                    except asyncio.TimeoutError:
                        continue
                    except asyncio.CancelledError:
                        return
            else:
                print("   (VAD bypassed, streaming directly to STT...)", flush=True)
            
            # Phase 2: Send buffered audio
            if voice_detected:
                for buffered_audio in vad.get_buffered_audio():
                    if stop_receiving.is_set():
                        break
                    chunk = {"type": "Audio", "pcm": [float(x) for x in buffered_audio]}
                    msg = msgpack.packb(chunk, use_bin_type=True, use_single_float=True)
                    try:
                        await websocket.send(msg)
                    except websockets.ConnectionClosed:
                        break
            
            # Phase 3: Continue streaming
            while not stop_receiving.is_set():
                try:
                    audio_data = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                    vad.update_activity(audio_data)
                    
                    chunk = {"type": "Audio", "pcm": [float(x) for x in audio_data]}
                    msg = msgpack.packb(chunk, use_bin_type=True, use_single_float=True)
                    await websocket.send(msg)
                    
                except asyncio.TimeoutError:
                    continue
                except websockets.ConnectionClosed:
                    break
                except asyncio.CancelledError:
                    return

        async def receiver(websocket):
            nonlocal ttfb, speech_started, last_word_time
            try:
                async for message in websocket:
                    if stop_receiving.is_set():
                        break
                        
                    data = msgpack.unpackb(message, raw=False)

                    if data["type"] == "Word":
                        if ttfb is None:
                            ttfb = time.perf_counter() - start
                        transcript_words.append(data["text"])
                        print(data["text"], end=" ", flush=True)
                        speech_started = True
                        last_word_time = time.perf_counter()
                        vad.update_activity(np.array([1.0], dtype=np.float32))  # Force activity update
                        
                    elif data["type"] == "Step":
                        prs = data.get("prs", [0, 0, 0, 0])
                        pause_short = prs[self.config.pause_prediction_head_index]
                        pause_medium = prs[self.config.pause_prediction_head_index_medium]
                        
                        if speech_started:
                            current_text = " ".join(transcript_words)
                            has_punct = has_sentence_ending(current_text)
                            
                            if has_punct and pause_short > self.config.pause_prediction_threshold_with_punct:
                                time_since = time.perf_counter() - last_word_time if last_word_time else 0
                                if time_since > 0.8:
                                    stop_receiving.set()
                            elif pause_short > self.config.pause_prediction_threshold and pause_medium > 0.6:
                                time_since = time.perf_counter() - last_word_time if last_word_time else 0
                                if time_since > 0.6:
                                    stop_receiving.set()

            except websockets.ConnectionClosed:
                pass
            except asyncio.CancelledError:
                pass

        async def timeout_monitor():
            nonlocal last_word_time
            while not stop_receiving.is_set():
                await asyncio.sleep(0.1)
                current_time = time.perf_counter()
                elapsed = current_time - start
                
                if elapsed > self.config.max_duration:
                    print("\n   (max duration reached)")
                    stop_receiving.set()
                    break
                
                if speech_started and last_word_time:
                    time_since_last_word = current_time - last_word_time
                    current_text = " ".join(transcript_words)
                    
                    if has_sentence_ending(current_text):
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

        print(f"   (connecting to Unmute STT: {url})")
        try:
            async with websockets.connect(url, additional_headers=headers) as websocket:
                send_task = asyncio.create_task(sender(websocket))
                recv_task = asyncio.create_task(receiver(websocket))
                monitor_task = asyncio.create_task(timeout_monitor())

                await monitor_task
                
                send_task.cancel()
                recv_task.cancel()
                try:
                    await send_task
                except asyncio.CancelledError:
                    pass
                try:
                    await recv_task
                except asyncio.CancelledError:
                    pass
                    
        except websockets.exceptions.WebSocketException as e:
            print(f"\n   (STT connection error: {e})")
        except Exception as e:
            print(f"\n   (STT error: {e})")

        total = time.perf_counter() - start
        final_text = " ".join(transcript_words)
        print()

        return STTResult(
            text=final_text,
            ttfb=ttfb if ttfb else 0,
            total_time=total
        )


def create(config: "Config" = None) -> UnmuteSTT:
    """Create an Unmute STT instance."""
    from ...config import Config
    if config is None:
        config = Config()
    return UnmuteSTT(config.stt)
