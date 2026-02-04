"""WhisperSocket STT provider.

WhisperSocket is a WebSocket-based speech-to-text service using OpenAI's Whisper model.
See: https://github.com/jyvet/whispersocket

Usage:
    from reachy_mini_skills.perception.audio import stt_whispersocket
    
    stt = stt_whispersocket.create()
    result = await stt.transcribe(audio_queue)
"""

import asyncio
import io
import time
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf
import websockets

from .stt_base import STTProvider, STTResult
from .vad import WhisperSocketVAD
from ...config import STTConfig

if TYPE_CHECKING:
    from ...config import Config

__all__ = ["WhisperSocketSTT", "STTResult", "create"]

# Backwards compatibility alias
WhisperSTT = None  # Will be set after class definition


# Re-export WhisperSocketVAD as WhisperSocketSpeechDetector for backwards compatibility
WhisperSocketSpeechDetector = WhisperSocketVAD

# Legacy aliases for backwards compatibility
WhisperSpeechDetector = WhisperSocketSpeechDetector


class WhisperSocketSTT(STTProvider):
    """WhisperSocket STT provider.
    
    Uses WebSocket connection to WhisperSocket server for speech-to-text.
    See: https://github.com/jyvet/whispersocket
    """
    
    @property
    def name(self) -> str:
        return "whispersocket"
    
    async def transcribe(
        self,
        audio_queue: asyncio.Queue,
        bypass_vad: bool = False
    ) -> STTResult:
        """Transcribe audio using WhisperSocket server."""
        start = time.perf_counter()
        ttfb = None
        transcript_text = ""
        
        # Drain queue
        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        detector = WhisperSocketSpeechDetector(self.config)
        audio_chunks_collected = []
        
        try:
            async with websockets.connect(
                self.config.whispersocket_ws,
                ping_interval=20,
                ping_timeout=60,
                max_size=10 * 1024 * 1024,
            ) as ws:
                print("   (connected to WhisperSocket server...)", flush=True)
                
                if bypass_vad:
                    print("   (VAD bypassed, collecting audio...)", flush=True)
                    collection_start = time.perf_counter()
                    
                    while time.perf_counter() - collection_start < self.config.max_duration:
                        try:
                            audio_data = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                            if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                                audio_int16 = (audio_data * 32767).astype(np.int16)
                            else:
                                audio_int16 = audio_data.astype(np.int16)
                            audio_chunks_collected.append(audio_int16)
                        except asyncio.TimeoutError:
                            continue
                        except asyncio.CancelledError:
                            break
                    
                    if audio_chunks_collected:
                        transcript_text = await self._send_and_receive(
                            ws, audio_chunks_collected, 24000
                        )
                        ttfb = time.perf_counter() - start
                        if transcript_text:
                            print(f"{transcript_text}", flush=True)
                else:
                    print("   (waiting for voice...)", end="", flush=True)
                    
                    sentence_complete = False
                    start_wait = time.perf_counter()
                    speech_started_printed = False
                    
                    while not sentence_complete:
                        if time.perf_counter() - start_wait > self.config.pre_speech_timeout:
                            print("\r   (no voice detected)    ")
                            break
                        
                        try:
                            audio_data = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                            vad_result = detector.process(audio_data)
                            
                            # Print speech started message once
                            if detector.is_speaking and not speech_started_printed:
                                print("\r   (speech started...)   ", end=" ", flush=True)
                                speech_started_printed = True
                            
                            if vad_result.speech_ended and vad_result.audio_data:
                                print("\r   (sending to WhisperSocket...)", flush=True)
                                transcript_text = await self._send_and_receive(
                                    ws, vad_result.audio_data, 24000
                                )
                                ttfb = time.perf_counter() - start
                                if transcript_text:
                                    print(f"{transcript_text}", flush=True)
                                sentence_complete = True
                                
                        except asyncio.TimeoutError:
                            continue
                        except asyncio.CancelledError:
                            break
                        
        except websockets.exceptions.WebSocketException as e:
            print(f"\n   (WhisperSocket WebSocket error: {e})")
        except Exception as e:
            print(f"\n   (WhisperSocket STT error: {e})")
        
        total = time.perf_counter() - start
        print()
        
        return STTResult(
            text=transcript_text,
            ttfb=ttfb if ttfb else 0,
            total_time=total
        )
    
    async def _send_and_receive(self, ws, audio_chunks, sample_rate):
        """Send audio to WhisperSocket server and receive transcription."""
        if not audio_chunks:
            return ""
        
        combined = np.concatenate(audio_chunks)
        
        if sample_rate != self.config.whispersocket_sample_rate:
            ratio = self.config.whispersocket_sample_rate / sample_rate
            new_length = int(len(combined) * ratio)
            indices = np.linspace(0, len(combined) - 1, new_length).astype(int)
            combined = combined[indices]
        
        duration = len(combined) / self.config.whispersocket_sample_rate
        
        buf = io.BytesIO()
        sf.write(buf, combined, self.config.whispersocket_sample_rate, format='WAV', subtype='PCM_16')
        buf.seek(0)
        
        print(f"   (sending {duration:.2f}s of audio...)", end=" ", flush=True)
        
        await ws.send(buf.read())
        
        try:
            text = await asyncio.wait_for(ws.recv(), timeout=30.0)
            return text.strip()
        except asyncio.TimeoutError:
            print("   (timeout waiting for transcription)")
            return ""


def create(config: "Config" = None) -> WhisperSocketSTT:
    """Create a WhisperSocket STT instance."""
    from ...config import Config
    if config is None:
        config = Config()
    return WhisperSocketSTT(config.stt)


# Backwards compatibility alias
WhisperSTT = WhisperSocketSTT
