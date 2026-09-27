"""
Server-side Voice Activity and Silence Detector for real-time audio streams.
Segments continuous incoming PCM frames into discrete utterances based on energy and silence duration.
"""

import time
from enum import Enum
from typing import Optional, Tuple
import numpy as np

from config import config


class SpeechState(Enum):
    IDLE = "idle"
    SPEECH_ACTIVE = "speech_active"
    TRAILING_SILENCE = "trailing_silence"
    COMPLETE = "complete"


class StreamVAD:
    """
    Stateful audio stream processor.
    Buffers incoming raw PCM bytes, computes RMS energy, and flags completion
    when speech is followed by sufficient trailing silence.
    """

    def __init__(
        self,
        sample_rate: int = config.SAMPLE_RATE,
        energy_threshold: float = config.ENERGY_THRESHOLD,
        silence_timeout_sec: float = config.SILENCE_TIMEOUT_SEC,
        min_speech_duration_sec: float = config.MIN_SPEECH_DURATION_SEC,
        max_recording_sec: float = config.MAX_RECORDING_SEC,
    ):
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.silence_timeout_sec = silence_timeout_sec
        self.min_speech_duration_sec = min_speech_duration_sec
        self.max_recording_sec = max_recording_sec

        self.reset()

    def reset(self):
        """Resets stream buffer for a new utterance."""
        self.state = SpeechState.IDLE
        self.raw_buffer = bytearray()
        self.speech_start_time: Optional[float] = None
        self.silence_start_time: Optional[float] = None
        self.first_chunk_time: Optional[float] = None
        self.total_speech_samples = 0

    def compute_rms(self, pcm_int16: np.ndarray) -> float:
        """Computes root mean square (RMS) energy normalized to [0.0, 1.0]."""
        if len(pcm_int16) == 0:
            return 0.0
        # Convert to float in [-1.0, 1.0]
        floats = pcm_int16.astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(floats ** 2)))

    def feed(self, pcm_chunk_bytes: bytes) -> Tuple[SpeechState, bool]:
        """
        Feeds a chunk of 16-bit mono PCM bytes into the VAD state machine.
        Returns (current_state, is_complete).
        """
        if not pcm_chunk_bytes:
            return self.state, False

        now = time.time()
        if self.first_chunk_time is None:
            self.first_chunk_time = now

        # Convert to int16 numpy array
        int16_samples = np.frombuffer(pcm_chunk_bytes, dtype=np.int16)
        rms = self.compute_rms(int16_samples)
        is_speech = rms >= self.energy_threshold

        # Append to buffer
        self.raw_buffer.extend(pcm_chunk_bytes)
        num_samples = len(int16_samples)

        # State transitions
        if self.state == SpeechState.IDLE:
            if is_speech:
                self.state = SpeechState.SPEECH_ACTIVE
                self.speech_start_time = now
                self.total_speech_samples += num_samples
                self.silence_start_time = None
        elif self.state == SpeechState.SPEECH_ACTIVE:
            self.total_speech_samples += num_samples
            if not is_speech:
                self.state = SpeechState.TRAILING_SILENCE
                self.silence_start_time = now
        elif self.state == SpeechState.TRAILING_SILENCE:
            if is_speech:
                # User resumed speaking
                self.state = SpeechState.SPEECH_ACTIVE
                self.total_speech_samples += num_samples
                self.silence_start_time = None
            else:
                # Silence continues
                silence_duration = now - (self.silence_start_time or now)
                speech_duration = (
                    self.total_speech_samples / self.sample_rate
                )

                if (
                    silence_duration >= self.silence_timeout_sec
                    and speech_duration >= self.min_speech_duration_sec
                ):
                    self.state = SpeechState.COMPLETE
                    return self.state, True

        # Check maximum recording timeout safety cap
        total_duration = len(self.raw_buffer) / (self.sample_rate * 2)
        if total_duration >= self.max_recording_sec:
            self.state = SpeechState.COMPLETE
            return self.state, True

        return self.state, False

    def get_audio_float32(self) -> np.ndarray:
        """Returns the buffered audio as a 1D float32 numpy array normalized to [-1.0, 1.0]."""
        if len(self.raw_buffer) == 0:
            return np.empty(0, dtype=np.float32)
        int16_samples = np.frombuffer(self.raw_buffer, dtype=np.int16)
        return int16_samples.astype(np.float32) / 32768.0

    @property
    def duration_sec(self) -> float:
        """Returns total duration of accumulated audio in seconds."""
        return len(self.raw_buffer) / (self.sample_rate * 2)
