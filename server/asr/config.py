"""
Configuration for Vaani Cloud ASR Server.
Optimized for local laptop hosting with NVIDIA GPU acceleration (CUDA)
and Hinglish voice command transcription.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ASRConfig:
    # Server network settings
    HOST: str = os.getenv("VAANI_ASR_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("VAANI_ASR_PORT", 8000))

    # Audio specifications (must strictly match ESP32 INMP441 capture)
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    SAMPLE_WIDTH: int = 2  # 16-bit PCM (2 bytes per sample)

    # ASR Model configuration
    # Options: 'tiny', 'base', 'small', 'medium'
    # 'base' delivers the sweet spot of Hinglish accuracy and sub-100ms latency on RTX 3050
    MODEL_SIZE: str = os.getenv("VAANI_MODEL_SIZE", "base")
    DEVICE: str = os.getenv("VAANI_DEVICE", "cuda")  # 'cuda', 'cpu', or 'auto'
    COMPUTE_TYPE: str = os.getenv("VAANI_COMPUTE_TYPE", "float16")  # 'float16', 'int8', or 'auto'

    # Hinglish conditioning prompt
    # Biases Whisper's decoder to accurately transcribe code-mixed Hindi + English IoT commands
    INITIAL_PROMPT: str = (
        "Vaani, bedroom ki light on karo. Fan ki speed 3 karo. "
        "What is the temperature? Living room ka AC off kar do. "
        "Play some relaxing music."
    )
    DEFAULT_LANGUAGE: Optional[str] = os.getenv("VAANI_LANG", None)  # None = auto-detect, or 'hi', 'en'

    # Server-side VAD & silence segmentation parameters
    ENERGY_THRESHOLD: float = float(os.getenv("VAANI_ENERGY_TH", 0.015))  # Normalized RMS energy threshold
    SILENCE_TIMEOUT_SEC: float = float(os.getenv("VAANI_SILENCE_TIMEOUT", 0.8))  # Trailing silence to mark end of speech
    MIN_SPEECH_DURATION_SEC: float = 0.4  # Ignore clicks/pops shorter than this
    MAX_RECORDING_SEC: float = 10.0  # Hard cutoff safety cap

    # Streaming buffer configuration
    CHUNK_SIZE_BYTES: int = 1024  # 512 samples @ 16kHz = 32ms per chunk


config = ASRConfig()
