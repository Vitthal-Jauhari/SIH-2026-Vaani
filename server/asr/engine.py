"""
ASR Engine abstraction powered by faster-whisper (CTranslate2).
Optimized for low-latency Hinglish voice command transcription on NVIDIA GPU or CPU.
"""

import ctypes
import logging
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple
import numpy as np

from config import config

logger = logging.getLogger("vaani_asr.engine")


def _preload_windows_cuda_dlls():
    """Ensures nvidia-cublas and nvidia-cudnn DLLs are loaded properly on Windows."""
    if sys.platform != "win32":
        return
    venv_base = sys.prefix
    for pkg in ["cublas", "cudnn", "cuda_nvrtc"]:
        bin_dir = os.path.join(venv_base, "Lib", "site-packages", "nvidia", pkg, "bin")
        if os.path.isdir(bin_dir):
            try:
                os.add_dll_directory(bin_dir)
            except Exception:
                pass
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            for fname in os.listdir(bin_dir):
                if fname.endswith(".dll"):
                    try:
                        ctypes.CDLL(os.path.join(bin_dir, fname))
                    except Exception:
                        pass

_preload_windows_cuda_dlls()


class ASREngine:
    """
    Wraps faster-whisper for fast on-device inference with automatic device detection
    and Hinglish conditioning prompts.
    """

    def __init__(
        self,
        model_size: str = config.MODEL_SIZE,
        device: str = config.DEVICE,
        compute_type: str = config.COMPUTE_TYPE,
        initial_prompt: str = config.INITIAL_PROMPT,
        default_language: Optional[str] = config.DEFAULT_LANGUAGE,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.initial_prompt = initial_prompt
        self.default_language = default_language
        self.model = None

        self._resolve_device_and_compute()

    def _resolve_device_and_compute(self):
        """Auto-detects CUDA availability and selects optimal compute type."""
        resolved_device = self.device
        resolved_compute = self.compute_type

        if resolved_device in ("auto", "cuda"):
            try:
                import ctranslate2
                if ctranslate2.get_cuda_device_count() > 0:
                    resolved_device = "cuda"
                    logger.info("NVIDIA GPU detected via CTranslate2 CUDA support")
                else:
                    logger.warning("No CUDA device found, falling back to CPU")
                    resolved_device = "cpu"
            except Exception as e:
                logger.warning(f"Error checking CUDA devices ({e}), falling back to CPU")
                resolved_device = "cpu"

        if resolved_compute == "auto":
            if resolved_device == "cuda":
                resolved_compute = "float16"
            else:
                resolved_compute = "int8"

        self.device = resolved_device
        self.compute_type = resolved_compute
        logger.info(f"Targeting device={self.device}, compute_type={self.compute_type}")

    def load_model(self):
        """Loads the faster-whisper model into memory/VRAM."""
        if self.model is not None:
            return

        from faster_whisper import WhisperModel

        logger.info(
            f"Loading faster-whisper model '{self.model_size}' "
            f"on {self.device} ({self.compute_type})..."
        )
        t0 = time.time()
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        t_elapsed = (time.time() - t0) * 1000
        logger.info(f"Model loaded successfully in {t_elapsed:.1f} ms")

        # Warm-up inference with 0.5s of silence to prime CUDA kernels
        self._warmup()

    def _warmup(self):
        """Performs a brief dummy inference to initialize CUDA contexts and avoid initial cold-start delay."""
        try:
            logger.info("Performing model warmup...")
            dummy_audio = np.zeros(8000, dtype=np.float32)
            _ = list(
                self.model.transcribe(
                    dummy_audio,
                    beam_size=1,
                    language="en",
                    vad_filter=False,
                )[0]
            )
            logger.info("Warmup complete.")
        except Exception as e:
            logger.warning(f"Warmup skipped or encountered exception: {e}")

    def transcribe(
        self,
        audio_float32: np.ndarray,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transcribes 16kHz float32 audio numpy array.
        Returns a dictionary with transcript, language, and latency metrics.
        """
        if self.model is None:
            self.load_model()

        if len(audio_float32) == 0:
            return {
                "transcript": "",
                "language": None,
                "language_probability": 0.0,
                "asr_latency_ms": 0.0,
                "audio_duration_sec": 0.0,
            }

        audio_duration_sec = len(audio_float32) / config.SAMPLE_RATE
        lang = language or self.default_language
        prompt = initial_prompt or self.initial_prompt

        t0 = time.time()
        segments, info = self.model.transcribe(
            audio_float32,
            beam_size=1,  # Greedy search for lowest latency
            best_of=1,
            temperature=0.0,
            language=lang,
            initial_prompt=prompt,
            vad_filter=False,  # Audio has already been segmented by server VAD
            condition_on_previous_text=False,
        )

        # Collect segment texts
        text_segments = [s.text.strip() for s in segments]
        transcript = " ".join(text_segments).strip()
        asr_latency_ms = (time.time() - t0) * 1000

        return {
            "transcript": transcript,
            "language": info.language,
            "language_probability": float(info.language_probability),
            "asr_latency_ms": round(asr_latency_ms, 2),
            "audio_duration_sec": round(audio_duration_sec, 2),
        }
