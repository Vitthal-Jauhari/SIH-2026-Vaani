"""
FastAPI WebSocket and REST Server for Vaani Real-Time Speech Recognition.
Provides low-latency binary audio streaming endpoint for ESP32 edge devices.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import asyncio
import json
import logging
import socket
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

import colorama
from colorama import Fore, Style
import numpy as np
import pyfiglet
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import config
from engine import ASREngine
from vad_stream import StreamVAD, SpeechState

# Initialize colorama for ANSI support on Windows
colorama.init()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vaani_asr.server")

# Global ASR engine instance
asr_engine: ASREngine = None


def print_large_transcript_banner(
    transcript: str,
    latency_ms: float = 0.0,
    duration_sec: float = 0.0,
    lang: str = "hi",
    confidence: float = 0.98,
):
    """Prints a prominent, large-font high-contrast banner for demonstration videos."""
    width = 78
    clean_text = transcript.strip()

    try:
        b_top = f"{Fore.CYAN}╔{'═' * (width - 2)}╗{Style.RESET_ALL}"
        b_mid = f"{Fore.CYAN}╠{'═' * (width - 2)}╣{Style.RESET_ALL}"
        b_bot = f"{Fore.CYAN}╚{'═' * (width - 2)}╝{Style.RESET_ALL}"
    except Exception:
        b_top = f"+{'-' * (width - 2)}+"
        b_mid = f"+{'-' * (width - 2)}+"
        b_bot = f"+{'-' * (width - 2)}+"

    print("\n" + b_top)
    title = "🎯  VAANI CLOUD SPEECH RECOGNITION (ASR)"
    pad_title = max(0, (width - 2 - len(title)) // 2)
    right_pad = max(0, width - 2 - pad_title - len(title))
    print(f"{Fore.CYAN}║{Style.RESET_ALL}{' ' * pad_title}{Fore.YELLOW}{Style.BRIGHT}{title}{Style.RESET_ALL}{' ' * right_pad}{Fore.CYAN}║{Style.RESET_ALL}")
    print(b_mid)

    # Render in ASCII big block letters if reasonable length and ASCII characters
    if clean_text and all(ord(c) < 128 for c in clean_text) and len(clean_text) <= 35:
        try:
            fig = pyfiglet.figlet_format(clean_text.upper(), font="small", width=width - 6)
            for line in fig.splitlines():
                if line.strip():
                    l_pad = max(0, width - 4 - len(line))
                    print(f"{Fore.CYAN}║{Style.RESET_ALL}  {Fore.GREEN}{Style.BRIGHT}{line}{' ' * l_pad}{Style.RESET_ALL}{Fore.CYAN}║{Style.RESET_ALL}")
            print(f"{Fore.CYAN}║{' ' * (width - 2)}║{Style.RESET_ALL}")
        except Exception:
            pass

    # High-contrast readable quote line
    print(f"{Fore.CYAN}║{Style.RESET_ALL}  {Fore.WHITE}{Style.BRIGHT}TRANSCRIPT:{Style.RESET_ALL}{' ' * max(0, width - 15)}{Fore.CYAN}║{Style.RESET_ALL}")
    quote = f'>>> "{clean_text}" <<<'
    pad_q = max(0, width - 4 - len(quote))
    print(f"{Fore.CYAN}║{Style.RESET_ALL}  {Fore.YELLOW}{Style.BRIGHT}{quote}{' ' * pad_q}{Style.RESET_ALL}{Fore.CYAN}║{Style.RESET_ALL}")
    print(b_mid)

    conf_pct = confidence * 100.0 if confidence <= 1.0 else confidence
    stats = f"⚡ Latency: {latency_ms:.1f} ms  |  🔊 Audio: {duration_sec:.1f} s  |  🌐 Lang: {lang} ({conf_pct:.0f}%)"
    pad_s = max(0, width - 4 - len(stats))
    print(f"{Fore.CYAN}║{Style.RESET_ALL}  {Fore.CYAN}{Style.BRIGHT}{stats}{' ' * pad_s}{Style.RESET_ALL}{Fore.CYAN}║{Style.RESET_ALL}")
    print(b_bot + "\n", flush=True)


def get_local_lan_ip() -> str:
    """Attempts to find the local LAN IP address of this machine."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Doesn't need to be reachable, just triggers routing resolution
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes and pre-warms the ASR engine on startup."""
    global asr_engine
    logger.info("Starting Vaani ASR Server...")
    asr_engine = ASREngine(
        model_size=config.MODEL_SIZE,
        device=config.DEVICE,
        compute_type=config.COMPUTE_TYPE,
    )
    asr_engine.load_model()
    lan_ip = get_local_lan_ip()
    logger.info("=" * 60)
    logger.info("🎤 Vaani Cloud ASR Server is READY")
    logger.info(f"👉 Localhost:    http://127.0.0.1:{config.PORT}")
    logger.info(f"👉 LAN / ESP32:  ws://{lan_ip}:{config.PORT}/ws/transcribe")
    logger.info(f"👉 Target Model: {config.MODEL_SIZE} ({asr_engine.device} / {asr_engine.compute_type})")
    logger.info("=" * 60)
    yield
    logger.info("Shutting down Vaani ASR Server...")


app = FastAPI(
    title="Vaani Cloud ASR Server",
    description="Low-latency streaming ASR server for ESP32 edge wake-word detection",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint showing server status, IP, and device info."""
    lan_ip = get_local_lan_ip()
    return {
        "status": "online",
        "service": "Vaani Cloud ASR",
        "lan_ip": lan_ip,
        "websocket_endpoint": f"ws://{lan_ip}:{config.PORT}/ws/transcribe",
        "model_size": config.MODEL_SIZE,
        "device": asr_engine.device if asr_engine else "not_loaded",
        "compute_type": asr_engine.compute_type if asr_engine else "not_loaded",
        "sample_rate": config.SAMPLE_RATE,
        "silence_timeout_sec": config.SILENCE_TIMEOUT_SEC,
    }


@app.post("/api/transcribe_wav")
async def transcribe_wav(file: UploadFile = File(...)) -> Dict[str, Any]:
    """HTTP endpoint to test transcription on uploaded 16kHz WAV files."""
    try:
        import soundfile as sf
        import io

        contents = await file.read()
        audio_data, sr = sf.read(io.BytesIO(contents), dtype="float32")

        if sr != config.SAMPLE_RATE:
            # Resample if necessary using simple interpolation
            duration = len(audio_data) / sr
            target_length = int(duration * config.SAMPLE_RATE)
            audio_data = np.interp(
                np.linspace(0, len(audio_data), target_length, endpoint=False),
                np.arange(len(audio_data)),
                audio_data,
            ).astype(np.float32)

        # Ensure mono
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        result = asr_engine.transcribe(audio_data)
        return {"status": "success", **result}
    except Exception as e:
        logger.error(f"Error in transcribe_wav: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """
    Primary WebSocket streaming endpoint for ESP32 / Mock client.
    Accepts:
      - Raw binary frames: 16-bit 16kHz mono PCM chunks (e.g., 512-1024 bytes per frame)
      - Text JSON frames: Control commands like {"action": "reset"} or {"action": "start"}
    Sends:
      - JSON transcription results upon silence detection or explicit stop.
    """
    await websocket.accept()
    client_ip = websocket.client.host if websocket.client else "unknown"
    logger.info(f"WebSocket client connected: {client_ip}")

    vad = StreamVAD()
    session_start_time = time.time()

    try:
        # Send initial handshake confirmation
        await websocket.send_text(
            json.dumps({
                "type": "ready",
                "message": "Connected to Vaani ASR Server. Send binary 16kHz PCM chunks.",
                "sample_rate": config.SAMPLE_RATE,
            })
        )

        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                logger.info(f"WebSocket client disconnected: {client_ip}")
                break

            if "bytes" in message and message["bytes"]:
                chunk = message["bytes"]
                state, is_complete = vad.feed(chunk)

                if is_complete:
                    audio_float32 = vad.get_audio_float32()
                    vad.reset()

                    # Immediately signal ESP32 to stop sending audio frames
                    await websocket.send_text(json.dumps({"type": "stop"}))

                    if len(audio_float32) > 0:
                        t_transcribe_start = time.time()
                        # Run transcribe in background worker thread so the event loop is NEVER blocked
                        result = await asyncio.to_thread(asr_engine.transcribe, audio_float32)

                        # Display prominent large banner directly in server terminal for the video
                        print_large_transcript_banner(
                            transcript=result["transcript"],
                            latency_ms=result["asr_latency_ms"],
                            duration_sec=result["audio_duration_sec"],
                            lang=result["language"],
                            confidence=result["language_probability"],
                        )

            elif "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                    action = payload.get("action", "")

                    if action == "reset":
                        vad.reset()
                        await websocket.send_text(
                            json.dumps({"type": "status", "message": "Buffer reset"})
                        )
                    elif action == "flush":
                        # Manually force transcription of current buffer
                        audio_float32 = vad.get_audio_float32()
                        vad.reset()

                        # Signal ESP32 that stream is finished
                        await websocket.send_text(json.dumps({"type": "stop"}))

                        if len(audio_float32) > 0:
                            result = await asyncio.to_thread(asr_engine.transcribe, audio_float32)
                            print_large_transcript_banner(
                                transcript=result["transcript"],
                                latency_ms=result["asr_latency_ms"],
                                duration_sec=result["audio_duration_sec"],
                                lang=result["language"],
                                confidence=result["language_probability"],
                            )
                    elif action == "ping":
                        await websocket.send_text(
                            json.dumps({"type": "pong", "time": time.time()})
                        )
                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {client_ip}")
    except Exception as e:
        logger.error(f"WebSocket session error with {client_ip}: {e}")
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=config.HOST,
        port=config.PORT,
        log_level="info",
        reload=False,
    )
