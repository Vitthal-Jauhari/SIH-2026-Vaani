"""
FastAPI WebSocket and REST Server for Vaani Real-Time Speech Recognition.
Provides low-latency binary audio streaming endpoint for ESP32 edge devices.
"""

import json
import logging
import socket
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import config
from engine import ASREngine
from vad_stream import StreamVAD, SpeechState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vaani_asr.server")

# Global ASR engine instance
asr_engine: ASREngine = None


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
                    t_transcribe_start = time.time()

                    logger.info(
                        f"Utterance complete ({vad.duration_sec:.2f}s). Transcribing..."
                    )
                    result = asr_engine.transcribe(audio_float32)

                    total_turnaround_ms = (time.time() - t_transcribe_start) * 1000

                    response = {
                        "type": "transcription",
                        "status": "success",
                        "transcript": result["transcript"],
                        "language": result["language"],
                        "confidence": result["language_probability"],
                        "audio_duration_sec": result["audio_duration_sec"],
                        "asr_latency_ms": result["asr_latency_ms"],
                        "turnaround_ms": round(total_turnaround_ms, 2),
                    }

                    logger.info(
                        f"🎯 Transcript: '{result['transcript']}' "
                        f"({result['asr_latency_ms']:.1f}ms ASR, {result['audio_duration_sec']}s audio)"
                    )
                    await websocket.send_text(json.dumps(response))

                    # Reset stream buffer to accept the next utterance seamlessly
                    vad.reset()

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
                        if len(audio_float32) > 0:
                            result = asr_engine.transcribe(audio_float32)
                            await websocket.send_text(
                                json.dumps({
                                    "type": "transcription",
                                    "status": "success",
                                    **result,
                                })
                            )
                        else:
                            await websocket.send_text(
                                json.dumps({
                                    "type": "transcription",
                                    "status": "empty",
                                    "transcript": "",
                                    "audio_duration_sec": 0.0,
                                    "asr_latency_ms": 0.0,
                                })
                            )
                        vad.reset()
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
