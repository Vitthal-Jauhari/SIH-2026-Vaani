"""
Mock ESP32 Client for Vaani ASR Server.
Simulates an ESP32 edge device streaming 16kHz 16-bit PCM audio chunks over WebSocket.

Modes:
  1. Live Mic Mode: Speak into PC microphone in real-time.
  2. WAV File Mode: Stream an existing audio clip in simulated 20ms packets.

Usage:
  # Live microphone streaming
  python mock_esp32_client.py --mic

  # Stream a WAV file
  python mock_esp32_client.py --wav path/to/sample.wav

  # Specify custom server URL
  python mock_esp32_client.py --mic --url ws://127.0.0.1:8000/ws/transcribe
"""

import argparse
import asyncio
import json
import time
from pathlib import Path
import numpy as np
import websockets

SAMPLE_RATE = 16000
CHUNK_MS = 20  # 20ms packets (typical for ESP32 I2S DMA hop)
CHUNK_SAMPLES = int(SAMPLE_RATE * (CHUNK_MS / 1000.0))  # 320 samples
CHUNK_BYTES = CHUNK_SAMPLES * 2  # 640 bytes (16-bit signed PCM)


async def stream_wav_file(ws_url: str, wav_path: Path):
    """Streams a WAV audio file to the ASR server in real-time chunks."""
    import soundfile as sf

    print(f"Reading WAV file: {wav_path}")
    data, sr = sf.read(str(wav_path), dtype="int16")

    # Resample if needed
    if sr != SAMPLE_RATE:
        print(f"Resampling from {sr} Hz to {SAMPLE_RATE} Hz...")
        duration = len(data) / sr
        target_len = int(duration * SAMPLE_RATE)
        data = np.interp(
            np.linspace(0, len(data), target_len, endpoint=False),
            np.arange(len(data)),
            data,
        ).astype(np.int16)

    # Convert to mono
    if data.ndim > 1:
        data = np.mean(data, axis=1).astype(np.int16)

    raw_bytes = data.tobytes()
    total_chunks = len(raw_bytes) // CHUNK_BYTES

    print(f"Connecting to {ws_url}...")
    async with websockets.connect(ws_url) as ws:
        # Await server welcome message
        welcome = await ws.recv()
        print(f"Server Handshake: {welcome}")

        print(f"Streaming {total_chunks} audio chunks (~{len(data)/SAMPLE_RATE:.1f}s)...")
        t_start = time.time()

        for i in range(0, len(raw_bytes), CHUNK_BYTES):
            chunk = raw_bytes[i : i + CHUNK_BYTES]
            await ws.send(chunk)
            await asyncio.sleep(CHUNK_MS / 1000.0)  # Real-time simulation

        # Send 1 second of trailing silence to trigger VAD
        silence_chunk = (np.zeros(CHUNK_SAMPLES, dtype=np.int16)).tobytes()
        for _ in range(int(1000 / CHUNK_MS)):
            await ws.send(silence_chunk)
            await asyncio.sleep(CHUNK_MS / 1000.0)

        # Await response
        print("Waiting for transcription response...")
        resp = await ws.recv()
        t_total = (time.time() - t_start) * 1000
        print("\n" + "=" * 50)
        print("🎉 TRANSCRIPTION RESULT:")
        print(json.dumps(json.loads(resp), indent=2, ensure_ascii=False))
        print(f"⏱️ Total test duration: {t_total:.1f} ms")
        print("=" * 50)


async def stream_live_mic(ws_url: str):
    """Streams live microphone input from the laptop to the ASR server in real-time."""
    import sounddevice as sd

    print(f"Connecting to {ws_url}...")
    async with websockets.connect(ws_url) as ws:
        welcome = await ws.recv()
        print(f"Server Handshake: {welcome}")
        print("\n🎤 LIVE STREAMING ACTIVE")
        print("Speak a Hinglish command (e.g. 'Vaani, bedroom ki light on karo')...")
        print("Press Ctrl+C to stop.\n")

        loop = asyncio.get_running_loop()
        audio_queue = asyncio.Queue()

        def mic_callback(indata, frames, time_info, status):
            if status:
                print(f"Audio status: {status}")
            # indata is float32 [-1.0, 1.0], convert to int16 bytes
            int16_data = (indata[:, 0] * 32767).clip(-32768, 32767).astype(np.int16)
            loop.call_soon_threadsafe(audio_queue.put_nowait, int16_data.tobytes())

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            callback=mic_callback,
        )

        async def sender():
            with stream:
                while True:
                    chunk = await audio_queue.get()
                    await ws.send(chunk)

        async def receiver():
            while True:
                msg = await ws.recv()
                try:
                    payload = json.loads(msg)
                    if payload.get("type") == "transcription":
                        transcript = payload.get("transcript", "")
                        latency = payload.get("asr_latency_ms", 0)
                        dur = payload.get("audio_duration_sec", 0)
                        print("-" * 50)
                        print(f"🎯 Transcript: {transcript}")
                        print(f"⏱️ ASR Latency: {latency:.1f} ms | Audio: {dur:.2f} s")
                        print("-" * 50)
                    else:
                        print(f"Server status: {msg}")
                except Exception:
                    print(f"Raw message: {msg}")

        await asyncio.gather(sender(), receiver())


def main():
    parser = argparse.ArgumentParser(description="Mock ESP32 Streaming Client")
    parser.add_argument(
        "--url",
        type=str,
        default="ws://127.0.0.1:8000/ws/transcribe",
        help="WebSocket URL of Vaani ASR Server",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--mic", action="store_true", help="Stream live PC microphone")
    group.add_argument("--wav", type=Path, help="Path to WAV file to stream")

    args = parser.parse_args()

    if args.mic:
        try:
            asyncio.run(stream_live_mic(args.url))
        except KeyboardInterrupt:
            print("\nStopped live microphone streaming.")
    elif args.wav:
        if not args.wav.exists():
            print(f"Error: File not found: {args.wav}")
            return
        asyncio.run(stream_wav_file(args.url, args.wav))


if __name__ == "__main__":
    main()
