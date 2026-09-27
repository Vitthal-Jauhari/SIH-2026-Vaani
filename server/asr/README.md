# ☁️ Vaani Cloud ASR Server (Hinglish)

Low-latency, open-source streaming Automated Speech Recognition (ASR) server for the **Vaani** hybrid edge-cloud Keyword Spotting system.

Designed to fulfill the **SIH 2026** problem statement:
- **Zero Proprietary SDKs:** Powered 100% by open-source [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2).
- **Hinglish Optimization:** Decoder primed with custom prompts for seamless English + Hindi code-switched voice commands.
- **Hardware Acceleration:** Native NVIDIA GPU (CUDA / TensorRT / FP16) acceleration on laptop (RTX 3050) with transparent CPU (INT8) fallback.
- **Sub-100ms Inference Latency:** Transcribes 2-3 second voice commands in **~50–100 ms** on GPU.

---

## 📡 WebSocket Streaming Protocol

The server exposes a persistent WebSocket endpoint at `ws://<SERVER_IP>:8000/ws/transcribe`.

### 1. Connection & Handshake
The edge device (or test client) opens a persistent WebSocket connection:
```json
// Server responds immediately upon connect:
{
  "type": "ready",
  "message": "Connected to Vaani ASR Server. Send binary 16kHz PCM chunks.",
  "sample_rate": 16000
}
```

### 2. Audio Streaming Format
- **Format:** Raw 16-bit signed PCM, 16,000 Hz, Mono (matching INMP441 capture).
- **Chunk Size:** 320 to 512 samples (20–32 ms per packet) sent as raw binary WebSocket frames.
- **Bandwidth:** Only 32 KB/sec (256 kbps) during active speech.

### 3. Automatic Speech Segmentation
The server runs an internal energy-based VAD (`StreamVAD`):
1. Detects onset of user's voice command.
2. Buffers streaming packets in memory.
3. Automatically triggers transcription when trailing silence threshold is reached (default: 800 ms).

### 4. Server Response
```json
{
  "type": "transcription",
  "status": "success",
  "transcript": "bedroom ki lights on karo",
  "language": "hi",
  "confidence": 0.96,
  "audio_duration_sec": 2.15,
  "asr_latency_ms": 74.2,
  "turnaround_ms": 82.5
}
```
*Note: The WebSocket remains open after transcription so consecutive commands require no reconnection.*

---

## 🚀 Quick Start (Running on Laptop)

### 1. Install Dependencies

Use Python 3.11:
```bash
# From repository root
cd server/asr
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Launch Server

```bash
python app.py
```
Upon startup, the server automatically prints:
```text
🎤 Vaani Cloud ASR Server is READY
👉 Localhost:    http://127.0.0.1:8000
👉 LAN / ESP32:  ws://192.168.1.X:8000/ws/transcribe
👉 Target Model: base (cuda / float16)
```

---

## 🧪 Testing with Mock ESP32 Client

Before testing on the physical ESP32 microcontroller, test the streaming pipeline using your laptop's microphone or audio files:

### Live Microphone Test (Speak Hinglish commands)
```bash
python mock_esp32_client.py --mic
```
*Speak into your mic: "Vaani, turn on the fan" or "Vaani, light band kar do". Observe the real-time transcription and latency.*

### WAV File Stream Test
```bash
python mock_esp32_client.py --wav path/to/sample.wav
```

---

## ⚙️ Configuration Options

Environment variables can be configured in `.env` or set prior to launch:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `VAANI_ASR_HOST` | `0.0.0.0` | Bind host (all network interfaces for LAN access) |
| `VAANI_ASR_PORT` | `8000` | Port number |
| `VAANI_MODEL_SIZE` | `base` | Model: `tiny`, `base`, `small`, or `medium` |
| `VAANI_DEVICE` | `auto` | `cuda` (NVIDIA GPU) or `cpu` |
| `VAANI_COMPUTE_TYPE` | `auto` | `float16` for CUDA, `int8` for CPU |
| `VAANI_SILENCE_TIMEOUT`| `0.8` | Trailing silence in seconds to cut off utterance |
| `VAANI_ENERGY_TH` | `0.015` | Normalized RMS threshold for speech activity |
