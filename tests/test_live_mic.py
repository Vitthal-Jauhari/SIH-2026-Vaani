"""
Live Microphone Wake-Word Tester for Vaani (Model V2 & V1)

Real-time streaming inference using your PC microphone to test live wake-word
detection ("Vaani") and near-miss rejection ("Paani", "Rani", conversational speech).

Usage:
    python test_live_mic.py                     # Runs Model V2 (Default)
    python test_live_mic.py --model v1          # Runs Model V1 (Phase 3 Baseline)
    python test_live_mic.py --threshold 0.45    # Custom threshold (default: 0.45)
    python test_live_mic.py --list-devices      # List available microphones
    python test_live_mic.py --device 1          # Select specific mic device ID
"""

import argparse
import queue
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.features import extract_mfcc
from model.architecture import LABELS

SAMPLE_RATE = 16000
WINDOW_SECONDS = 1.0
BUFFER_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)
VAANI_IDX = LABELS.index("vaani")
UNKNOWN_IDX = LABELS.index("unknown")
SILENCE_IDX = LABELS.index("silence")


class LiveWakeWordEngine:
    def __init__(self, model_path: Path):
        self.path = Path(model_path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"TFLite model not found at: {self.path}")

        self.interpreter = tf.lite.Interpreter(model_path=str(self.path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

        self.in_scale, self.in_zp = self.input_details["quantization"]
        self.out_scale, self.out_zp = self.output_details["quantization"]
        self.is_quantized = self.input_details["dtype"] == np.int8

    def predict(self, window_audio: np.ndarray):
        t0 = time.perf_counter()
        mfcc = extract_mfcc(window_audio)  # (63, 13)
        inp = np.expand_dims(mfcc, (0, -1))  # (1, 63, 13, 1)

        if self.is_quantized:
            inp = np.round(inp / self.in_scale + self.in_zp).astype(np.int8)

        self.interpreter.set_tensor(self.input_details["index"], inp)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details["index"])[0]

        if self.is_quantized:
            probs = (out.astype(np.float32) - self.out_zp) * self.out_scale
        else:
            probs = out.astype(np.float32)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return probs, latency_ms


def draw_bar(val: float, length: int = 15) -> str:
    filled = int(round(val * length))
    return "█" * filled + "░" * (length - filled)


def main():
    parser = argparse.ArgumentParser(description="Live Microphone Wake-Word Tester for Vaani")
    parser.add_argument(
        "--model",
        type=str,
        default="v2",
        choices=["v1", "v2"],
        help="Model version to test: 'v2' (Phase 5, default) or 'v1' (Phase 3 baseline)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Optional custom path to a .tflite model file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.60,
        help="Confidence threshold for 'Vaani' detection (default: 0.60)",
    )
    parser.add_argument(
        "--consecutive",
        type=int,
        default=2,
        help="Number of consecutive positive windows required to confirm wake word (default: 2)",
    )
    parser.add_argument(
        "--min_db",
        type=float,
        default=-45.0,
        help="Minimum audio energy (RMS in dB) to trigger speech processing (default: -45.0 dB)",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=0.8,
        help="Debounce cooldown in seconds to prevent repeat triggers on the same word (default: 0.8s)",
    )
    parser.add_argument(
        "--hop_ms",
        type=int,
        default=100,
        help="Sliding window hop step in milliseconds (default: 100 ms)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Microphone input device index (run with --list-devices to view options)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print list of available audio input devices and exit",
    )
    args = parser.parse_args()

    if args.list_devices:
        print("\nAvailable Audio Input Devices:")
        print("=" * 60)
        devices = sd.query_devices()
        for idx, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                print(f"[{idx:2d}] {dev['name']} (Channels: {dev['max_input_channels']})")
        print("=" * 60)
        sys.exit(0)

    # Determine model path
    if args.model_path:
        tflite_path = Path(args.model_path)
    else:
        v2_path = ROOT_DIR / "model" / "weights" / "vaani_v2_int8.tflite"
        v1_path = ROOT_DIR / "model" / "weights" / "vaani_v1_int8.tflite"
        if args.model.lower() == "v1" and v1_path.exists():
            tflite_path = v1_path
        else:
            tflite_path = v2_path

    print("\n" + "=" * 75)
    print("🎤 VAANI LIVE MICROPHONE WAKE-WORD TESTER")
    print("=" * 75)
    print(f"Model Version     : {args.model.upper()}")
    print(f"Model File        : {tflite_path}")
    print(f"Trigger Threshold : {args.threshold:.2f}")
    print(f"Window / Hop      : 1.0s window / {args.hop_ms}ms hop")
    print(f"Cooldown Period   : {args.cooldown:.2f}s")

    # Initialize Engine
    engine = LiveWakeWordEngine(tflite_path)

    # Audio Buffer & Queue
    hop_samples = int(SAMPLE_RATE * (args.hop_ms / 1000.0))
    audio_buffer = deque(maxlen=BUFFER_SAMPLES)
    audio_buffer.extend(np.zeros(BUFFER_SAMPLES, dtype=np.float32))
    audio_q = queue.Queue()

    def mic_callback(indata, frames, time_info, status):
        if status:
            print(f"[Audio Stream Warning] {status}", file=sys.stderr)
        audio_q.put(indata[:, 0].copy())

    device_info = (
        sd.query_devices(args.device, "input")
        if args.device is not None
        else sd.query_devices(kind="input")
    )
    print(f"Active Microphone : {device_info['name']}")
    print("=" * 75)
    print("STATUS: 🟢 LISTENING... Speak 'Vaani' or near-misses ('Paani', 'Rani', speech).")
    print("Press Ctrl+C at any time to exit.\n")

    detections = 0
    start_time = time.time()
    cooldown_until = 0.0
    accumulated_samples = 0
    consecutive_hits = 0

    try:
        with sd.InputStream(
            device=args.device,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=hop_samples,
            callback=mic_callback,
        ):
            while True:
                chunk = audio_q.get()
                audio_buffer.extend(chunk)
                accumulated_samples += len(chunk)

                if accumulated_samples >= hop_samples:
                    accumulated_samples = 0
                    window_audio = np.array(audio_buffer, dtype=np.float32)

                    # Compute RMS for visual volume meter
                    rms = np.sqrt(np.mean(window_audio**2))
                    rms_db = 20 * np.log10(max(rms, 1e-5))

                    # Energy Gate (Voice Activity Detection):
                    # Real speech is between -35 dB and -10 dB.
                    # Ambient silence / Bluetooth muted mic is < -45 dB.
                    is_speech = rms_db >= args.min_db

                    if not is_speech:
                        p_silence = 1.0
                        p_unknown = 0.0
                        p_vaani = 0.0
                        latency = 0.1
                        status_str = "💤 SILENCE"
                        consecutive_hits = 0
                    else:
                        probs, latency = engine.predict(window_audio)
                        p_silence = float(probs[SILENCE_IDX])
                        p_unknown = float(probs[UNKNOWN_IDX])
                        p_vaani = float(probs[VAANI_IDX])

                        # Must dominate over unknown
                        is_candidate = (
                            p_vaani >= args.threshold
                            and p_vaani > p_unknown
                            and (p_vaani - p_unknown) >= 0.10
                        )

                        if is_candidate:
                            consecutive_hits += 1
                            status_str = f"🟡 DETECTING ({consecutive_hits}/{args.consecutive})"
                        else:
                            consecutive_hits = 0
                            if p_unknown >= 0.50:
                                status_str = f"🔵 REJECTED: UNKNOWN ({p_unknown*100:.0f}%)"
                            else:
                                status_str = "🟢 LISTENING"

                    now = time.time()

                    # Trigger condition: requires speech, threshold dominance, and consecutive window confirmation
                    if is_speech and consecutive_hits >= args.consecutive and now >= cooldown_until:
                        detections += 1
                        cooldown_until = now + args.cooldown
                        consecutive_hits = 0
                        elapsed = now - start_time
                        print("\n" + "=" * 70)
                        print(
                            f"🎯 [TRIGGER #{detections}] WAKE WORD CONFIRMED: 'VAANI'!\n"
                            f"   Confidence  : {p_vaani*100:.1f}% (Unknown: {p_unknown*100:.1f}%)\n"
                            f"   Volume      : {rms_db:.1f} dB\n"
                            f"   Consistency : {args.consecutive} consecutive windows\n"
                            f"   Inference   : {latency:.1f} ms\n"
                            f"   Time        : {elapsed:.1f}s after start"
                        )
                        print("=" * 70 + "\n")
                    else:
                        # Visual meters
                        v_bar = draw_bar(p_vaani, length=10)
                        u_bar = draw_bar(p_unknown, length=10)
                        vol_meter = draw_bar(min(1.0, max(0.0, (rms_db + 60) / 45)), length=8)

                        sys.stdout.write(
                            f"\r[Vol: {vol_meter} {rms_db:4.0f}dB]  "
                            f"Vaani: [{v_bar}] {p_vaani*100:4.1f}%  "
                            f"Unknown: [{u_bar}] {p_unknown*100:4.1f}%  "
                            f"{status_str}         "
                        )
                        sys.stdout.flush()

    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n\nSession stopped.")
        print(f"Total Detections : {detections}")
        print(f"Session Duration : {elapsed:.1f} seconds")
        print("Done!")


if __name__ == "__main__":
    main()
