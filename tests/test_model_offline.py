"""
test_model_offline.py — isolate the V2 wakeword model from the ESP32 firmware.

Runs vaani_dscnn_v2_int8.tflite directly on a WAV file on your PC, replicating
the exact same feature pipeline as mel_features.c (63x13 MFCC, librosa-style,
512-pt FFT, hop 256, 128 mel bins, top_db=80, ortho DCT-II) and the exact same
INT8 quantize/dequantize steps as kws_model.cpp.

WHY THIS TEST MATTERS
----------------------
The firmware showed high-confidence "VAANI DETECTED" results during true
silence. Two competing explanations:
  (A) The model doesn't generalize well to this mic's real noise floor
      (a training/dataset problem — needs more/better negative data).
  (B) The on-device C implementation has a bug (mic gain, feature
      extraction, or quantization) that's feeding the model garbage.

This script tests (A) directly: if this exact model, given real silence
audio and NORMAL scaling, still gives high P(Vaani), that points at (A).
If it stays low here but the firmware still misfires, that points at (B).

It also includes a --emulate-gain-bug flag that deliberately multiplies
the signal by 4x (equivalent to the suspected `>> 14` vs `>> 16` shift bug
in i2s_mic.c) so you can test that specific hypothesis in isolation too,
without needing to reflash anything.

USAGE
-----
    pip install tensorflow librosa soundfile numpy --break-system-packages
    # (or: pip install tflite-runtime librosa soundfile numpy)

    python test_model_offline.py --wav mic_test.wav
    python test_model_offline.py --wav silence_test.wav
    python test_model_offline.py --wav silence_test.wav --emulate-gain-bug
"""

import argparse
import os
import sys
import warnings

# Suppress TensorFlow C++ and Python warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np

try:
    import librosa
except ImportError:
    sys.exit("Missing dependency: pip install librosa soundfile --break-system-packages")

try:
    from tensorflow.lite.python.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        sys.exit("Missing dependency: pip install tensorflow --break-system-packages "
                 "(or: pip install tflite-runtime)")

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 16000      # 1 second, matches AUDIO_BUF_SAMPLES
HOP_SAMPLES = 3200          # 200 ms, matches KWS_HOP_SAMPLES
N_FFT = 512
HOP_LENGTH = 256
N_MELS = 40
N_MFCC = 13
TOP_DB = 80.0

CLASS_NAMES = ["Silence", "Unknown", "Vaani"]


def find_default_model() -> str | None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidates = [
        os.path.join(root, "model", "weights", "vaani_v2_int8.tflite"),
        os.path.join(root, "phase5", "artifacts", "vaani_v2_int8.tflite"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return os.path.abspath(p)
    return None


def extract_features(audio_f32: np.ndarray) -> np.ndarray:
    """Replicates mel_features.c exactly: librosa MFCC with center=True,
    top_db=80 power-to-db clamping, ortho DCT-II. Output shape (63, 13)."""
    mfcc = librosa.feature.mfcc(
        y=audio_f32,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        center=True,
        power=2.0,
    )
    # librosa's default power_to_db uses top_db=80.0, ref=1.0 already —
    # matches the on-device clamp. mfcc shape is (n_mfcc, n_frames); the
    # device wants [frame][mfcc], so transpose.
    return mfcc.T.astype(np.float32)  # (frames, 13)


def quantize_int8(features: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    q = np.round(features / scale) + zero_point
    return np.clip(q, -128, 127).astype(np.int8)


def dequantize(value: int, scale: float, zero_point: int) -> float:
    return (int(value) - zero_point) * scale


def run_model(interpreter: Interpreter, features: np.ndarray) -> tuple[float, float, float]:
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    in_scale, in_zero_point = input_details["quantization"]
    out_scale, out_zero_point = output_details["quantization"]

    q_input = quantize_int8(features, in_scale, in_zero_point)
    q_input = q_input.reshape(input_details["shape"])  # (1, 63, 13, 1)

    interpreter.set_tensor(input_details["index"], q_input)
    interpreter.invoke()
    q_output = interpreter.get_tensor(output_details["index"])[0]

    p_silence = dequantize(q_output[0], out_scale, out_zero_point)
    p_unknown = dequantize(q_output[1], out_scale, out_zero_point)
    p_vaani = dequantize(q_output[2], out_scale, out_zero_point)
    return p_silence, p_unknown, p_vaani


def main():
    default_model = find_default_model()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model",
        default=default_model,
        help=f"Path to vaani_dscnn_v2_int8.tflite (default: {default_model})",
    )
    parser.add_argument("--wav", required=True, help="Path to a WAV file to test against")
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument(
        "--emulate-gain-bug",
        action="store_true",
        help="Multiply signal by 4x before feature extraction, emulating the "
             "suspected `>> 14` (should be `>> 16`) shift bug in i2s_mic.c",
    )
    args = parser.parse_args()

    if not args.model or not os.path.isfile(args.model):
        sys.exit(f"Error: Model file not found at '{args.model}'. Please specify with --model.")

    wav_path = args.wav
    if not os.path.isfile(wav_path):
        # Fallback to parent directory check
        parent_candidate = os.path.join("..", wav_path)
        if os.path.isfile(parent_candidate):
            wav_path = parent_candidate
        else:
            sys.exit(f"Error: WAV file not found at '{args.wav}'.")

    print(f"Loading WAV:   {os.path.abspath(wav_path)}")
    print(f"Loading Model: {os.path.abspath(args.model)}")

    audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    if sr != SAMPLE_RATE:
        sys.exit(f"Unexpected sample rate {sr}, expected {SAMPLE_RATE}")

    if args.emulate_gain_bug:
        print(">>> Emulating 4x gain (the suspected mic scaling bug) <<<\n")
        audio = np.clip(audio * 4.0, -1.0, 1.0)

    interpreter = Interpreter(model_path=args.model)
    interpreter.allocate_tensors()

    n_windows = max(1, (len(audio) - WINDOW_SAMPLES) // HOP_SAMPLES + 1)
    detections = 0
    vaani_scores = []

    print(f"\n{'t(s)':>6}  {'Silence':>8}  {'Unknown':>8}  {'Vaani':>8}  {'dBFS':>7}  flag")
    print("-" * 55)

    for i in range(n_windows):
        start = i * HOP_SAMPLES
        window = audio[start:start + WINDOW_SAMPLES]
        if len(window) < WINDOW_SAMPLES:
            window = np.pad(window, (0, WINDOW_SAMPLES - len(window)))

        rms = np.sqrt(np.mean(window.astype(np.float64) ** 2))
        dbfs = 20 * np.log10(max(rms, 1e-10))

        features = extract_features(window)
        p_silence, p_unknown, p_vaani = run_model(interpreter, features)
        vaani_scores.append(p_vaani)

        flag = ""
        if p_vaani >= args.threshold:
            detections += 1
            flag = "*** DETECTED ***"

        t = start / SAMPLE_RATE
        print(f"{t:6.2f}  {p_silence:8.4f}  {p_unknown:8.4f}  {p_vaani:8.4f}  {dbfs:7.1f}  {flag}")

    vaani_scores = np.array(vaani_scores)
    print("\n" + "=" * 55)
    print(f"Windows tested       : {n_windows}")
    print(f"Detections (>= {args.threshold:.2f}): {detections}  ({100*detections/n_windows:.1f}%)")
    print(f"P(Vaani) mean/max    : {vaani_scores.mean():.4f} / {vaani_scores.max():.4f}")
    print("=" * 55)

    if detections > 0:
        print(
            "\nModel fired on this audio even offline, with the exact same "
            "feature pipeline as the device. This points toward (A): the "
            "model/training data, not just the firmware, needs attention — "
            "especially if this happens on plain silence with no gain "
            "emulation applied."
        )
    else:
        print(
            "\nModel stayed quiet offline on this audio. If the firmware "
            "still misfires on the same physical audio, the bug is most "
            "likely in the C implementation (mic gain, feature extraction, "
            "or quantization) rather than the model itself."
        )


if __name__ == "__main__":
    main()
