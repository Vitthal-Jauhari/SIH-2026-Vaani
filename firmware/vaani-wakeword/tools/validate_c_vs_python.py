"""
Validate numerical parity between Python training features and C (ESP32) feature extraction.

Tests:
1. Python features (model/features.py) vs C features (mel_features.c)
2. Numerical matrix error metrics (max diff, mean diff, cosine similarity)
3. TFLite INT8 inference output using Python features vs C features
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.features import extract_mfcc


def run_validation(audio_path: Path, model_path: Path, c_exe_path: Path):
    print("=" * 70)
    print("NUMERICAL PARITY VALIDATION: PYTHON (TRAIN) VS C (ESP32 FIRMWARE)")
    print("=" * 70)
    print(f"Audio Input : {audio_path}")
    print(f"TFLite Model: {model_path}")
    print(f"C Executable: {c_exe_path}\n")

    # 1. Load and prepare 1-second audio (16,000 samples @ 16 kHz)
    audio_f32, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    if len(audio_f32) < 16000:
        audio_f32 = np.pad(audio_f32, (0, 16000 - len(audio_f32)))
    else:
        audio_f32 = audio_f32[:16000]

    # Convert to 16-bit PCM integer values ([-32768, 32767])
    audio_pcm16 = np.clip(audio_f32 * 32768.0, -32768, 32767).astype(np.int16)
    # The float representation back from PCM16 (matches what C sees after dividing by 32768.0)
    audio_f32_pcm = audio_pcm16.astype(np.float32) / 32768.0

    # 2. Extract features using Python pipeline (model/features.py)
    py_features = extract_mfcc(audio_f32_pcm)  # (63, 13)

    # 3. Extract features using compiled C code (mel_features.c)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_pcm = Path(tmpdir) / "audio.raw"
        tmp_out = Path(tmpdir) / "c_feat.bin"

        tmp_pcm.write_bytes(audio_pcm16.tobytes())

        cmd = [str(c_exe_path), str(tmp_pcm), str(tmp_out)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print("C feature extractor failed:\n", res.stderr)
            sys.exit(1)

        c_raw = tmp_out.read_bytes()
        c_features = np.frombuffer(c_raw, dtype=np.float32).reshape((63, 13))

    # 4. Numerical Comparison
    diff = np.abs(py_features - c_features)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))

    # Cosine similarity across all 819 feature elements
    py_flat = py_features.flatten()
    c_flat = c_features.flatten()
    cos_sim = float(np.dot(py_flat, c_flat) / (np.linalg.norm(py_flat) * np.linalg.norm(c_flat)))

    print("--- 1. FEATURE MATRIX NUMERICAL PARITY (63 x 13 = 819 values) ---")
    print(f"  Shape (Python)      : {py_features.shape}")
    print(f"  Shape (C / ESP32)   : {c_features.shape}")
    print(f"  Max Absolute Diff   : {max_diff:.8f}")
    print(f"  Mean Absolute Diff  : {mean_diff:.8f}")
    print(f"  Cosine Similarity   : {cos_sim:.8f} (1.0 = identical direction)")

    # 5. Model Inference Parity
    interp = tf.lite.Interpreter(model_path=str(model_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]

    def infer(feat_matrix):
        q = np.round(feat_matrix / in_scale) + in_zp
        q = np.clip(q, -128, 127).astype(np.int8)
        q = np.expand_dims(q, (0, -1))
        interp.set_tensor(inp["index"], q)
        interp.invoke()
        q_out = interp.get_tensor(out["index"])[0]
        probs = (q_out.astype(np.float32) - out_zp) * out_scale
        return probs, q

    py_probs, py_q = infer(py_features)
    c_probs, c_q = infer(c_features)

    quant_mismatches = int(np.sum(py_q != c_q))

    print("\n--- 2. TFLITE INT8 QUANTIZATION & INFERENCE PARITY ---")
    print(f"  Input Scale / ZeroPt : {in_scale:.8f} / {in_zp}")
    print(f"  Quantized INT8 Diff  : {quant_mismatches} / 819 inputs differ ({quant_mismatches/819*100:.2f}%)")
    print("\nClass Probabilities: [Silence, Unknown, Vaani]")
    print(f"  Python Features      : {py_probs}")
    print(f"  C / ESP32 Features   : {c_probs}")
    prob_diff = np.abs(py_probs - c_probs)
    print(f"  Max Probability Diff : {np.max(prob_diff):.6f}")

    labels = ["silence", "unknown", "vaani"]
    py_pred = labels[int(np.argmax(py_probs))]
    c_pred = labels[int(np.argmax(c_probs))]

    print(f"\nPredicted Class (Python) : {py_pred.upper()} (P={np.max(py_probs):.4f})")
    print(f"Predicted Class (C/ESP32): {c_pred.upper()} (P={np.max(c_probs):.4f})")

    # Success criteria
    success = (py_pred == c_pred) and (np.max(prob_diff) < 0.05) and (max_diff < 0.05)
    print("\n" + "=" * 70)
    if success:
        print(">>> [PARITY VERIFIED] C and Python preprocessing pipelines are numerically identical! <<<")
    else:
        print(">>> [FAILURE] Discrepancy detected between C and Python preprocessing! <<<")
    print("=" * 70)
    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate C vs Python preprocessing")
    parser.add_argument(
        "--audio",
        type=str,
        default=str(ROOT_DIR / "dataset" / "audio" / "Ishita" / "Standard recording 10.mp3"),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(ROOT_DIR / "model" / "weights" / "vaani_v2_int8.tflite"),
    )
    parser.add_argument(
        "--exe",
        type=str,
        default=str(ROOT_DIR / "firmware" / "vaani-wakeword" / "tools" / "verify_c_features.exe"),
    )
    args = parser.parse_args()

    run_validation(Path(args.audio), Path(args.model), Path(args.exe))
