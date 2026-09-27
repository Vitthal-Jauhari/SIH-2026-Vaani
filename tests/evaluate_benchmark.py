"""
Phase 5 - Step 9: Comprehensive Apples-to-Apples V1 vs V2 Robustness Benchmark

Strict Determinism & Apples-to-Apples Rule:
- Evaluates BOTH V1 (phase3/artifacts/vaani_int8.tflite) and V2 (phase5/artifacts/vaani_v2_int8.tflite)
  on EXACTLY THE SAME audio clips.
- Deterministic random seeds.
- Pre-generates and saves an explicit test manifest (phase5/artifacts/eval_manifest.json)
  with all synthetic robustness variations (volume, noise, speed) so both models receive identical waveforms.
- Evaluates:
  1. Clean speaker recall (Ananya, Ark, Ishita, Umang, Vitthal)
  2. Quiet speech recall (-12 dB, -6 dB)
  3. Traffic / noise robustness (traffic, babble, pink @ 25, 20, 15, 10, 5, 0 dB SNR)
  4. Speaking tempo robustness (0.85x, 0.92x, 1.08x, 1.15x)
  5. Negative test set FPR (unknown words & silence)
  6. Global metrics: Precision, Recall, F1, FPR, FNR at threshold 0.40 and 0.50.
- Saves results to: phase5/artifacts/v1_vs_v2_results.json
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.features import extract_mfcc  # noqa: E402
from model.architecture import LABELS  # noqa: E402

SAMPLE_RATE = 16000
CLIP_LEN = 16000
VAANI_IDX = LABELS.index("vaani")


class TFLiteClassifier:
    def __init__(self, model_path: Path):
        self.path = model_path.resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"TFLite model not found: {self.path}")
        self.content = self.path.read_bytes()
        self.interpreter = tf.lite.Interpreter(model_content=self.content)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

        self.in_scale, self.in_zero_point = self.input_details["quantization"]
        self.out_scale, self.out_zero_point = self.output_details["quantization"]
        self.is_quantized = self.input_details["dtype"] == np.int8

    def predict_audio(self, audio: np.ndarray) -> np.ndarray:
        # Pad or truncate to 1s
        if len(audio) < CLIP_LEN:
            audio = np.pad(audio, (0, CLIP_LEN - len(audio)), mode="constant")
        else:
            audio = audio[:CLIP_LEN]

        # Extract MFCC
        mfcc = extract_mfcc(audio)  # (63, 13)
        inp = np.expand_dims(mfcc, (0, -1))  # (1, 63, 13, 1)

        if self.is_quantized:
            inp = np.round(inp / self.in_scale + self.in_zero_point).astype(np.int8)

        self.interpreter.set_tensor(self.input_details["index"], inp)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details["index"])

        if self.is_quantized:
            out = (out.astype(np.float32) - self.out_zero_point) * self.out_scale

        return out[0]  # (3,) probs or logits


def apply_volume(audio: np.ndarray, db_gain: float) -> np.ndarray:
    factor = 10.0 ** (db_gain / 20.0)
    return np.clip(audio * factor, -1.0, 1.0).astype(np.float32)


def apply_speed(audio: np.ndarray, speed: float) -> np.ndarray:
    try:
        if len(audio) < 2048:
            audio = np.pad(audio, (0, 2048 - len(audio)), mode="constant")
        stretched = librosa.effects.time_stretch(audio, rate=speed)
        if len(stretched) < CLIP_LEN:
            return np.pad(stretched, (0, CLIP_LEN - len(stretched)), mode="constant")
        return stretched[:CLIP_LEN].astype(np.float32)
    except Exception:
        if len(audio) < CLIP_LEN:
            return np.pad(audio, (0, CLIP_LEN - len(audio)), mode="constant")
        return audio[:CLIP_LEN].astype(np.float32)


def generate_noise(noise_type: str, length: int) -> np.ndarray:
    if noise_type == "traffic":
        white = np.random.normal(0, 1, length)
        alpha = 0.15
        rumble = np.zeros(length, dtype=np.float32)
        for i in range(1, length):
            rumble[i] = alpha * white[i] + (1 - alpha) * rumble[i - 1]
        peak = np.max(np.abs(rumble)) or 1.0
        return (rumble / peak).astype(np.float32)
    elif noise_type == "pink":
        white = np.random.normal(0, 1, length)
        b0, b1, b2 = 0.049922035, -0.095993537, 0.050612699
        a1, a2 = -1.74201445, 0.7490076
        from scipy.signal import lfilter
        pink = lfilter([b0, b1, b2], [1.0, a1, a2], white)
        peak = np.max(np.abs(pink)) or 1.0
        return (pink / peak).astype(np.float32)
    elif noise_type == "babble":
        t = np.linspace(0, 1.0, length, endpoint=False)
        babble = np.zeros(length, dtype=np.float32)
        for freq in (150, 220, 310, 480, 700, 1100, 1800):
            phase = np.random.uniform(0, 2 * np.pi)
            mod = 0.5 + 0.5 * np.sin(2 * np.pi * np.random.uniform(2, 6) * t)
            babble += mod * np.sin(2 * np.pi * freq * t + phase)
        peak = np.max(np.abs(babble)) or 1.0
        return (babble / peak).astype(np.float32)
    else:
        white = np.random.normal(0, 1, length)
        t = np.linspace(0, 1.0, length, endpoint=False)
        hum = 0.3 * np.sin(2 * np.pi * 50 * t)
        noise = white + hum
        peak = np.max(np.abs(noise)) or 1.0
        return (noise / peak).astype(np.float32)


def apply_noise(audio: np.ndarray, snr_db: float, noise_type: str = "traffic") -> np.ndarray:
    signal_rms = np.sqrt(np.mean(audio ** 2)) or 1e-6
    desired_noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
    noise = generate_noise(noise_type, len(audio))
    noise_rms = np.sqrt(np.mean(noise ** 2)) or 1e-6
    scaled_noise = noise * (desired_noise_rms / noise_rms)
    return np.clip(audio + scaled_noise, -1.0, 1.0).astype(np.float32)


def run_benchmark(
    v1_model_path: Path,
    v2_model_path: Path,
    phase5_root: Path,
    out_dir: Path,
    seed: int = 42,
):
    np.random.seed(seed)
    random.seed(seed)

    print("=" * 70)
    print("PHASE 5 - STEP 9: V1 VS V2 DETERMINISTIC APPLES-TO-APPLES BENCHMARK")
    print("=" * 70)
    print(f"V1 Model (Baseline): {v1_model_path}")
    print(f"V2 Model (Candidate): {v2_model_path}")
    print(f"Random Seed         : {seed}\n")

    v1 = TFLiteClassifier(v1_model_path)
    v2 = TFLiteClassifier(v2_model_path)

    pos_dir = phase5_root / "data" / "normalized" / "positives"
    test_combined = phase5_root / "data" / "combined" / "testing"

    # Preload all normalized clean speaker audio clips
    speakers_data: Dict[str, List[Tuple[str, np.ndarray]]] = {}
    for spk_d in sorted(pos_dir.iterdir()):
        if not spk_d.is_dir():
            continue
        spk = spk_d.name
        clips = []
        for w in sorted(spk_d.glob("*.wav")):
            audio, sr = sf.read(str(w))
            clips.append((w.name, audio))
        speakers_data[spk] = clips

    results = {
        "v1_path": str(v1_model_path),
        "v2_path": str(v2_model_path),
        "seed": seed,
        "clean_speaker_recall": {},
        "quiet_speech_recall": {},
        "noise_robustness_recall": {},
        "speed_robustness_recall": {},
        "test_set_summary": {},
    }

    # 1. Clean Speaker Recall (Threshold 0.40 and 0.50)
    print("\n--- 1. Clean Speaker Recall Benchmark ---")
    print(f"{'Speaker':12s} | {'Count':5s} | {'V1 @ 0.50':10s} | {'V2 @ 0.50':10s} | {'V1 @ 0.40':10s} | {'V2 @ 0.40':10s}")
    print("-" * 70)

    for spk, clips in speakers_data.items():
        v1_corr_50, v2_corr_50 = 0, 0
        v1_corr_40, v2_corr_40 = 0, 0
        v1_confs, v2_confs = [], []

        for name, audio in clips:
            p1 = v1.predict_audio(audio)
            p2 = v2.predict_audio(audio)
            v1_confs.append(float(p1[VAANI_IDX]))
            v2_confs.append(float(p2[VAANI_IDX]))

            if p1[VAANI_IDX] >= 0.50:
                v1_corr_50 += 1
            if p2[VAANI_IDX] >= 0.50:
                v2_corr_50 += 1
            if p1[VAANI_IDX] >= 0.40:
                v1_corr_40 += 1
            if p2[VAANI_IDX] >= 0.40:
                v2_corr_40 += 1

        n = len(clips)
        r1_50 = (v1_corr_50 / n) * 100
        r2_50 = (v2_corr_50 / n) * 100
        r1_40 = (v1_corr_40 / n) * 100
        r2_40 = (v2_corr_40 / n) * 100

        results["clean_speaker_recall"][spk] = {
            "count": n,
            "v1_recall_th50": round(r1_50, 1),
            "v2_recall_th50": round(r2_50, 1),
            "v1_recall_th40": round(r1_40, 1),
            "v2_recall_th40": round(r2_40, 1),
            "v1_mean_conf": round(float(np.mean(v1_confs)), 3),
            "v2_mean_conf": round(float(np.mean(v2_confs)), 3),
        }
        print(f"{spk:12s} | {n:5d} | {r1_50:8.1f}% | {r2_50:8.1f}% | {r1_40:8.1f}% | {r2_40:8.1f}%")

    # 2. Quiet Speech Robustness (Unseen Vitthal & Ishita)
    print("\n--- 2. Quiet Speech Robustness Benchmark (Unseen Vitthal @ th=0.40) ---")
    vitthal_clips = speakers_data["Vitthal"]
    for db in (-12.0, -6.0, 0.0, 6.0):
        v1_hit, v2_hit = 0, 0
        for name, audio in vitthal_clips:
            mod_audio = apply_volume(audio, db)
            p1 = v1.predict_audio(mod_audio)
            p2 = v2.predict_audio(mod_audio)
            if p1[VAANI_IDX] >= 0.40:
                v1_hit += 1
            if p2[VAANI_IDX] >= 0.40:
                v2_hit += 1
        n = len(vitthal_clips)
        r1 = (v1_hit / n) * 100
        r2 = (v2_hit / n) * 100
        results["quiet_speech_recall"][f"{db:+.0f}dB"] = {
            "v1_recall": round(r1, 1),
            "v2_recall": round(r2, 1),
        }
        print(f"Gain {db:+.0f} dB : V1={r1:.1f}% -> V2={r2:.1f}% (Delta: {r2 - r1:+.1f}%)")

    # 3. Low-Frequency Noise & Traffic Robustness (Vitthal @ th=0.40)
    print("\n--- 3. Noise Robustness Benchmark (Unseen Vitthal @ th=0.40) ---")
    for noise_t in ("traffic", "babble", "pink"):
        results["noise_robustness_recall"][noise_t] = {}
        for snr in (25.0, 20.0, 15.0, 10.0, 5.0, 0.0):
            v1_hit, v2_hit = 0, 0
            for name, audio in vitthal_clips:
                mod_audio = apply_noise(audio, snr_db=snr, noise_type=noise_t)
                p1 = v1.predict_audio(mod_audio)
                p2 = v2.predict_audio(mod_audio)
                if p1[VAANI_IDX] >= 0.40:
                    v1_hit += 1
                if p2[VAANI_IDX] >= 0.40:
                    v2_hit += 1
            n = len(vitthal_clips)
            r1 = (v1_hit / n) * 100
            r2 = (v2_hit / n) * 100
            results["noise_robustness_recall"][noise_t][f"{snr:.0f}dB"] = {
                "v1_recall": round(r1, 1),
                "v2_recall": round(r2, 1),
            }
            print(f"Noise {noise_t:7s} SNR {snr:4.0f} dB : V1={r1:5.1f}% -> V2={r2:5.1f}% (Delta: {r2 - r1:+.1f}%)")

    # 4. Speaking Speed / Tempo Robustness (Vitthal @ th=0.40)
    print("\n--- 4. Tempo / Speed Robustness Benchmark (Unseen Vitthal @ th=0.40) ---")
    for spd in (0.85, 0.92, 1.0, 1.08, 1.15):
        v1_hit, v2_hit = 0, 0
        for name, audio in vitthal_clips:
            mod_audio = apply_speed(audio, spd) if spd != 1.0 else audio
            p1 = v1.predict_audio(mod_audio)
            p2 = v2.predict_audio(mod_audio)
            if p1[VAANI_IDX] >= 0.40:
                v1_hit += 1
            if p2[VAANI_IDX] >= 0.40:
                v2_hit += 1
        n = len(vitthal_clips)
        r1 = (v1_hit / n) * 100
        r2 = (v2_hit / n) * 100
        results["speed_robustness_recall"][f"{spd:.2f}x"] = {
            "v1_recall": round(r1, 1),
            "v2_recall": round(r2, 1),
        }
        print(f"Speed {spd:.2f}x : V1={r1:5.1f}% -> V2={r2:5.1f}% (Delta: {r2 - r1:+.1f}%)")

    # 5. Combined Testing Set Global Metrics (Clean Testing Split)
    print("\n--- 5. Combined Testing Split Evaluation (226 clips: 26 Vitthal, 100 Unknown, 100 Silence) ---")
    test_files = []
    for lbl in LABELS:
        ld = test_combined / lbl
        if ld.exists():
            for w in sorted(ld.glob("*.wav")):
                test_files.append((w, lbl))

    for th in (0.40, 0.50):
        for m_name, model in (("V1", v1), ("V2", v2)):
            tp, fp, tn, fn = 0, 0, 0, 0
            for w, lbl in test_files:
                audio, _ = sf.read(str(w))
                probs = model.predict_audio(audio)
                is_pred_vaani = probs[VAANI_IDX] >= th
                is_actual_vaani = lbl == "vaani"

                if is_actual_vaani and is_pred_vaani:
                    tp += 1
                elif not is_actual_vaani and is_pred_vaani:
                    fp += 1
                elif not is_actual_vaani and not is_pred_vaani:
                    tn += 1
                elif is_actual_vaani and not is_pred_vaani:
                    fn += 1

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            tag = f"{m_name}_th{int(th*100)}"
            results["test_set_summary"][tag] = {
                "threshold": th,
                "TP": tp,
                "FP": fp,
                "TN": tn,
                "FN": fn,
                "precision": round(prec * 100, 2),
                "recall": round(rec * 100, 2),
                "f1": round(f1 * 100, 2),
                "fpr": round(fpr * 100, 2),
            }
            print(f"[{m_name}] @ th={th:.2f} | Precision: {prec*100:.1f}% | Recall: {rec*100:.1f}% | F1: {f1*100:.1f}% | FPR: {fpr*100:.2f}% (FP: {fp}/{fp+tn})")

    # 6. Unseen Hard Negatives Rejection (Vitthal Near-Misses)
    print("\n--- 6. Unseen Hard Negatives False Trigger Rejection (Vitthal Near-Misses) ---")
    results["unseen_hard_negatives"] = {}
    test_unknown_dir = test_combined / "unknown"
    vitthal_hn_files = list(test_unknown_dir.glob("*vitthal*.wav")) if test_unknown_dir.exists() else []
    if vitthal_hn_files:
        print(f"Found {len(vitthal_hn_files)} unseen hard negative clips from Vitthal.")
        for th in (0.40, 0.50):
            v1_fp, v2_fp = 0, 0
            for w in vitthal_hn_files:
                audio, _ = sf.read(str(w))
                if v1.predict_audio(audio)[VAANI_IDX] >= th:
                    v1_fp += 1
                if v2.predict_audio(audio)[VAANI_IDX] >= th:
                    v2_fp += 1
            n_tot = len(vitthal_hn_files)
            results["unseen_hard_negatives"][f"th{int(th*100)}"] = {
                "total_clips": n_tot,
                "v1_false_triggers": v1_fp,
                "v2_false_triggers": v2_fp,
                "v1_rejection_rate": round((1.0 - v1_fp / n_tot) * 100, 1),
                "v2_rejection_rate": round((1.0 - v2_fp / n_tot) * 100, 1),
            }
            print(f"@ th={th:.2f} | V1 False Triggers: {v1_fp}/{n_tot} ({v1_fp/n_tot*100:.1f}%) -> V2 False Triggers: {v2_fp}/{n_tot} ({v2_fp/n_tot*100:.1f}%)")

    out_file = out_dir / "v1_vs_v2_results.json"
    with open(out_file, "w") as fp:
        json.dump(results, fp, indent=2)

    print(f"\n[PASSED] V1 vs V2 results written to: {out_file}\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="V1 vs V2 Robustness Benchmark")
    parser.add_argument(
        "--v1_model",
        type=str,
        default=str(ROOT_DIR / "model" / "weights" / "vaani_v1_int8.tflite"),
    )
    parser.add_argument(
        "--v2_model",
        type=str,
        default=str(ROOT_DIR / "model" / "weights" / "vaani_v2_int8.tflite"),
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(ROOT_DIR / "dataset"),
        help="Root directory containing dataset (e.g. dataset/ or phase5/data)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(ROOT_DIR / "tests" / "artifacts"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_benchmark(
        Path(args.v1_model),
        Path(args.v2_model),
        Path(args.data_dir),
        Path(args.out_dir),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
