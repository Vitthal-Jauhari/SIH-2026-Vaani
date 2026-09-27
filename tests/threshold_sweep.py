"""
Phase 5 - Step 12: Comprehensive Threshold Analysis for V1 vs V2

Sweeps detection thresholds from 0.30 to 0.90 in increments of 0.05 across:
- All 206 positive Vaani recordings (Ananya, Ark, Ishita, Umang, Vitthal)
- 200 clean test negative clips (100 unknown speech + 100 silence)

Outputs for every threshold:
- TP, FP, TN, FN
- Recall, Precision, F1
- False Positive Rate (FPR), False Negative Rate (FNR)

Saves comparison to: phase5/artifacts/threshold_sweep.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.features import extract_mfcc  # noqa: E402
from model.architecture import LABELS  # noqa: E402

VAANI_IDX = LABELS.index("vaani")
CLIP_LEN = 16000


class ModelEvaluator:
    def __init__(self, model_path: Path):
        self.path = model_path.resolve()
        self.interpreter = tf.lite.Interpreter(model_path=str(self.path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        self.in_scale, self.in_zp = self.input_details["quantization"]
        self.out_scale, self.out_zp = self.output_details["quantization"]
        self.is_quantized = self.input_details["dtype"] == np.int8

    def predict_file(self, wav_path: Path) -> float:
        audio, sr = sf.read(str(wav_path))
        if len(audio) < CLIP_LEN:
            audio = np.pad(audio, (0, CLIP_LEN - len(audio)), mode="constant")
        else:
            audio = audio[:CLIP_LEN]

        mfcc = extract_mfcc(audio)
        inp = np.expand_dims(mfcc, (0, -1))
        if self.is_quantized:
            inp = np.round(inp / self.in_scale + self.in_zp).astype(np.int8)

        self.interpreter.set_tensor(self.input_details["index"], inp)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details["index"])
        if self.is_quantized:
            out = (out.astype(np.float32) - self.out_zp) * self.out_scale
        return float(out[0, VAANI_IDX])


def run_threshold_sweep(
    v1_model_path: Path,
    v2_model_path: Path,
    phase5_root: Path,
    out_dir: Path,
):
    print("=" * 70)
    print("PHASE 5 - STEP 12: THRESHOLD SWEEP (0.30 - 0.90) FOR V1 VS V2")
    print("=" * 70)

    v1_eval = ModelEvaluator(v1_model_path)
    v2_eval = ModelEvaluator(v2_model_path)

    pos_dir = phase5_root / "data" / "normalized" / "positives"
    test_neg_dir = phase5_root / "data" / "combined" / "testing"

    # Collect positives
    pos_files = list(pos_dir.rglob("*.wav"))
    # Collect negatives (unknown + silence from testing split)
    neg_files = list((test_neg_dir / "unknown").glob("*.wav")) + list((test_neg_dir / "silence").glob("*.wav"))

    print(f"Total Positive Clips: {len(pos_files)}")
    print(f"Total Negative Clips: {len(neg_files)}")

    # Precompute scores
    print("Precomputing scores for V1 and V2...")
    v1_pos_scores = [v1_eval.predict_file(f) for f in pos_files]
    v2_pos_scores = [v2_eval.predict_file(f) for f in pos_files]
    v1_neg_scores = [v1_eval.predict_file(f) for f in neg_files]
    v2_neg_scores = [v2_eval.predict_file(f) for f in neg_files]

    thresholds = [round(t, 2) for t in np.arange(0.30, 0.95, 0.05)]
    sweep_results = {"v1": {}, "v2": {}}

    print("\nTHRESHOLD SWEEP SUMMARY:")
    print(f"{'Thresh':6s} | {'V1 Rec':8s} {'V1 Prec':8s} {'V1 F1':8s} {'V1 FPR':8s} | {'V2 Rec':8s} {'V2 Prec':8s} {'V2 F1':8s} {'V2 FPR':8s}")
    print("-" * 85)

    for th in thresholds:
        for tag, pos_sc, neg_sc in (("v1", v1_pos_scores, v1_neg_scores), ("v2", v2_pos_scores, v2_neg_scores)):
            tp = sum(1 for s in pos_sc if s >= th)
            fn = len(pos_sc) - tp
            fp = sum(1 for s in neg_sc if s >= th)
            tn = len(neg_sc) - fp

            rec = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
            prec = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            fpr = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0.0
            fnr = (fn / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0

            sweep_results[tag][f"{th:.2f}"] = {
                "threshold": th,
                "TP": tp,
                "FP": fp,
                "TN": tn,
                "FN": fn,
                "recall": round(rec, 2),
                "precision": round(prec, 2),
                "f1": round(f1, 2),
                "fpr": round(fpr, 2),
                "fnr": round(fnr, 2),
            }

        s1 = sweep_results["v1"][f"{th:.2f}"]
        s2 = sweep_results["v2"][f"{th:.2f}"]
        print(f"{th:6.2f} | {s1['recall']:7.1f}% {s1['precision']:7.1f}% {s1['f1']:7.1f}% {s1['fpr']:7.2f}% | "
              f"{s2['recall']:7.1f}% {s2['precision']:7.1f}% {s2['f1']:7.1f}% {s2['fpr']:7.2f}%")

    out_file = out_dir / "threshold_sweep.json"
    with open(out_file, "w") as fp:
        json.dump(sweep_results, fp, indent=2)

    print(f"\nSaved threshold sweep results to: {out_file}\n")
    return sweep_results


def main():
    parser = argparse.ArgumentParser(description="Threshold Sweep for Vaani Models")
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
        help="Root directory containing dataset",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(ROOT_DIR / "tests" / "artifacts"),
    )
    args = parser.parse_args()

    run_threshold_sweep(
        Path(args.v1_model),
        Path(args.v2_model),
        Path(args.data_dir),
        Path(args.out_dir),
    )


if __name__ == "__main__":
    main()
