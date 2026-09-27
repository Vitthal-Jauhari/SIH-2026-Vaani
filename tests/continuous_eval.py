"""
Phase 5 - Step 13: Continuous Streaming Evaluation for V1 vs V2

Simulates real-world streaming deployment:
- 1.0-second rolling circular buffer @ 16 kHz
- 100 ms hop step (1,600 samples)
- Cooldown / debounce: 0.8s
- 10-minute continuous timeline with 15 embedded wake words amidst continuous speech dialogue and background sounds
- Side-by-side comparison of V1 vs V2 on EXACTLY THE SAME timeline:
  * Detection Recall (%)
  * Detection Latency (ms)
  * False Trigger Count
  * False Trigger Rate (triggers/hour)

Saves results to: phase5/artifacts/continuous_eval.json
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import soundfile as sf
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.features import extract_mfcc  # noqa: E402
from model.architecture import LABELS  # noqa: E402

VAANI_IDX = LABELS.index("vaani")
TARGET_SAMPLE_RATE = 16000
BUFFER_SAMPLES = 16000  # 1.0s
HOP_SAMPLES = 1600      # 100ms
COOLDOWN_FRAMES = 8     # 8 * 100ms = 800ms


class StreamingClassifier:
    def __init__(self, model_path: Path):
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        self.in_scale, self.in_zp = self.input_details["quantization"]
        self.out_scale, self.out_zp = self.output_details["quantization"]
        self.is_quantized = self.input_details["dtype"] == np.int8

    def predict_window(self, window_audio: np.ndarray) -> float:
        mfcc = extract_mfcc(window_audio)
        inp = np.expand_dims(mfcc, (0, -1))
        if self.is_quantized:
            inp = np.round(inp / self.in_scale + self.in_zp).astype(np.int8)

        self.interpreter.set_tensor(self.input_details["index"], inp)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details["index"])
        if self.is_quantized:
            out = (out.astype(np.float32) - self.out_zp) * self.out_scale
        return float(out[0, VAANI_IDX])


def generate_timeline(
    pos_clips: List[np.ndarray],
    neg_clips: List[np.ndarray],
    duration_sec: float = 600.0,
    n_targets: int = 15,
    seed: int = 42,
):
    random.seed(seed)
    np.random.seed(seed)

    total_samples = int(duration_sec * TARGET_SAMPLE_RATE)
    timeline = np.random.normal(0, 0.005, total_samples).astype(np.float32)

    # Pick 15 evenly spaced target intervals with random jitter
    spacing = total_samples // (n_targets + 1)
    ground_truth_events = []

    for i in range(1, n_targets + 1):
        jitter = random.randint(-int(TARGET_SAMPLE_RATE * 3), int(TARGET_SAMPLE_RATE * 3))
        center_sample = i * spacing + jitter
        clip = random.choice(pos_clips)
        clip = clip[:TARGET_SAMPLE_RATE]
        start = max(0, min(total_samples - len(clip), center_sample - len(clip) // 2))
        end = start + len(clip)

        # Mix clip into timeline
        timeline[start:end] += clip
        ground_truth_events.append({
            "target_id": i,
            "start_sec": round(start / TARGET_SAMPLE_RATE, 3),
            "end_sec": round(end / TARGET_SAMPLE_RATE, 3),
            "start_sample": start,
            "end_sample": end,
        })

    # Fill background gaps with conversational negative speech
    neg_idx = 0
    cur = 0
    while cur < total_samples - TARGET_SAMPLE_RATE:
        # Check if inside any target wake-word region (+/- 1.5s margin)
        in_target = False
        for gt in ground_truth_events:
            if abs(cur - gt["start_sample"]) < int(TARGET_SAMPLE_RATE * 2.0):
                in_target = True
                break
        if not in_target and random.random() < 0.7:
            neg_clip = neg_clips[neg_idx % len(neg_clips)][:TARGET_SAMPLE_RATE]
            neg_idx += 1
            gain = random.uniform(0.3, 0.8)
            end_s = min(total_samples, cur + len(neg_clip))
            timeline[cur:end_s] += neg_clip[: end_s - cur] * gain
            cur += len(neg_clip) + random.randint(int(TARGET_SAMPLE_RATE * 0.5), int(TARGET_SAMPLE_RATE * 2.0))
        else:
            cur += random.randint(int(TARGET_SAMPLE_RATE * 0.5), int(TARGET_SAMPLE_RATE * 1.5))

    timeline = np.clip(timeline, -1.0, 1.0)
    return timeline, ground_truth_events


def simulate_stream(
    classifier: StreamingClassifier,
    timeline: np.ndarray,
    ground_truth_events: List[Dict[str, Any]],
    threshold: float = 0.40,
):
    total_samples = len(timeline)
    buffer = np.zeros(BUFFER_SAMPLES, dtype=np.float32)

    detections = []
    cooldown_counter = 0

    cur = 0
    frame_idx = 0
    while cur + HOP_SAMPLES <= total_samples:
        new_samples = timeline[cur : cur + HOP_SAMPLES]
        buffer = np.roll(buffer, -HOP_SAMPLES)
        buffer[-HOP_SAMPLES:] = new_samples

        if cooldown_counter > 0:
            cooldown_counter -= 1
        else:
            prob = classifier.predict_window(buffer)
            if prob >= threshold:
                det_time_sec = (cur + HOP_SAMPLES) / TARGET_SAMPLE_RATE
                detections.append({
                    "frame_idx": frame_idx,
                    "time_sec": round(det_time_sec, 3),
                    "confidence": round(prob, 4),
                })
                cooldown_counter = COOLDOWN_FRAMES

        cur += HOP_SAMPLES
        frame_idx += 1

    # Match detections with ground truth
    matched_targets = set()
    false_triggers = 0
    latencies = []

    for det in detections:
        det_t = det["time_sec"]
        matched = False
        for gt in ground_truth_events:
            # Wake word ends at gt["end_sec"]. Detection should happen near or shortly after onset
            if gt["start_sec"] - 0.2 <= det_t <= gt["end_sec"] + 1.2:
                matched = True
                if gt["target_id"] not in matched_targets:
                    matched_targets.add(gt["target_id"])
                    lat = max(0.0, (det_t - gt["start_sec"]) * 1000.0)
                    latencies.append(lat)
                break
        if not matched:
            false_triggers += 1

    recall = (len(matched_targets) / len(ground_truth_events)) * 100
    mean_lat = np.mean(latencies) if latencies else 0.0
    duration_hours = (total_samples / TARGET_SAMPLE_RATE) / 3600.0
    fpr_per_hr = false_triggers / duration_hours if duration_hours > 0 else 0.0

    return {
        "total_targets": len(ground_truth_events),
        "detected_targets": len(matched_targets),
        "recall": round(recall, 1),
        "false_triggers": false_triggers,
        "false_triggers_per_hour": round(fpr_per_hr, 1),
        "mean_latency_ms": round(float(mean_lat), 1),
        "total_detections": len(detections),
    }


def run_continuous_eval(
    v1_model_path: Path,
    v2_model_path: Path,
    phase5_root: Path,
    out_dir: Path,
    duration_sec: float = 600.0,
    seed: int = 42,
):
    print("=" * 70)
    print(f"PHASE 5 - STEP 13: 10-MINUTE CONTINUOUS STREAMING BENCHMARK ({duration_sec/60:.1f} mins)")
    print("=" * 70)

    v1 = StreamingClassifier(v1_model_path)
    v2 = StreamingClassifier(v2_model_path)

    # Collect positive and negative clips
    pos_dir = phase5_root / "data" / "normalized" / "positives"
    neg_dir = phase5_root / "data" / "combined" / "testing" / "unknown"

    pos_clips = [sf.read(str(w))[0] for w in pos_dir.rglob("*.wav")]
    neg_clips = [sf.read(str(w))[0] for w in neg_dir.glob("*.wav")]

    timeline, ground_truth = generate_timeline(
        pos_clips, neg_clips, duration_sec=duration_sec, n_targets=15, seed=seed
    )

    print(f"Synthesized continuous timeline: {len(timeline)/TARGET_SAMPLE_RATE:.1f}s with {len(ground_truth)} wake-word instances.")
    print("Evaluating streaming sliding-window (100ms hop, 0.8s cooldown) on identical audio...")

    v1_res = simulate_stream(v1, timeline, ground_truth, threshold=0.40)
    v2_res = simulate_stream(v2, timeline, ground_truth, threshold=0.40)

    results = {
        "duration_minutes": duration_sec / 60.0,
        "n_embedded_targets": len(ground_truth),
        "hop_ms": 100,
        "buffer_ms": 1000,
        "cooldown_ms": 800,
        "threshold": 0.40,
        "v1": v1_res,
        "v2": v2_res,
    }

    print("\nCONTINUOUS STREAMING RESULTS (th=0.40):")
    print(f"  V1 Baseline : Recall={v1_res['recall']:.1f}% ({v1_res['detected_targets']}/{v1_res['total_targets']}) | "
          f"False Triggers={v1_res['false_triggers']} ({v1_res['false_triggers_per_hour']}/hr) | Latency={v1_res['mean_latency_ms']:.1f} ms")
    print(f"  V2 Candidate: Recall={v2_res['recall']:.1f}% ({v2_res['detected_targets']}/{v2_res['total_targets']}) | "
          f"False Triggers={v2_res['false_triggers']} ({v2_res['false_triggers_per_hour']}/hr) | Latency={v2_res['mean_latency_ms']:.1f} ms")

    out_file = out_dir / "continuous_eval.json"
    with open(out_file, "w") as fp:
        json.dump(results, fp, indent=2)

    print(f"\nSaved streaming results to: {out_file}\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="Continuous Streaming Evaluation for Vaani Models")
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
    parser.add_argument("--duration_sec", type=float, default=600.0)
    args = parser.parse_args()

    run_continuous_eval(
        Path(args.v1_model),
        Path(args.v2_model),
        Path(args.data_dir),
        Path(args.out_dir),
        duration_sec=args.duration_sec,
    )


if __name__ == "__main__":
    main()
