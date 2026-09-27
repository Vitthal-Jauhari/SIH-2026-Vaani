"""
Phase 5 - Step 8: Post-Training INT8 Quantization & Budget Verification

- Converts trained V2 Keras model to full INT8 TFLite model.
- Uses representative dataset from training split (200 samples).
- Input: INT8 (shape: 1, 63, 13, 1)
- Output: INT8 (shape: 1, 3)
- STRICT GUARDRAIL: Aborts execution if model size >= 256 KB.
- Saves model to: phase5/artifacts/vaani_v2_int8.tflite
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.features import featurize_file  # noqa: E402
from model.architecture import LABELS  # noqa: E402

MAX_BUDGET_KB = 256.0


def representative_dataset_gen(data_dir: Path, n_samples: int = 200):
    paths = []
    for label in LABELS:
        label_dir = data_dir / "training" / label
        if label_dir.exists():
            wavs = list(label_dir.glob("*.wav"))
            paths.extend(wavs[: n_samples // len(LABELS) + 1])

    np.random.seed(42)
    np.random.shuffle(paths)

    def gen():
        for p in paths[:n_samples]:
            feat = featurize_file(str(p))
            feat = np.expand_dims(feat, (0, -1)).astype(np.float32)
            yield [feat]

    return gen


def quantize_v2_model(model_path: Path, data_dir: Path, out_dir: Path):
    model_path = model_path.resolve()
    data_dir = data_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 5 - STEP 8: INT8 QUANTIZATION & BUDGET VERIFICATION")
    print("=" * 70)
    print(f"Source Model     : {model_path}")
    print(f"Data Source      : {data_dir}")
    print(f"Target INT8 Model: {out_dir / 'vaani_v2_int8.tflite'}\n")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = tf.keras.models.load_model(model_path)

    # Export to temporary SavedModel to ensure robust LLVM lowering
    with tempfile.TemporaryDirectory() as tmpdir:
        model.export(tmpdir)
        converter = tf.lite.TFLiteConverter.from_saved_model(tmpdir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen(data_dir, n_samples=200)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        print("Converting to full INT8 TFLite...")
        tflite_model = converter.convert()

    out_path = out_dir / "vaani_v2_int8.tflite"
    out_path.write_bytes(tflite_model)

    size_bytes = len(tflite_model)
    size_kb = size_bytes / 1024.0

    print("\nTFLITE INT8 MODEL SPECIFICATION:")
    print(f"  File Path         : {out_path}")
    print(f"  Model Size (bytes): {size_bytes:,} bytes")
    print(f"  Model Size (KB)   : {size_kb:.2f} KB")
    print(f"  Budget Ceiling    : {MAX_BUDGET_KB:.2f} KB")
    print(f"  Budget Headroom   : {MAX_BUDGET_KB - size_kb:.2f} KB under budget")

    # Verify input / output types using TFLite Interpreter
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    print(f"  Input Tensor Name : {input_details['name']}")
    print(f"  Input Shape       : {input_details['shape'].tolist()}")
    print(f"  Input Dtype       : {input_details['dtype']}")
    print(f"  Input Quantization: {input_details['quantization']}")
    print(f"  Output Tensor Name: {output_details['name']}")
    print(f"  Output Shape      : {output_details['shape'].tolist()}")
    print(f"  Output Dtype      : {output_details['dtype']}")
    print(f"  Output Quant      : {output_details['quantization']}")

    # STRICT GUARDRAIL ASSERTION
    if size_kb >= MAX_BUDGET_KB:
        msg = f"FATAL ERROR: INT8 model size ({size_kb:.2f} KB) exceeds maximum allowed budget ({MAX_BUDGET_KB} KB)!"
        print(msg, file=sys.stderr)
        raise RuntimeError(msg)

    print(f"\n[PASSED] V2 Model is fully INT8 quantized and strictly under 256 KB ({size_kb:.2f} KB < 256 KB).\n")
    return out_path, size_kb


def main():
    parser = argparse.ArgumentParser(description="Quantize V2 Keras model to INT8")
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(ROOT_DIR / "model" / "checkpoints" / "vaani_v2_best.keras"),
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(ROOT_DIR / "dataset" / "combined"),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(ROOT_DIR / "model" / "weights"),
    )
    args = parser.parse_args()

    p = Path(args.model_path)
    if not p.exists() and (p.parent / "vaani_v2_float32.keras").exists():
        p = p.parent / "vaani_v2_float32.keras"

    quantize_v2_model(p, Path(args.data_dir), Path(args.out_dir))


if __name__ == "__main__":
    main()
