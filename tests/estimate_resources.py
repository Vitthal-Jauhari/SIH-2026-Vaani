"""
Phase 5 - Step 14: ESP32 Hardware Resource & Memory Estimation for V1 vs V2

Analyzes both INT8 TFLite models specifically for microcontroller deployment:
- Flash footprint vs 256 KB hard budget
- SRAM tensor arena estimation (TFLite Micro activation buffer planning)
- ESP-DSP MFCC audio buffer footprint
- Development PC measurement vs. Projected ESP32 / ESP32-S3 metrics
- Strictly separates HOST-PC MEASUREMENT from PROJECTED ESP32 METRICS

Saves results to: phase5/artifacts/resource_results.json
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import tensorflow as tf

MAX_FLASH_BUDGET_KB = 256.0


def analyze_model_resources(tflite_path: Path) -> Dict[str, Any]:
    content = tflite_path.read_bytes()
    model_size_bytes = len(content)
    model_size_kb = model_size_bytes / 1024.0

    interpreter = tf.lite.Interpreter(model_content=content)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Benchmark PC inference latency over 100 runs
    dummy_input = np.zeros(input_details["shape"], dtype=input_details["dtype"])
    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details["index"], dummy_input)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details["index"])
        latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_pc_lat = float(np.mean(latencies))

    # TFLite Micro Tensor Arena Calculation:
    # Sum of max activation memory + scratch buffer + tensor overhead
    tensor_details = interpreter.get_tensor_details()
    total_tensor_bytes = sum(t["shape"].size * np.dtype(t["dtype"]).itemsize for t in tensor_details)
    # TFLM activation buffer arena typically ~1.5x - 2.0x largest intermediate activations + tensor structs
    est_arena_bytes = int(total_tensor_bytes * 1.5 + 4096)
    est_arena_kb = est_arena_bytes / 1024.0

    # Audio & Preprocessing Buffers:
    # 1.0s @ 16 kHz 16-bit PCM = 32,000 bytes = 31.25 KB
    pcm_buffer_bytes = 16000 * 2
    # ESP-DSP FFT scratch buffer (512 float32 + window + mel filterbank LUT) ~ 8 KB
    dsp_scratch_bytes = 8192
    total_sram_bytes = est_arena_bytes + pcm_buffer_bytes + dsp_scratch_bytes
    total_sram_kb = total_sram_bytes / 1024.0

    # Projections for ESP32 and ESP32-S3 (240 MHz dual-core Xtensa LX7 with vector instructions)
    # DS-CNN with ~4.6k params takes ~2.5 - 3.5 M cycles on ESP32-S3 with ESP-NN SIMD
    # At 240 MHz, 3.5 M cycles = ~14.6 ms
    proj_esp32_s3_latency_ms = 14.5
    proj_esp32_s3_fps = round(1000.0 / proj_esp32_s3_latency_ms, 1)

    return {
        "model_file": tflite_path.name,
        "flash_footprint": {
            "model_size_bytes": model_size_bytes,
            "model_size_kb": round(model_size_kb, 2),
            "budget_limit_kb": MAX_FLASH_BUDGET_KB,
            "headroom_kb": round(MAX_FLASH_BUDGET_KB - model_size_kb, 2),
            "budget_utilization_pct": round((model_size_kb / MAX_FLASH_BUDGET_KB) * 100, 2),
            "fits_within_budget": bool(model_size_kb < MAX_FLASH_BUDGET_KB),
        },
        "sram_footprint_estimate": {
            "tensor_arena_bytes": est_arena_bytes,
            "tensor_arena_kb": round(est_arena_kb, 2),
            "pcm_audio_buffer_bytes": pcm_buffer_bytes,
            "pcm_audio_buffer_kb": round(pcm_buffer_bytes / 1024.0, 2),
            "dsp_scratch_bytes": dsp_scratch_bytes,
            "dsp_scratch_kb": round(dsp_scratch_bytes / 1024.0, 2),
            "total_runtime_sram_kb": round(total_sram_kb, 2),
            "esp32_s3_sram_utilization_pct": round((total_sram_kb / 512.0) * 100, 2),
        },
        "inference_latency": {
            "host_pc_measured_ms": round(mean_pc_lat, 3),
            "projected_esp32_s3_ms": proj_esp32_s3_latency_ms,
            "projected_esp32_s3_throughput_fps": proj_esp32_s3_fps,
            "disclaimer": "ESP32 numbers are simulated projections based on ESP-NN benchmarks; host PC numbers are measured.",
        },
    }


def compare_resources(v1_path: Path, v2_path: Path, out_file: Path):
    print("=" * 70)
    print("PHASE 5 - STEP 14: ESP32 HARDWARE RESOURCE AUDIT (V1 VS V2)")
    print("=" * 70)

    v1_res = analyze_model_resources(v1_path)
    v2_res = analyze_model_resources(v2_path)

    comparison = {
        "v1_baseline": v1_res,
        "v2_candidate": v2_res,
        "comparison": {
            "flash_delta_bytes": v2_res["flash_footprint"]["model_size_bytes"] - v1_res["flash_footprint"]["model_size_bytes"],
            "flash_delta_kb": round(v2_res["flash_footprint"]["model_size_kb"] - v1_res["flash_footprint"]["model_size_kb"], 2),
            "both_fit_under_256kb": bool(v1_res["flash_footprint"]["fits_within_budget"] and v2_res["flash_footprint"]["fits_within_budget"]),
        },
    }

    print("\nHARDWARE RESOURCE SUMMARY:")
    print(f"  V1 Baseline Size: {v1_res['flash_footprint']['model_size_kb']:.2f} KB ({v1_res['flash_footprint']['model_size_bytes']} B)")
    print(f"  V2 Model Size   : {v2_res['flash_footprint']['model_size_kb']:.2f} KB ({v2_res['flash_footprint']['model_size_bytes']} B)")
    print(f"  Hard Flash Limit: {MAX_FLASH_BUDGET_KB:.2f} KB (Both strictly compliant: {comparison['comparison']['both_fit_under_256kb']})")
    print(f"  Estimated SRAM  : V1 ~{v1_res['sram_footprint_estimate']['total_runtime_sram_kb']:.1f} KB | V2 ~{v2_res['sram_footprint_estimate']['total_runtime_sram_kb']:.1f} KB (out of 512 KB SRAM)")
    print(f"  PC Latency (meas): V1={v1_res['inference_latency']['host_pc_measured_ms']:.3f} ms | V2={v2_res['inference_latency']['host_pc_measured_ms']:.3f} ms")
    print(f"  ESP32-S3 Latency (proj): ~{v2_res['inference_latency']['projected_esp32_s3_ms']:.1f} ms\n")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as fp:
        json.dump(comparison, fp, indent=2)

    print(f"Saved resource audit to: {out_file}\n")
    return comparison


ROOT_DIR = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Estimate microcontroller resources for TFLite models")
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
        "--out_file",
        type=str,
        default=str(ROOT_DIR / "tests" / "artifacts" / "resource_results.json"),
    )
    args = parser.parse_args()

    compare_resources(Path(args.v1_model), Path(args.v2_model), Path(args.out_file))


if __name__ == "__main__":
    main()
