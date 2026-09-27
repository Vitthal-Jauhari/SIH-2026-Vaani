<p align="center">
  <h1 align="center">🎤 Vaani</h1>
  <p align="center">
    <strong>Edge-Native Wake-Word Detection & Dual-Core ESP-IDF Pipeline</strong>
  </p>
  <p align="center">
    A production-grade Keyword Spotting (KWS) system that trains, evaluates, and deploys a custom wake-word detector to ESP32 / ESP32-S3 — fitting the entire neural network in <strong>13.38 KB</strong> of Flash with <strong>14.5 ms</strong> inference latency on a dual-core FreeRTOS architecture.
  </p>
  <p align="center">
    <a href="#-key-results--benchmarks"><img src="https://img.shields.io/badge/Unseen_Speaker_Recall-96.1%25-brightgreen?style=for-the-badge" alt="Recall"></a>
    <a href="#-microcontroller-resource-budget"><img src="https://img.shields.io/badge/Model_Size-13.38_KB-blue?style=for-the-badge" alt="Size"></a>
    <a href="#-microcontroller-resource-budget"><img src="https://img.shields.io/badge/Latency-14.5_ms-orange?style=for-the-badge" alt="Latency"></a>
    <a href="#-microcontroller-resource-budget"><img src="https://img.shields.io/badge/Parameters-4,643-purple?style=for-the-badge" alt="Params"></a>
    <a href="#-dual-core-esp-idf-architecture"><img src="https://img.shields.io/badge/Runtime-ESP--IDF_Dual--Core-red?style=for-the-badge" alt="ESP-IDF"></a>
  </p>
</p>

---

## 📌 Problem Statement

Voice-activated edge devices require a lightweight, always-on wake-word detector running **completely on-device** — without cloud latency, mandatory internet connectivity, or power-hungry coprocessors.

The core challenge is fitting an accurate neural network within the strict memory constraints of an ultra-low-cost ESP32 microcontroller, while generalizing reliably across **unseen speakers**, **noisy acoustic environments**, and **phonetically similar hard decoy words** (*"Paani"*, *"Rani"*, *"Naani"*, *"Kahaani"*).

---

## 🏗️ Dual-Core ESP-IDF Architecture

The production firmware in [firmware/vaani-wakeword/](file:///C:/Codes/SIH-2026-Workspace/firmware/vaani-wakeword) leverages both 240 MHz Xtensa cores of the ESP32 in an asymmetric FreeRTOS pipeline:

```
                        ┌─────────────────────────────────────────────────────────┐
                        │             CORE 0 (PRO_CPU) — Capture & Gate           │
                        │                                                         │
  INMP441 I2S MEMS      │  Blocking DMA Read     1.0s Rolling Buffer              │
  ─────────────────────▶│ ───────────────────▶ [ 16,000 samples ]                │
  16 kHz Mono PCM       │   (~0% CPU idle)              │                         │
                        │                               ▼                         │
                        │                       Energy VAD Gate                   │
                        │                      (Speech START / END)               │
                        └───────────────────────────────┬─────────────────────────┘
                                                        │
                                    Window Ready & VAD Active (200ms Hop)
                                    [Single Snapshot Buffer | s_infer_busy]
                                                        │
                                                        ▼
                        ┌─────────────────────────────────────────────────────────┐
                        │             CORE 1 (APP_CPU) — DSP & Inference          │
                        │                                                         │
                        │   MFCC Feature Extraction (63 frames × 13 coeffs)       │
                        │   [40 Mel Bins | Slaney Normalized | Ortho DCT-II]      │
                        │                       │                                 │
                        │                       ▼                                 │
                        │   DS-CNN (INT8 TFLite Micro)                            │
                        │   [13.38 KB Flash | 4,643 parameters | 14.5ms latency]  │
                        │                       │                                 │
                        │                       ▼                                 │
                        │          [ Silence | Unknown | Vaani ]                  │
                        │                       │                                 │
                        │                       ▼                                 │
                        │   Multi-Window Confirmation (KWS_MIN_CONSECUTIVE = 2)   │
                        │                       │                                 │
                        │                       ▼                                 │
                        │             🎯 WAKE WORD CONFIRMED!                     │
                        │         (GPIO 2 Status LED + Serial Trigger)            │
                        └─────────────────────────────────────────────────────────┘
```

### Key Engineering Optimizations

1. **Zero-Lag Dual-Core Decoupling**: Core 0 stays DMA-bound capturing audio and running lightweight Voice Activity Detection (VAD). Core 1 only activates when a full, speech-active window is available.
2. **40-Mel Exact Numerical Parity**: Feature extraction matches the Python training pipeline (`N_MELS=40`, 512-point FFT, 13 MFCCs), achieving **0 / 819 (0.00%)** difference on INT8 quantized tensors between C and Python.
3. **DRAM-Safe Snapshot Buffer**: A single snapshot buffer guarded by an atomic busy gate (`s_infer_busy`) replaces memory-heavy double ping-pong buffers, avoiding static `dram0_0_seg` overflow on ESP32-WROOM-32.
4. **Drop-on-Busy Duty Cycle Bounding**: If Core 1 is still processing inference when the next hop arrives, the hop is dropped rather than queued, eliminating latency queues.
5. **2-Window Spike Filter**: Requires two consecutive positive window triggers (`KWS_MIN_CONSECUTIVE = 2`) to confirm an activation, suppressing isolated false positives.

---

## ✨ Key Results & Benchmarks

### Model V1 → V2 Improvement

| Metric | V1 (Baseline) | V2 (Final) | Improvement |
| :--- | :---: | :---: | :---: |
| **Unseen Speaker Recall** (Vitthal, th=0.40) | 82.7% | **96.1%** | **+13.4 pp** |
| **Quiet Speech Recall** (−12 dB) | 52.0% | **89.8%** | **+37.8 pp** |
| **Slow Tempo Recall** (0.85×) | 63.8% | **90.6%** | **+26.8 pp** |
| **Test Set F1 Score** (th=0.40) | 77.2% | **86.5%** | **+9.3 pp** |
| **Test Set Precision** (th=0.40) | 72.4% | **78.7%** | **+6.3 pp** |
| **Continuous Stream False Triggers/hr** | 2,322 | **1,800** | **−22.5%** |

### Per-Speaker Recall @ threshold 0.50

| Speaker | Role | Clips | V1 Recall | V2 Recall |
| :--- | :--- | :---: | :---: | :---: |
| Ananya | Train | 39 | 59.0% | **92.3%** |
| Ark | Train | 49 | 8.2% | **100.0%** |
| Umang | Train | 50 | 50.0% | **100.0%** |
| Mayank | Train | 48 | 20.8% | **100.0%** |
| Ishita | Validation | 42 | 50.0% | **83.3%** |
| **Vitthal** | **🔒 Unseen Test** | **127** | 70.9% | **93.7%** |

> **Zero Leakage:** Vitthal's recordings were **never seen** during training or validation, establishing the true generalization benchmark.

### Noise Robustness (Unseen Speaker, th=0.40)

| Noise Type | 0 dB SNR | 5 dB | 10 dB | 15 dB | 20 dB |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Traffic | 100% | 100% | 100% | 100% | 100% |
| Pink Noise | 100% | 100% | 100% | 100% | 99.2% |
| Babble | 52.0% | 59.8% | 68.5% | 75.6% | 81.9% |

---

## 💾 Microcontroller Resource Budget

| Resource | Budget | Vaani V2 (ESP32) | Utilization |
| :--- | ---: | ---: | :---: |
| **Flash (Model)** | 256.00 KB | **13.38 KB** (13,696 B) | **5.2%** |
| **SRAM (Tensor Arena)** | 512.00 KB | **~43.40 KB** | **8.5%** |
| **Inference Latency** | < 100 ms | **14.5 ms** | ✅ |
| **Model Parameters** | — | **4,643** | — |
| **Quantization** | — | Full INT8 | — |

> The model occupies only **5.2% of Flash**, leaving over **94% headroom** for application logic, Wi-Fi networking, Bluetooth stacks, and OTA updates.

---

## 📂 Repository Structure

```text
SIH-2026-Workspace/
├── dataset/
│   ├── audio/                     # 🎙️ Ground-truth audio recordings (6 speakers, 250+ clips)
│   └── README.md                  #   Dataset provenance, speaker roles & guidelines
│
├── docs/
│   ├── data_collection_plan.md    #   Acoustic diversity and protocol specification
│   ├── dataset_prep_arch.txt      #   Data pipeline architectural diagram
│   ├── notebooks/                 #   Exploratory research Jupyter notebooks
│   └── reports/                   #   Historical phase reports & validation audits
│
├── firmware/
│   └── vaani-wakeword/            # ⚡ PRODUCTION ESP-IDF FIRMWARE
│       ├── main/
│       │   ├── CMakeLists.txt     #   Component configuration & sources
│       │   ├── idf_component.yml  #   Managed dependency: espressif/esp-tflite-micro
│       │   ├── main.c             #   Dual-core FreeRTOS coordinator & task pinning
│       │   ├── i2s_mic.c / .h     #   INMP441 I2S DMA driver (GPIO 26/25/22)
│       │   ├── vad.c / .h         #   Energy VAD speech gate & state machine
│       │   ├── mel_features.c / .h#   On-device 63×13 MFCC DSP engine
│       │   ├── mel_tables.h       #   Precomputed 40-Mel filterbank & DCT tables
│       │   ├── kws_model.cpp / .h #   TFLite Micro C++ wrapper & inference engine
│       │   └── model_data.cc / .h #   Compiled 13.38 KB INT8 model byte array
│       ├── tools/
│       │   ├── generate_mel_tables.py  # Generates mel_tables.h from model/features.py
│       │   ├── validate_c_vs_python.py # Validates exact numerical feature parity
│       │   ├── verify_c_features.c     # Native C validation harness
│       │   ├── receive_mic.py          # Serial audio capture from ESP32
│       │   └── convert_raw_wav.py      # Raw I2S sample conversion tool
│       ├── CMakeLists.txt         #   ESP-IDF project CMakeLists
│       ├── dependencies.lock      #   Component manager lockfile
│       └── sdkconfig.defaults     #   Target configuration (240MHz, FreeRTOS dual-core)
│
├── model/
│   ├── architecture.py            # 🧠 Canonical Tiny DS-CNN architecture (Keras)
│   ├── features.py                # 🎛️ Librosa 40-Mel, 13-MFCC preprocessing definition
│   ├── export_to_c.py             # 🔄 Exports .tflite weights to model_data.cc / .h
│   └── weights/
│       ├── vaani_v1_int8.tflite   #   Phase 3 baseline model
│       └── vaani_v2_int8.tflite   #   Canonical V2 production INT8 model (13,696 bytes)
│
├── tests/
│   ├── test_model_offline.py      # 🧪 Offline WAV tester replicating firmware feature pipeline
│   ├── test_live_mic.py           # 🎤 Real-time PC microphone wake-word streaming test
│   ├── evaluate_benchmark.py      # 📊 Apples-to-apples V1 vs V2 robustness benchmark
│   ├── threshold_sweep.py         # 📈 FAR/FRR threshold calibration sweep
│   ├── continuous_eval.py         # ⏱️ 10-minute continuous streaming evaluation
│   ├── estimate_resources.py      # 💾 Microcontroller memory & latency estimator
│   └── artifacts/                 #   Benchmark results, sweeps & metadata JSONs
│
├── training/
│   ├── prepare_dataset.py         # 🔄 Normalizes raw audio to standard 16 kHz mono WAV
│   ├── prepare_combined_dataset.py# 🔀 Combines positives, speech commands & noise
│   ├── split_speakers.py          # 🔒 Speaker-disjoint train/val/test partitioning
│   ├── augment.py                 # 🎚️ Noise, volume, pitch & speed augmentations
│   ├── train.py                   # 🚀 Trains Tiny DS-CNN on combined dataset
│   └── quantize.py                # 📦 Full INT8 post-training quantization
│
├── LICENSE                        # MIT License
└── README.md                      # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Test Wake-Word Locally with PC Microphone

Test the trained model directly on your computer before touching hardware:

```bash
# Install Python dependencies
pip install tensorflow numpy librosa soundfile sounddevice

# Run live tester with Model V2 — speak "Vaani" into your mic
python tests/test_live_mic.py

# Adjust sensitivity threshold or specify custom audio device
python tests/test_live_mic.py --threshold 0.50 --consecutive 2
```

### 2. Run Offline Model Verification on WAV

Verify that the model processes audio offline through the exact same feature pipeline:

```bash
python tests/test_model_offline.py --wav mic_test.wav
```

### 3. Verify C vs Python Numerical Parity

Confirm that the on-device C DSP pipeline produces the exact same INT8 tensors as Python:

```bash
python firmware/vaani-wakeword/tools/validate_c_vs_python.py
```

### 4. Build Firmware with ESP-IDF

```powershell
# Activate ESP-IDF environment (ESP-IDF v5.x / v6.x)
. C:\Espressif\tools\Microsoft.v6.1.PowerShell_profile.ps1

# Navigate to firmware project
cd firmware\vaani-wakeword

# Build the firmware binary
idf.py build
```

When ready to flash to hardware:
```powershell
idf.py -p COM3 flash monitor
```

---

## 🔌 Hardware Wiring (INMP441 to ESP32)

| INMP441 Pin | ESP32 GPIO | Description |
| :--- | :--- | :--- |
| **VDD / 3V3** | **3V3** | 3.3V Power |
| **GND** | **GND** | Ground |
| **SD / DIN** | **GPIO 22** | Serial Data Input |
| **WS / LRCLK** | **GPIO 25** | Word Select (Left/Right Clock) |
| **SCK / BCK** | **GPIO 26** | Bit Clock |
| **L/R** | **GND** | Left Channel Select |
| **Status LED** | **GPIO 2** | Built-in / External Indicator LED |

---

## 🔬 Reproducing the Training Pipeline

To retrain the model from scratch:

```bash
# 1. Normalize dataset audio to 16 kHz mono WAV
python training/prepare_dataset.py

# 2. Partition speakers with zero leakage (held-out test speaker)
python training/split_speakers.py

# 3. Build balanced train/val/test combined dataset
python training/prepare_combined_dataset.py

# 4. Train the Tiny DS-CNN model
python training/train.py

# 5. Full INT8 post-training quantization
python training/quantize.py

# 6. Export quantized model to C arrays for firmware
python model/export_to_c.py
```

---

## 👥 Authors & Acknowledgments

Developed for **SIH (Smart India Hackathon) 2026**:
- **Vitthal Jauhari** — Dual-core ESP-IDF firmware, I2S DMA, VAD gating, DSP numerical parity, and hardware deployment
- **Umang** — Model research, training pipeline, hard negative mining, benchmarks
- **Mayank Singh** — Model architecture, dataset processing, and quantization pipeline
- **Dataset Contributors** — Ananya, Ark, Ishita, Mayank, Umang, Vitthal

---

## 📜 License

Distributed under the [MIT License](LICENSE).
