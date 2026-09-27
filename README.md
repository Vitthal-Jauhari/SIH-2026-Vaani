<p align="center">
  <h1 align="center">🎤 Vaani</h1>
  <p align="center">
    <strong>Edge-Native Wake-Word Detection & Dual-Core ESP-IDF Pipeline</strong>
  </p>
  <p align="center">
    A production-grade Keyword Spotting (KWS) system that trains, evaluates, and deploys a custom wake-word detector to ESP32 (validated on ESP32 DevKit C) — fitting the entire neural network in <strong>13.38 KB</strong> of Flash with <strong>~14.5 ms</strong> TFLite model inference (<strong>~179 ms</strong> full DSP + KWS cycle on ESP32) on a dual-core FreeRTOS architecture.
  </p>
  <p align="center">
    <a href="#-key-results--benchmarks"><img src="https://img.shields.io/badge/Unseen_Speaker_Recall-96.1%25-brightgreen?style=for-the-badge" alt="Recall"></a>
    <a href="#-microcontroller-resource-budget"><img src="https://img.shields.io/badge/Model_Size-13.38_KB-blue?style=for-the-badge" alt="Size"></a>
    <a href="#-microcontroller-resource-budget"><img src="https://img.shields.io/badge/DSP_+_KWS_Cycle-~179_ms-orange?style=for-the-badge" alt="Cycle Latency"></a>
    <a href="#-microcontroller-resource-budget"><img src="https://img.shields.io/badge/Parameters-4,643-purple?style=for-the-badge" alt="Params"></a>
    <a href="#-dual-core-esp-idf-architecture"><img src="https://img.shields.io/badge/Runtime-ESP--IDF_Dual--Core-red?style=for-the-badge" alt="ESP-IDF"></a>
  </p>
</p>

---

## 📌 Problem Statement

Voice-activated edge devices require a lightweight, always-on wake-word detector running **completely on-device** — without cloud latency, mandatory internet connectivity, or power-hungry coprocessors.

The core challenge is fitting an accurate neural network within the strict memory constraints of an ultra-low-cost ESP32 microcontroller, while generalizing reliably across **unseen speakers**, **noisy acoustic environments**, and **phonetically similar hard decoy words** (_"Paani"_, _"Rani"_, _"Naani"_, _"Kahaani"_).

---

## ✅ Current Validation Status

The Vaani V2 pipeline has been successfully validated on physical ESP32 hardware (**ESP32 DevKit C**):

- **V2 INT8 Model Embedded:** 13.38 KB model array compiled into firmware flash (`model_data.cc`).
- **40-Mel C/Python Parity:** **819 / 819** INT8 elements identical; maximum tested float difference is only **~0.00028**.
- **ESP-IDF v6.1 Build:** PASS (clean dual-core FreeRTOS compilation).
- **Firmware Flashed:** Successfully flashed and booted over UART on ESP32 DevKit C.
- **Physical Wake-Word Detection:** Verified on real hardware using an INMP441 I2S MEMS microphone.
- **Multi-Window Confirmation:** 2-window consecutive trigger filter verified (`KWS_MIN_CONSECUTIVE = 2`).

### 📋 Remaining Validation & Roadmap

- **VAD Speech-End Transition:** Tuning energy hysteresis and silence timeout (hardware tests observe Speech START triggering reliably, but silence does not consistently transition to Speech END, keeping KWS active).
- **Formal Idle CPU Measurement:** Measuring actual idle-duty current draw and CPU utilization once speech-end gating is fully calibrated.
- **Power Management:** Configuring ESP32 dynamic frequency scaling and light-sleep modes between speech windows.
- **Hard-Negative Robustness:** Empirical on-hardware false-accept testing against phonetically similar decoy words (_"Paani"_, _"Rani"_, _"Naani"_).
- **Post-Wake Audio Capture:** Buffering subsequent spoken command audio for transmission.
- **ASR Server Integration:** Streaming captured audio commands to a downstream ASR server.

---

## 🏗️ Dual-Core ESP-IDF Architecture

The production firmware in [firmware/vaani-wakeword/](firmware/vaani-wakeword/) leverages both 240 MHz Xtensa cores of the ESP32 (validated on ESP32 DevKit C) in an asymmetric FreeRTOS pipeline:

```
                        ┌─────────────────────────────────────────────────────────┐
                        │             CORE 0 (PRO_CPU) — Capture & Gate           │
                        │                                                         │
  INMP441 I2S MEMS      │  Blocking DMA Read     1.0s Rolling Buffer              │
  ─────────────────────▶│ ───────────────────▶ [ 16,000 samples ]                │
  16 kHz Mono PCM       │   (DMA-driven capture)        │                         │
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
                        │   [Hardware Execution: ~91 ms]                          │
                        │                       │                                 │
                        │                       ▼                                 │
                        │   DS-CNN (INT8 TFLite Micro)                            │
                        │   [13.38 KB Flash | 4,643 params | ~88 ms on ESP32]     │
                        │   (Isolated benchmark: ~14.5 ms)                        │
                        │                       │                                 │
                        │                       ▼                                 │
                        │          [ Silence | Unknown | Vaani ]                  │
                        │                       │                                 │
                        │                       ▼                                 │
                        │   Multi-Window Confirmation (KWS_MIN_CONSECUTIVE = 2)   │
                        │   [Full DSP + KWS Cycle: ~179 ms]                       │
                        │                       │                                 │
                        │                       ▼                                 │
                        │             🎯 WAKE WORD CONFIRMED!                     │
                        │         (GPIO 2 Status LED + Serial Trigger)            │
                        └─────────────────────────────────────────────────────────┘
```

### Key Engineering Optimizations

1. **Dual-Core Capture/Inference Decoupling**: Core 0 manages I2S DMA audio capture and Energy VAD gating, while Core 1 handles 40-mel DSP feature extraction and TFLite Micro inference independently. The two cores synchronize via a non-blocking snapshot buffer guarded by an atomic busy gate (`s_infer_busy`). Inference is gated by VAD state and this busy flag; formal idle-duty-cycle optimization remains under validation.
2. **40-Mel Exact Numerical Parity**: Feature extraction matches the Python training pipeline (`N_MELS=40`, 512-point FFT, 13 MFCCs), achieving **819 / 819 (100%)** exact INT8 tensor parity between C and Python, with a maximum float difference of only **~0.00028** across all frames.
3. **DRAM-Safe Snapshot Buffer**: A single snapshot buffer guarded by an atomic busy gate (`s_infer_busy`) replaces memory-heavy double ping-pong buffers, avoiding static `dram0_0_seg` overflow on ESP32-WROOM-32.
4. **Drop-on-Busy Duty Cycle Bounding**: If Core 1 is still processing inference when the next hop arrives, the hop is dropped rather than queued, eliminating latency queues.
5. **2-Window Spike Filter**: Requires two consecutive positive window triggers (`KWS_MIN_CONSECUTIVE = 2`) to confirm an activation, suppressing isolated false positives.

---

## ✨ Key Results & Benchmarks

> **Benchmark Provenance:** All benchmark results are generated from the project's offline evaluation suite in [tests/evaluate_benchmark.py](tests/evaluate_benchmark.py) and verified artifacts stored under [tests/artifacts/](tests/artifacts/).

### Model V1 → V2 Improvement

| Metric                                       | V1 (Baseline) | V2 (Final) | Improvement  |
| :------------------------------------------- | :-----------: | :--------: | :----------: |
| **Unseen Speaker Recall** (Vitthal, th=0.40) |     82.7%     | **96.1%**  | **+13.4 pp** |
| **Quiet Speech Recall** (−12 dB)             |     52.0%     | **89.8%**  | **+37.8 pp** |
| **Slow Tempo Recall** (0.85×)                |     63.8%     | **90.6%**  | **+26.8 pp** |
| **Test Set F1 Score** (th=0.40)              |     77.2%     | **86.5%**  | **+9.3 pp**  |
| **Test Set Precision** (th=0.40)             |     72.4%     | **78.7%**  | **+6.3 pp**  |
| **Continuous Stream False Triggers/hr**      |     2,322     | **1,800**  |  **−22.5%**  |

### Per-Speaker Recall @ threshold 0.50

| Speaker     | Role               |  Clips  | V1 Recall | V2 Recall  |
| :---------- | :----------------- | :-----: | :-------: | :--------: |
| Ananya      | Train              |   39    |   59.0%   | **92.3%**  |
| Ark         | Train              |   49    |   8.2%    | **100.0%** |
| Umang       | Train              |   50    |   50.0%   | **100.0%** |
| Mayank      | Train              |   48    |   20.8%   | **100.0%** |
| Ishita      | Validation         |   42    |   50.0%   | **83.3%**  |
| **Vitthal** | **🔒 Unseen Test** | **127** |   70.9%   | **93.7%**  |

> **Zero Leakage:** Vitthal's recordings were **never seen** during training or validation, establishing the true generalization benchmark.

### Noise Robustness (Unseen Speaker, th=0.40)

| Noise Type | 0 dB SNR | 5 dB  | 10 dB | 15 dB | 20 dB |
| :--------- | :------: | :---: | :---: | :---: | :---: |
| Traffic    |   100%   | 100%  | 100%  | 100%  | 100%  |
| Pink Noise |   100%   | 100%  | 100%  | 100%  | 99.2% |
| Babble     |  52.0%   | 59.8% | 68.5% | 75.6% | 81.9% |

---

## 💾 Microcontroller Resource Budget

| Resource                    |             Budget |                     Vaani V2 (ESP32 DevKit C) |          Utilization / Notes          |
| :-------------------------- | -----------------: | --------------------------------------------: | :-----------------------------------: |
| **Flash (Model Binary)**    |          256.00 KB |                       **13.38 KB** (13,696 B) |       **5.2%** of Flash budget        |
| **SRAM (Tensor Arena)**     | 20.00 KB allocated |                            **~17.10 KB used** |       **~3.3%** of 512 KB SRAM        |
| **TFLite Model Inference**  |                  — | **~14.5 ms** (benchmark) / **~88 ms** (ESP32) |     Neural network execution only     |
| **MFCC Feature Extraction** |                  — |                            **~91 ms** (ESP32) |   63 frames × 40 Mel bins × 13 DCT    |
| **Full DSP + KWS Cycle**    |       < 200 ms hop |                                   **~179 ms** | ✅ Completes within 200 ms window hop |
| **Model Parameters**        |                  — |                                     **4,643** |        Depthwise Separable CNN        |
| **Quantization**            |                  — |                                     Full INT8 |     Input & output INT8 quantized     |

> **Resource Breakdown:**
>
> - The model binary occupies only **13.38 KB of Flash** (5.2% of a standard 256 KB partition), leaving over **94% headroom** for application logic, Wi-Fi networking, Bluetooth stacks, and OTA updates.
> - The tensor arena allocates **20 KB** of internal SRAM, with **~17.1 KB** consumed at peak runtime by TFLite Micro tensors.
> - On physical ESP32 hardware (240 MHz Xtensa), one complete cycle takes **~179 ms** (~91 ms MFCC extraction + ~88 ms TFLite inference), comfortably fitting within the **200 ms** sliding window hop.

---

## 📂 Repository Structure

```text
SIH-2026-Vaani/
├── dataset/
│   ├── audio/                     # 🎙️ Ground-truth audio recordings (6 speakers, 250+ clips)
│   │   ├── Ananya/                #   Positive wake-word clips (Train)
│   │   ├── Ark/                   #   Positive wake-word clips (Train)
│   │   ├── Ishita/                #   Positive wake-word clips (Held-out Validation)
│   │   ├── Mayank/                #   Positive wake-word clips (Train)
│   │   ├── Umang/                 #   Positive wake-word clips (Train)
│   │   └── Vitthal/               #   Positive wake-word clips (🔒 Unseen Test)
│   └── README.md                  #   Dataset provenance, speaker partition policy & guidelines
│
├── docs/
│   ├── data_collection_plan.md    #   Acoustic diversity and protocol specification
│   ├── dataset_prep_arch.txt      #   Data pipeline architectural diagram
│   ├── notebooks/                 #   Exploratory research & experimentation notebooks
│   │   ├── 01_prepare_dataset_v2.ipynb # Interactive dataset prep & spectrogram validation
│   │   └── 02_train_dscnn_v2.ipynb     # Interactive model training & convergence analysis
│   └── reports/                   #   Milestone phase reports & validation audits
│       ├── phase3_report.md       #   Phase 3 baseline model & feature parity report
│       ├── phase4_report.md       #   Phase 4 evaluation & continuous stream audit
│       └── phase5_report.md       #   Phase 5 unseen speaker robustness & V2 benchmark
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
│       │   ├── model_config.h     #   KWS thresholds, buffer sizes & sliding hop parameters
│       │   ├── kws_model.cpp / .h #   TFLite Micro C++ wrapper & inference engine
│       │   └── model_data.cc / .h #   Compiled 13.38 KB INT8 model byte array
│       ├── tools/
│       │   ├── generate_mel_tables.py  # Generates mel_tables.h from model/features.py
│       │   ├── validate_c_vs_python.py # Validates exact numerical feature parity
│       │   ├── verify_c_features.c     # Native C validation harness
│       │   ├── receive_mic.py          # Serial audio capture from ESP32
│       │   ├── test_mic.py             # Quick host microphone sanity test
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
├── server/
│   └── asr/                       # ☁️ STREAMING ASR CLOUD SERVER (Hinglish)
│       ├── app.py                 #   FastAPI WebSocket & REST streaming endpoint
│       ├── engine.py              #   Faster-Whisper ASR engine (GPU/CPU)
│       ├── vad_stream.py          #   Real-time speech-end & silence detector
│       ├── config.py              #   Server, audio & Hinglish prompt parameters
│       ├── mock_esp32_client.py   #   PC mock streamer (live mic & WAV)
│       ├── requirements.txt       #   Server dependencies
│       └── README.md              #   Setup and deployment documentation
│
├── tests/
│   ├── test_model_offline.py      # 🧪 Offline WAV tester replicating firmware feature pipeline
│   ├── test_live_mic.py           # 🎤 Real-time PC microphone wake-word streaming test
│   ├── evaluate_benchmark.py      # 📊 Apples-to-apples V1 vs V2 robustness benchmark
│   ├── threshold_sweep.py         # 📈 FAR/FRR threshold calibration sweep
│   ├── continuous_eval.py         # ⏱️ 10-minute continuous streaming evaluation
│   ├── estimate_resources.py      # 💾 Microcontroller memory & latency estimator
│   └── artifacts/                 #   Benchmark results, sweeps & metadata JSONs
│       ├── continuous_eval.json   #   Continuous streaming metrics & false-trigger analysis
│       ├── dataset_stats.json     #   Dataset duration, sample rate & clip counts
│       ├── onset_diagnostics.json #   Acoustic onset detection diagnostics
│       ├── resource_results.json  #   SRAM, Flash, latency & MAC operations budget
│       ├── speaker_split.json     #   Zero-leakage split verification manifest
│       ├── threshold_sweep.json   #   Confidence threshold sweep data
│       └── v1_vs_v2_results.json  #   Comparative benchmark results
│
├── training/
│   ├── prepare_dataset.py         # 🔄 Normalizes raw audio to standard 16 kHz mono WAV
│   ├── prepare_combined_dataset.py# 🔀 Combines positives, speech commands & noise
│   ├── split_speakers.py          # 🔒 Speaker-disjoint train/val/test partitioning
│   ├── augment.py                 # 🎚️ Noise, volume, pitch & speed augmentations
│   ├── train.py                   # 🚀 Trains Tiny DS-CNN on combined dataset
│   └── quantize.py                # 📦 Full INT8 post-training quantization
│
├── .gitignore                     # Git ignore rules
├── LICENSE                        # MIT License
└── README.md                      # Project documentation
```

### Folder Relevance Matrix

| Directory                | Primary Purpose                     | Key Contents & Roles                                                                                                                                  |
| :----------------------- | :---------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`dataset/`](dataset/)   | **Raw Voice Data & Provenance**     | Speaker-separated positive clips (`Ananya`, `Ark`, `Ishita`, `Mayank`, `Umang`, `Vitthal`), acoustic capture guidelines & zero-leakage speaker policy |
| [`training/`](training/) | **ML Pipeline & Data Prep**         | Strict speaker partitioning, acoustic augmentations (pitch/tempo/noise/RIR), dataset mixing, DS-CNN training & INT8 quantization                      |
| [`model/`](model/)       | **Neural Architecture & Binaries**  | Keras Tiny DS-CNN topology, 40-mel DSP feature extractor, C array exporter (`export_to_c.py`), and pre-trained INT8 TFLite binaries                   |
| [`firmware/`](firmware/) | **Edge ESP-IDF C/C++ Firmware**     | Asymmetric dual-core FreeRTOS engine (Core 0: I2S DMA + VAD gate; Core 1: MFCC DSP + TFLM INT8 inference), and C vs. Python parity verifiers          |
| [`server/`](server/)     | **Cloud ASR Server (Hinglish)**     | Low-latency FastAPI WebSocket server powered by `faster-whisper` (GPU/CPU), server-side streaming VAD, and PC mock streamer                           |
| [`tests/`](tests/)       | **Benchmarking & Validation Suite** | Live mic tester, offline WAV evaluator, continuous stream FAR/FRR stress test, MCU resource profiler, and evaluation JSON artifacts                   |
| [`docs/`](docs/)         | **Documentation & Experimentation** | Phase 3–5 milestone reports, research notebooks (`01_prepare_dataset_v2.ipynb`, `02_train_dscnn_v2.ipynb`), and data collection plans                 |

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

### 5. Launch Cloud ASR Server & Test Audio Streaming

Run the open-source Hinglish ASR server locally on your laptop (with NVIDIA GPU acceleration):

```bash
# 1. Start the ASR streaming server
cd server/asr
python app.py

# 2. In another terminal, test streaming from your PC microphone
python server/asr/mock_esp32_client.py --mic
```

Speak Hinglish voice commands (e.g. _"Vaani, bedroom ki light on kar do"_) and observe real-time transcription with sub-100ms GPU turnaround.

---

## 🔌 Hardware Wiring (INMP441 to ESP32)

| INMP441 Pin    | ESP32 GPIO  | Description                       |
| :------------- | :---------- | :-------------------------------- |
| **VDD / 3V3**  | **3V3**     | 3.3V Power                        |
| **GND**        | **GND**     | Ground                            |
| **SD / DIN**   | **GPIO 22** | Serial Data Input                 |
| **WS / LRCLK** | **GPIO 25** | Word Select (Left/Right Clock)    |
| **SCK / BCK**  | **GPIO 26** | Bit Clock                         |
| **L/R**        | **GND**     | Left Channel Select               |
| **Status LED** | **GPIO 2**  | Built-in / External Indicator LED |

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
