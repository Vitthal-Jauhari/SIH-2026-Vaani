# Phase 4: Targeted Data Collection Plan

## 1. Overview
Based on empirical stress-testing of the Phase 3 INT8 TFLite model (`vaani_int8.tflite`) across 6 noise profiles, 4 volume levels, 5 tempo rates, and continuous sliding-window streaming, this document outlines the **targeted future data collection requirements**.

Rather than collecting arbitrary recordings, future collection must address the specific failure modes identified during Phase 4 evaluation.

---

## 2. Priority Ranking of Data Gaps

| Priority | Data Gap Category | Empirical Failure Mode Observed in Phase 4 | Target Future Collection |
| :---: | :--- | :--- | :--- |
| **P1** | **Continuous Conversational Speech (Negatives)** | In continuous streaming, rolling 100ms windows over non-target speech yielded high false trigger rates (~570 FP/hr in dense dialogue). | 5–10 hours of multi-speaker conversational podcast audio, radio broadcasts, and group room chatter divided into overlapping 1-second rolling cuts. |
| **P2** | **Phonetically Adjacent Negative Words (Near-Misses)** | Model was trained on English Speech Commands ("stop", "go", "yes"). High vulnerability to common Hindi/Indian rhyming words. | 500+ recordings of phonetically similar words: **"Pani"**, **"Rani"**, **"Mani"**, **"Dhaani"**, **"Gyaani"**, **"Baani"**, **"Kahani"**. |
| **P3** | **Low-Frequency Traffic / Vehicular Noise** | At 0 dB SNR traffic noise, unseen wake-word recall dropped from 100.0% to **50.0%** (and to 73.1% at 5 dB SNR). | Positive Vaani recordings in transit environments (inside cars, bus stops, road-facing rooms) and negative vehicular background tracks. |
| **P4** | **Far-Field Physical Distance Audio (1m – 3m)** | Baseline recordings are 100% near-field smartphone audio (0.2–0.3m). Real ESP32 smart home devices operate at 1m to 3m. | Calibrated physical distance recordings: 15–20 reps per speaker recorded at **0.5m**, **1.0m**, **1.5m**, **2.0m**, and **3.0m** across at least 3 distinct rooms. |
| **P5** | **Low-Amplitude & Whispered Vaani** | Recall dropped to **61.5%** on Vitthal and **47.6%** on Ishita when speech volume was attenuated to -12 dB. | Prompted collection of quiet, whispered, and relaxed/casual pronunciations of "Vaani". |
| **P6** | **Additional Diverse Speaker Demographics** | Current model trained on only 3 speakers (Ananya, Ark, Umang). Although Vitthal (unseen) succeeded at 84.6%, variance across accents/ages remains untested. | 10+ new speakers representing wider age groups, regional Indian accents, and child voices. |

---

## 3. Detailed Data Collection Protocols

### Protocol A: Phonetically Near-Miss Negative Dataset
- **Objective**: Harden the decision boundary against acoustic false triggers.
- **Word List**:
  1. *Pani* (Water) — identical vowel/consonant trajectory except initial plosive /p/ vs /v/.
  2. *Rani* (Queen) — rhyming /r/ liquid onset.
  3. *Mani* (Gem) — nasal bilabial onset.
  4. *Dhaani* (Light green / grain) — dental aspirated plosive onset.
  5. *Kahani* (Story) — tri-syllabic ending in -aani.
  6. *Gyaani* (Wise) — palatal nasal cluster.
- **Quantity**: 50 repetitions per word across at least 6 speakers = ~1,800 recordings.

### Protocol B: Far-Field Room Acoustic Dataset
- **Objective**: Capture real room reverberation (RT60 decay) and distance attenuation.
- **Setup**:
  - Device: I2S MEMS microphone (e.g. INMP441 or SPH0645) connected to development board / audio recorder.
  - Distances: Marked markers at 0.5m, 1.0m, 2.0m, 3.0m.
  - Angles: On-axis (0 degrees) and off-axis (45 degrees, 90 degrees).
  - Rooms: Living room (furnished), tiled room (echoic), office space.
- **Quantity**: 20 repetitions per speaker at each distance = ~400 recordings.

### Protocol C: Transit & High-Noise Environment Dataset
- **Objective**: Prevent wake-word dropouts in urban and vehicular environments.
- **Scenarios**:
  - In-cabin car audio with AC fan on medium/high.
  - Outdoor street ambiance.
  - Domestic kitchen environment with exhaust fan / running water.
- **Quantity**: 30 repetitions per speaker mixed with natural acoustic noise.

---

## 4. Next Training Iteration (Phase 5 Plan)
Once Protocols A, B, and C are collected:
1. Retain the **13.39 KB Tiny DS-CNN** topology (do not enlarge parameters).
2. Integrate near-miss phonemes into the `unknown` class pool.
3. Incorporate sliding-window negative training loss.
4. Export updated INT8 model for physical ESP32 flash validation.
