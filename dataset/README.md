# Vaani Wake-Word Audio Dataset

This directory contains the original human voice recordings collected for the **Vaani** wake-word detection project.

## Structure

```text
dataset/
├── audio/
│   ├── Ananya/      # Positive wake-word recordings
│   ├── Ark/         # Positive wake-word recordings (extended leading silence)
│   ├── Ishita/      # Positive wake-word recordings (validation speaker)
│   ├── Mayank/      # Positive wake-word recordings
│   ├── Umang/       # Positive wake-word recordings
│   └── Vitthal/     # Positive wake-word recordings (unseen test speaker)
└── README.md
```

## Acoustic Specifications

* **Target Keyword:** "Vaani" (वाणी)
* **Sampling Rate:** Recorded at various native phone mic rates; normalized to **16,000 Hz Mono, 16-bit PCM** by the preprocessing pipeline.
* **Speaker Partitioning (Zero-Leakage Policy):**
  * **Training Split:** `Ananya`, `Ark`, `Umang`, `Mayank`
  * **Validation Split (Strictly held out):** `Ishita`
  * **Test Split (Strictly held out unseen):** `Vitthal`
