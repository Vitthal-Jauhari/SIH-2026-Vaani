"""
Phase 5 - Step 5: Targeted Training-Only Acoustic Augmentations

Directly addresses Phase 4 empirical failure modes:
1. Timing Robustness:
   - Positives generated across diverse positions in 1.0s window:
     * Little/no leading silence (onset near 0.0s - 0.05s)
     * Moderate leading silence (onset ~0.20s - 0.30s)
     * Substantial leading silence (onset ~0.40s - 0.50s)
     * Natural timing
2. Volume Robustness:
   - Attenuation (-12 dB for quiet speech, -6 dB, +6 dB loud speech)
3. Speed / Tempo Robustness:
   - Tempo stretches (0.85x, 0.92x, 1.08x, 1.15x)
4. Noise Robustness:
   - Low-frequency traffic rumble (low-pass filtered engine rumble), typing, babble, pink noise
   - Across SNR levels: 25 dB, 20 dB, 15 dB, 10 dB, 5 dB, 0 dB SNR

STRICT RULE:
Applied EXCLUSIVELY to train partition.
Validation (Ishita) and Unseen Test (Vitthal) are strictly excluded and NEVER augmented.
"""

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import soundfile as sf

TARGET_SAMPLE_RATE = 16000
CLIP_SECONDS = 1.0
CLIP_LEN = int(TARGET_SAMPLE_RATE * CLIP_SECONDS)
ROOT_DIR = Path(__file__).resolve().parents[1]


def fit_exact_1s(audio: np.ndarray, target_len: int = CLIP_LEN) -> np.ndarray:
    """Pads or truncates audio to exactly target_len samples."""
    if len(audio) < target_len:
        return np.pad(audio, (0, target_len - len(audio)), mode="constant")
    return audio[:target_len]


def extract_timing_windows(
    audio: np.ndarray,
    sr: int = TARGET_SAMPLE_RATE,
    target_len: int = CLIP_LEN,
) -> List[Tuple[str, np.ndarray]]:
    """
    Extracts multiple 1.0s windows with distinct leading/trailing silence profiles:
    - little_lead: speech placed near the start of the 1s clip
    - moderate_lead: speech placed centrally
    - substantial_lead: speech placed towards the end of the 1s clip
    - natural: standard 1s crop from start
    """
    windows = []
    total_len = len(audio)

    # 1. Natural crop (from start, padded if < 1s)
    natural_clip = fit_exact_1s(audio, target_len)
    windows.append(("natural", natural_clip))

    if total_len <= target_len:
        # If the whole recording is already <= 1.0s, generate shifts via padding
        speech_dur = total_len
        slack = target_len - speech_dur
        if slack > 800:  # > 50ms slack
            # Little lead
            lead_little = int(sr * 0.02)
            c1 = np.pad(audio, (lead_little, target_len - speech_dur - lead_little), mode="constant")
            windows.append(("little_lead", c1[:target_len]))
            # Moderate lead
            lead_mod = slack // 2
            c2 = np.pad(audio, (lead_mod, target_len - speech_dur - lead_mod), mode="constant")
            windows.append(("mod_lead", c2[:target_len]))
            # Substantial lead
            lead_sub = max(0, slack - int(sr * 0.03))
            c3 = np.pad(audio, (lead_sub, target_len - speech_dur - lead_sub), mode="constant")
            windows.append(("sub_lead", c3[:target_len]))
        return windows

    # Find highest energy segment (approximate speech center)
    frame_size = int(sr * 0.05)  # 50ms
    n_frames = (total_len - frame_size) // (frame_size // 2)
    energies = []
    for f in range(n_frames):
        s = f * (frame_size // 2)
        energies.append((np.sum(audio[s : s + frame_size] ** 2), s))

    if not energies:
        return windows

    # Peak energy frame
    energies.sort(key=lambda x: x[0], reverse=True)
    peak_sample = energies[0][1] + frame_size // 2

    # A: Moderate lead (center peak around sample 8000, i.e. 0.5s)
    start_mod = max(0, min(total_len - target_len, peak_sample - target_len // 2))
    mod_clip = fit_exact_1s(audio[start_mod : start_mod + target_len], target_len)
    windows.append(("mod_lead", mod_clip))

    # B: Little lead (peak early, around sample 4000, i.e. 0.25s)
    start_little = max(0, min(total_len - target_len, peak_sample - int(target_len * 0.25)))
    little_clip = fit_exact_1s(audio[start_little : start_little + target_len], target_len)
    windows.append(("little_lead", little_clip))

    # C: Substantial lead (peak late, around sample 12000, i.e. 0.75s)
    start_sub = max(0, min(total_len - target_len, peak_sample - int(target_len * 0.75)))
    sub_clip = fit_exact_1s(audio[start_sub : start_sub + target_len], target_len)
    windows.append(("sub_lead", sub_clip))

    return windows


def apply_volume(audio: np.ndarray, db_gain: float) -> np.ndarray:
    factor = 10.0 ** (db_gain / 20.0)
    return np.clip(audio * factor, -1.0, 1.0).astype(np.float32)


def apply_speed(audio: np.ndarray, speed_factor: float) -> np.ndarray:
    stretched = librosa.effects.time_stretch(audio, rate=speed_factor)
    return fit_exact_1s(stretched, CLIP_LEN)


def generate_noise(noise_type: str, length: int) -> np.ndarray:
    if noise_type == "traffic":
        # White noise low-pass filtered to simulate deep engine and tire rumble (< 600 Hz)
        white = np.random.normal(0, 1, length)
        # 1st-order IIR low-pass filter
        alpha = 0.15
        rumble = np.zeros(length, dtype=np.float32)
        for i in range(1, length):
            rumble[i] = alpha * white[i] + (1 - alpha) * rumble[i - 1]
        # Normalize
        peak = np.max(np.abs(rumble)) or 1.0
        return (rumble / peak).astype(np.float32)
    elif noise_type == "pink":
        # Approximate 1/f pink noise with multi-pole IIR filter
        white = np.random.normal(0, 1, length)
        b0, b1, b2 = 0.049922035, -0.095993537, 0.050612699
        a1, a2 = -1.74201445, 0.7490076
        from scipy.signal import lfilter
        pink = lfilter([b0, b1, b2], [1.0, a1, a2], white)
        peak = np.max(np.abs(pink)) or 1.0
        return (pink / peak).astype(np.float32)
    elif noise_type == "babble":
        # Multi-frequency harmonic babble simulation
        t = np.linspace(0, 1.0, length, endpoint=False)
        babble = np.zeros(length, dtype=np.float32)
        for freq in (150, 220, 310, 480, 700, 1100, 1800):
            phase = np.random.uniform(0, 2 * np.pi)
            mod = 0.5 + 0.5 * np.sin(2 * np.pi * np.random.uniform(2, 6) * t)
            babble += mod * np.sin(2 * np.pi * freq * t + phase)
        peak = np.max(np.abs(babble)) or 1.0
        return (babble / peak).astype(np.float32)
    else:
        # Default ambient room noise
        white = np.random.normal(0, 1, length)
        t = np.linspace(0, 1.0, length, endpoint=False)
        hum = 0.3 * np.sin(2 * np.pi * 50 * t)
        noise = white + hum
        peak = np.max(np.abs(noise)) or 1.0
        return (noise / peak).astype(np.float32)


def apply_noise(audio: np.ndarray, snr_db: float, noise_type: str = "traffic") -> np.ndarray:
    signal_rms = np.sqrt(np.mean(audio ** 2))
    if signal_rms < 1e-6:
        signal_rms = 1e-6
    desired_noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
    noise = generate_noise(noise_type, len(audio))
    noise_rms = np.sqrt(np.mean(noise ** 2)) or 1e-6
    scaled_noise = noise * (desired_noise_rms / noise_rms)
    return np.clip(audio + scaled_noise, -1.0, 1.0).astype(np.float32)


def augment_training_set(
    splits_dir: Path,
    out_dir: Path,
    held_out_speakers: List[str],
    val_speakers: List[str],
    seed: int = 42,
):
    random.seed(seed)
    np.random.seed(seed)

    splits_dir = splits_dir.resolve()
    out_dir = out_dir.resolve()
    train_dir = splits_dir / "train"

    held_out_set = set(held_out_speakers or ["Vitthal"])
    val_set = set(val_speakers or ["Ishita"])

    print("=" * 70)
    print("PHASE 5 - STEP 5: TARGETED TRAINING-ONLY AUGMENTATIONS")
    print("=" * 70)
    print(f"Source Train Dir   : {train_dir}")
    print(f"Output Aug Dir     : {out_dir}")
    print(f"Protected Val Spkrs: {sorted(val_set)} (NEVER AUGMENTED)")
    print(f"Protected Test Spkr: {sorted(held_out_set)} (NEVER AUGMENTED)\n")

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    # Guardrail check
    for spk_dir in train_dir.iterdir():
        if spk_dir.name in held_out_set:
            raise RuntimeError(f"FATAL: Unseen test speaker '{spk_dir.name}' found in train directory!")
        if spk_dir.name in val_set:
            raise RuntimeError(f"FATAL: Validation speaker '{spk_dir.name}' found in train directory!")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_source = 0
    total_augmented = 0
    speaker_aug_counts = {}

    for spk_dir in sorted(train_dir.iterdir()):
        if not spk_dir.is_dir():
            continue
        spk = spk_dir.name
        dest_spk_dir = out_dir / spk
        dest_spk_dir.mkdir(parents=True, exist_ok=True)

        wavs = sorted(spk_dir.glob("*.wav"))
        total_source += len(wavs)
        n_spk_out = 0

        for w in wavs:
            audio, sr = sf.read(str(w))
            base_name = w.stem

            # 1. Extract timing windows (natural, little_lead, mod_lead, sub_lead)
            windows = extract_timing_windows(audio, sr, CLIP_LEN)

            # Save clean windows
            for tag, win_audio in windows:
                fname = f"{base_name}_{tag}.wav"
                sf.write(str(dest_spk_dir / fname), win_audio, sr, subtype="PCM_16")
                n_spk_out += 1

            # Select primary window (mod_lead or natural) for acoustic perturbations
            primary_audio = windows[1][1] if len(windows) > 1 else windows[0][1]

            # 2. Quiet speech volume perturbations (-12 dB, -6 dB, +6 dB)
            for db in (-12.0, -6.0, 6.0):
                v_audio = apply_volume(primary_audio, db)
                fname = f"{base_name}_vol_{db:+.0f}dB.wav"
                sf.write(str(dest_spk_dir / fname), v_audio, sr, subtype="PCM_16")
                n_spk_out += 1

            # 3. Speed / Tempo perturbations (0.85x, 0.92x, 1.08x, 1.15x)
            for speed in (0.85, 0.92, 1.08, 1.15):
                s_audio = apply_speed(primary_audio, speed)
                fname = f"{base_name}_spd_{speed:.2f}x.wav"
                sf.write(str(dest_spk_dir / fname), s_audio, sr, subtype="PCM_16")
                n_spk_out += 1

            # 4. Traffic & noise perturbations (0 dB, 5 dB, 10 dB, 15 dB, 20 dB SNR)
            for noise_t in ("traffic", "babble", "pink"):
                for snr in (0.0, 5.0, 10.0, 20.0):
                    n_audio = apply_noise(primary_audio, snr_db=snr, noise_type=noise_t)
                    fname = f"{base_name}_noise_{noise_t}_{snr:.0f}dB.wav"
                    sf.write(str(dest_spk_dir / fname), n_audio, sr, subtype="PCM_16")
                    n_spk_out += 1

        speaker_aug_counts[spk] = n_spk_out
        total_augmented += n_spk_out
        print(f"  Speaker {spk:10s} : {len(wavs):3d} source clips -> {n_spk_out:5d} targeted clips")

    print("\nAUGMENTATION SUMMARY:")
    print(f"  Source Train Clips : {total_source}")
    print(f"  Total Augmented    : {total_augmented} clips")
    print(f"  Multiplication     : {total_augmented / total_source:.1f}x")
    print(f"  Destination        : {out_dir}\n")

    return total_augmented


def main():
    parser = argparse.ArgumentParser(description="Phase 5 Training Augmentation")
    parser.add_argument(
        "--splits_dir",
        type=str,
        default=str(ROOT_DIR / "dataset" / "splits"),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(ROOT_DIR / "dataset" / "train_augmented"),
    )
    parser.add_argument(
        "--held_out_speakers",
        nargs="+",
        default=["Vitthal"],
    )
    parser.add_argument(
        "--val_speakers",
        nargs="+",
        default=["Ishita"],
    )
    args = parser.parse_args()

    augment_training_set(
        Path(args.splits_dir),
        Path(args.out_dir),
        held_out_speakers=args.held_out_speakers,
        val_speakers=args.val_speakers,
    )


if __name__ == "__main__":
    main()
