"""
Phase 5 - Step 3: Dataset Preparation & Audio Normalization with Timing Robustness

- Normalizes raw incoming audio (from Phase 3 and any new Phase 5 recordings)
  to standard: 16 kHz, mono, 16-bit signed PCM WAV.
- Preserves natural leading/trailing silence in source recordings (no automatic centering
  on evaluation or test data).
- Provides an energy-based onset detector as an ANALYSIS / DIAGNOSTIC TOOL to measure
  speech timing and diagnose alignment without silently modifying test data.
- Prepares normalized positive speakers and normalizes any provided hard negatives / background clips.

Usage:
    python prepare_dataset.py
    python prepare_dataset.py --analyze_onsets
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import soundfile as sf

try:
    import av
except ImportError:
    av = None

TARGET_SAMPLE_RATE = 16000
CLIP_SECONDS = 1.0
CLIP_LEN = int(TARGET_SAMPLE_RATE * CLIP_SECONDS)
ROOT_DIR = Path(__file__).resolve().parents[1]


def decode_audio_robust(file_path: Path) -> Tuple[np.ndarray, int]:
    """
    Decodes an audio file of any format (m4a, mp3, aac, wav, raw) using PyAV or soundfile,
    resampling to 16 kHz mono float32 in range [-1.0, 1.0].
    """
    if av is not None:
        try:
            container = av.open(str(file_path))
            audio_stream = next((s for s in container.streams if s.type == "audio"), None)
            if not audio_stream:
                container.close()
                raise ValueError("No audio stream found")

            resampler = av.AudioResampler(format="fltp", layout="mono", rate=TARGET_SAMPLE_RATE)
            frames = []
            for frame in container.decode(audio_stream):
                frame.pts = None
                resampled = resampler.resample(frame)
                if resampled:
                    for rf in resampled:
                        frames.append(rf.to_ndarray())
            container.close()

            if frames:
                audio = np.concatenate(frames, axis=-1).flatten().astype(np.float32)
                return audio, TARGET_SAMPLE_RATE
        except Exception:
            pass  # Fall back to soundfile

    # Fallback using soundfile / librosa
    import librosa
    audio, sr = librosa.load(str(file_path), sr=TARGET_SAMPLE_RATE, mono=True)
    return audio.astype(np.float32), TARGET_SAMPLE_RATE


def detect_speech_onset(
    audio: np.ndarray,
    sr: int = TARGET_SAMPLE_RATE,
    frame_ms: float = 20.0,
    hop_ms: float = 10.0,
    energy_threshold_rel: float = 0.05,
) -> Dict[str, Any]:
    """
    DIAGNOSTIC / ANALYSIS TOOL ONLY:
    Measures speech onset, offset, peak energy, and speech duration without modifying audio.
    """
    frame_len = int(sr * frame_ms / 1000.0)
    hop_len = int(sr * hop_ms / 1000.0)

    if len(audio) < frame_len:
        return {
            "duration_sec": len(audio) / sr,
            "onset_sec": 0.0,
            "offset_sec": len(audio) / sr,
            "speech_len_sec": len(audio) / sr,
            "leading_silence_sec": 0.0,
            "trailing_silence_sec": 0.0,
            "max_frame_energy": 0.0,
        }

    # Calculate short-time frame energy
    n_frames = 1 + (len(audio) - frame_len) // hop_len
    energies = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_len
        seg = audio[start : start + frame_len]
        energies[i] = np.sum(seg ** 2) / frame_len

    max_e = np.max(energies) if len(energies) > 0 else 0.0
    thresh = max_e * energy_threshold_rel

    active_frames = np.where(energies >= thresh)[0]
    if len(active_frames) == 0:
        onset_sec = 0.0
        offset_sec = len(audio) / sr
    else:
        onset_frame = active_frames[0]
        offset_frame = active_frames[-1]
        onset_sec = (onset_frame * hop_len) / sr
        offset_sec = min(len(audio) / sr, ((offset_frame * hop_len) + frame_len) / sr)

    duration_sec = len(audio) / sr
    return {
        "duration_sec": round(duration_sec, 4),
        "onset_sec": round(onset_sec, 4),
        "offset_sec": round(offset_sec, 4),
        "speech_len_sec": round(max(0.0, offset_sec - onset_sec), 4),
        "leading_silence_sec": round(onset_sec, 4),
        "trailing_silence_sec": round(max(0.0, duration_sec - offset_sec), 4),
        "max_frame_energy": float(max_e),
    }


def normalize_directory(
    in_dir: Path,
    out_dir: Path,
    label_prefix: str = "",
    preserve_natural_timing: bool = True,
) -> int:
    """
    Normalizes all audio files in in_dir recursively to 16 kHz mono 16-bit PCM WAV.
    Preserves natural timing.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    extensions = {".wav", ".mp3", ".m4a", ".ma4", ".aac", ".flac", ".ogg", ".10", ".15", ".2"}

    for p in sorted(in_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in extensions and not any(p.name.endswith(ext) for ext in extensions):
            continue

        try:
            audio, sr = decode_audio_robust(p)
            if len(audio) == 0:
                continue

            # Target filename
            rel = p.relative_to(in_dir)
            target_file = out_dir / rel.parent / f"{p.stem}.wav"
            target_file.parent.mkdir(parents=True, exist_ok=True)

            # Export strictly as 16-bit PCM WAV
            sf.write(str(target_file), audio, sr, subtype="PCM_16")
            count += 1
        except Exception as e:
            print(f"Warning: Failed to decode {p}: {e}")

    return count


def prepare_dataset(
    audio_dir: Path,
    out_dir: Path,
    meta_dir: Path,
    analyze_onsets: bool = False,
) -> Dict[str, Any]:
    print("=" * 70)
    print("DATASET PREPARATION & AUDIO NORMALIZATION")
    print("=" * 70)
    print(f"Source Audio Dir: {audio_dir}")
    print(f"Normalized Out  : {out_dir}")
    print(f"Metadata Dir    : {meta_dir}\n")

    norm_pos = out_dir / "positives"
    norm_pos.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "speakers": {},
        "hard_negatives": {},
        "background": {},
        "total_positive_clips": 0,
        "total_hard_negative_clips": 0,
        "total_background_clips": 0,
    }

    # 1. Normalize Raw Positive Speakers
    if audio_dir.exists():
        print(f"[1/3] Normalizing Raw Audio from: {audio_dir}")
        for spk_dir in sorted(audio_dir.iterdir()):
            if not spk_dir.is_dir() or spk_dir.name.startswith("."):
                continue
            spk_name = spk_dir.name
            dest_spk = norm_pos / spk_name
            n_norm = normalize_directory(spk_dir, dest_spk)
            stats["speakers"][spk_name] = n_norm
            print(f"  - Normalized Speaker '{spk_name}': {n_norm} recordings")
    else:
        print(f"Warning: Audio directory not found at {audio_dir}")

    stats["total_positive_clips"] = sum(stats["speakers"].values())
    print(f"  Total Positive Vaani Clips: {stats['total_positive_clips']}")

    # 2. Ingest Hard Negatives if provided
    hard_neg_dir = audio_dir.parent / "hard_negatives"
    if hard_neg_dir.exists():
        print(f"\n[2/3] Normalizing Hard Negatives from: {hard_neg_dir}")
        for cat_dir in sorted(hard_neg_dir.iterdir()):
            if not cat_dir.is_dir() or cat_dir.name.startswith("."):
                continue
            cat_name = cat_dir.name
            dest_cat = out_dir / "hard_negatives" / cat_name
            n_norm = normalize_directory(cat_dir, dest_cat)
            stats["hard_negatives"][cat_name] = n_norm
            if n_norm > 0:
                print(f"  - Hard Negative '{cat_name}': {n_norm} recordings")
    stats["total_hard_negative_clips"] = sum(stats["hard_negatives"].values())

    # 3. Ingest Background Audio if provided
    bg_dir = audio_dir.parent / "background"
    if bg_dir.exists():
        print(f"\n[3/3] Normalizing Background Audio from: {bg_dir}")
        for b_dir in sorted(bg_dir.iterdir()):
            if not b_dir.is_dir() or b_dir.name.startswith("."):
                continue
            bg_name = b_dir.name
            dest_bg = out_dir / "background" / bg_name
            n_norm = normalize_directory(b_dir, dest_bg)
            stats["background"][bg_name] = n_norm
            if n_norm > 0:
                print(f"  - Background '{bg_name}': {n_norm} recordings")
    stats["total_background_clips"] = sum(stats["background"].values())

    # Save summary stats
    out_stats_file = meta_dir / "dataset_stats.json"
    with open(out_stats_file, "w") as fp:
        json.dump(stats, fp, indent=2)
    print(f"\nSaved preparation stats to: {out_stats_file}")

    # 4. Diagnostic Speech Onset Analysis (Tool Only)
    if analyze_onsets:
        print("\n" + "=" * 70)
        print("DIAGNOSTIC ONSET ANALYSIS (ANALYSIS TOOL ONLY)")
        print("=" * 70)
        onset_reports = {}
        for spk, count in stats["speakers"].items():
            spk_dir = norm_pos / spk
            wavs = list(spk_dir.glob("*.wav"))
            leads = []
            durations = []
            for w in wavs:
                audio, sr = sf.read(str(w))
                diag = detect_speech_onset(audio, sr)
                leads.append(diag["leading_silence_sec"])
                durations.append(diag["duration_sec"])
            mean_lead = np.mean(leads) if leads else 0.0
            max_lead = np.max(leads) if leads else 0.0
            mean_dur = np.mean(durations) if durations else 0.0
            onset_reports[spk] = {
                "count": len(wavs),
                "mean_duration_sec": round(float(mean_dur), 3),
                "mean_leading_silence_sec": round(float(mean_lead), 3),
                "max_leading_silence_sec": round(float(max_lead), 3),
            }
            print(f"  Speaker {spk:10s} | Clips: {len(wavs):3d} | Mean Duration: {mean_dur:.2f}s | "
                  f"Mean Leading Silence: {mean_lead:.2f}s | Max Lead: {max_lead:.2f}s")

        diag_path = meta_dir / "onset_diagnostics.json"
        with open(diag_path, "w") as fp:
            json.dump(onset_reports, fp, indent=2)
        print(f"Diagnostic onset analysis saved to: {diag_path}\n")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Dataset Preparation & Audio Normalization")
    parser.add_argument(
        "--audio_dir",
        type=str,
        default=str(ROOT_DIR / "dataset" / "audio"),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(ROOT_DIR / "dataset" / "normalized"),
    )
    parser.add_argument(
        "--meta_dir",
        type=str,
        default=str(ROOT_DIR / "dataset" / "metadata"),
    )
    parser.add_argument(
        "--analyze_onsets",
        action="store_true",
        help="Run diagnostic onset measurement tool on all positive speakers",
    )
    args = parser.parse_args()

    prepare_dataset(
        Path(args.audio_dir),
        Path(args.out_dir),
        Path(args.meta_dir),
        analyze_onsets=args.analyze_onsets,
    )


if __name__ == "__main__":
    main()
