"""
Phase 5 - Step 4: Speaker-Level Split (Strictly Before Augmentation)

Splits normalized recordings by SPEAKER into:
- phase5/data/splits/train/<speaker>/
- phase5/data/splits/validation/<speaker>/
- phase5/data/splits/test_unseen/<speaker>/

Strict Zero-Leakage Guardrail:
- Explicitly verifies that no speaker appears in more than one partition.
- Immediately halts execution if speaker leakage is detected.
- Saves machine-readable split verification manifest to:
  phase5/artifacts/speaker_split.json
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import List, Set

ROOT_DIR = Path(__file__).resolve().parents[1]


def link_or_copy(src: Path, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    try:
        os.link(src.resolve(), dest)
    except OSError:
        shutil.copy(src, dest)


def split_speakers(
    norm_positives_dir: Path,
    splits_out_dir: Path,
    artifacts_dir: Path,
    train_speakers: List[str],
    val_speakers: List[str],
    held_out_speakers: List[str],
):
    norm_positives_dir = norm_positives_dir.resolve()
    splits_out_dir = splits_out_dir.resolve()
    artifacts_dir = artifacts_dir.resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_set: Set[str] = set(train_speakers)
    val_set: Set[str] = set(val_speakers)
    held_out_set: Set[str] = set(held_out_speakers)

    print("=" * 70)
    print("PHASE 5 - STEP 4: SPEAKER-LEVEL SPLIT & ZERO-LEAKAGE VERIFICATION")
    print("=" * 70)
    print(f"Source Positives   : {norm_positives_dir}")
    print(f"Splits Output Dir  : {splits_out_dir}")
    print(f"Train Speakers     : {sorted(train_set)}")
    print(f"Validation Speakers: {sorted(val_set)}")
    print(f"Unseen Test Spkrs  : {sorted(held_out_set)}\n")

    # Guardrail 1: Disjoint configuration check
    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & held_out_set
    val_test_overlap = val_set & held_out_set

    if train_val_overlap or train_test_overlap or val_test_overlap:
        msg = (
            f"FATAL ERROR: Speaker configuration is not disjoint!\n"
            f"Train/Val Overlap: {train_val_overlap}\n"
            f"Train/Test Overlap: {train_test_overlap}\n"
            f"Val/Test Overlap: {val_test_overlap}\n"
            f"Phase 5 pipeline aborted to prevent data leakage."
        )
        print(msg, file=sys.stderr)
        raise RuntimeError(msg)

    # Clean previous splits
    if splits_out_dir.exists():
        shutil.rmtree(splits_out_dir)

    all_found_speakers = {d.name for d in norm_positives_dir.iterdir() if d.is_dir()}
    print(f"Discovered positive speakers in normalized data: {sorted(all_found_speakers)}")

    # Assign any newly discovered speakers not explicitly listed to train by default
    unassigned = all_found_speakers - (train_set | val_set | held_out_set)
    if unassigned:
        print(f"Assigning new unassigned speakers to Train split: {sorted(unassigned)}")
        train_set.update(unassigned)

    split_stats = {"train": {}, "validation": {}, "test_unseen": {}}

    # Ingest Train
    for spk in sorted(train_set):
        spk_dir = norm_positives_dir / spk
        if not spk_dir.exists():
            print(f"Note: Train speaker '{spk}' directory not found, skipping.")
            continue
        dest_dir = splits_out_dir / "train" / spk
        wavs = list(spk_dir.glob("*.wav"))
        for w in wavs:
            link_or_copy(w, dest_dir / w.name)
        split_stats["train"][spk] = len(wavs)

    # Ingest Validation
    for spk in sorted(val_set):
        spk_dir = norm_positives_dir / spk
        if not spk_dir.exists():
            print(f"Note: Val speaker '{spk}' directory not found, skipping.")
            continue
        dest_dir = splits_out_dir / "validation" / spk
        wavs = list(spk_dir.glob("*.wav"))
        for w in wavs:
            link_or_copy(w, dest_dir / w.name)
        split_stats["validation"][spk] = len(wavs)

    # Ingest Held-out Unseen Test
    for spk in sorted(held_out_set):
        spk_dir = norm_positives_dir / spk
        if not spk_dir.exists():
            raise FileNotFoundError(f"Unseen test speaker '{spk}' directory not found at: {spk_dir}")
        dest_dir = splits_out_dir / "test_unseen" / spk
        wavs = list(spk_dir.glob("*.wav"))
        for w in wavs:
            link_or_copy(w, dest_dir / w.name)
        split_stats["test_unseen"][spk] = len(wavs)

    # Guardrail 2: Verify physical directory isolation
    physical_train = {d.name for d in (splits_out_dir / "train").iterdir() if d.is_dir()} if (splits_out_dir / "train").exists() else set()
    physical_val = {d.name for d in (splits_out_dir / "validation").iterdir() if d.is_dir()} if (splits_out_dir / "validation").exists() else set()
    physical_test = {d.name for d in (splits_out_dir / "test_unseen").iterdir() if d.is_dir()} if (splits_out_dir / "test_unseen").exists() else set()

    for spk in physical_test:
        if spk in physical_train:
            msg = f"FATAL LEAKAGE DETECTED: Unseen test speaker '{spk}' found in train directory!"
            print(msg, file=sys.stderr)
            raise RuntimeError(msg)
        if spk in physical_val:
            msg = f"FATAL LEAKAGE DETECTED: Unseen test speaker '{spk}' found in validation directory!"
            print(msg, file=sys.stderr)
            raise RuntimeError(msg)

    split_manifest = {
        "verified_zero_leakage": True,
        "train_speakers": sorted(physical_train),
        "validation_speakers": sorted(physical_val),
        "unseen_test_speakers": sorted(physical_test),
        "counts": split_stats,
        "total_train_clips": sum(split_stats["train"].values()),
        "total_val_clips": sum(split_stats["validation"].values()),
        "total_test_clips": sum(split_stats["test_unseen"].values()),
    }

    manifest_path = artifacts_dir / "speaker_split.json"
    with open(manifest_path, "w") as fp:
        json.dump(split_manifest, fp, indent=2)

    print("\nSPLIT SUMMARY:")
    print(f"  Train Clipse        : {split_manifest['total_train_clips']} across {split_manifest['train_speakers']}")
    print(f"  Validation Clips    : {split_manifest['total_val_clips']} across {split_manifest['validation_speakers']}")
    print(f"  Unseen Test Clips   : {split_manifest['total_test_clips']} across {split_manifest['unseen_test_speakers']}")
    print(f"\n[PASSED] Zero-leakage verified. Manifest written to: {manifest_path}\n")

    return split_manifest


def main():
    parser = argparse.ArgumentParser(description="Phase 5 Speaker Split")
    parser.add_argument(
        "--norm_dir",
        type=str,
        default=str(ROOT_DIR / "dataset" / "normalized" / "positives"),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(ROOT_DIR / "dataset" / "splits"),
    )
    parser.add_argument(
        "--artifacts_dir",
        type=str,
        default=str(ROOT_DIR / "dataset" / "metadata"),
    )
    parser.add_argument(
        "--train_speakers",
        nargs="+",
        default=["Ananya", "Ark", "Umang"],
    )
    parser.add_argument(
        "--val_speakers",
        nargs="+",
        default=["Ishita"],
    )
    parser.add_argument(
        "--held_out_speakers",
        nargs="+",
        default=["Vitthal"],
    )
    args = parser.parse_args()

    split_speakers(
        Path(args.norm_dir),
        Path(args.out_dir),
        Path(args.artifacts_dir),
        train_speakers=args.train_speakers,
        val_speakers=args.val_speakers,
        held_out_speakers=args.held_out_speakers,
    )


if __name__ == "__main__":
    main()
