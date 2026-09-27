"""
Phase 5 - Step 7: Train V2 Tiny DS-CNN Wake-Word Model

- Uses the established tiny DS-CNN architecture from Phase 1 / Phase 3.
- Trains on Phase 5 balanced 3-class dataset:
  * "vaani": Train-augmented clips (Ananya, Ark, Umang + any new train speakers)
  * "unknown": Negative words from mini_speech_commands (+ any hard negatives)
  * "silence": Calibrated ambient room noise & silence
- Caches MFCC features to disk (.npz) for instant subsequent iterations.
- Validation: Clean Ishita + validation negatives.
- Test: Clean Unseen Vitthal + test negatives.
- Produces:
  * phase5/artifacts/vaani_v2_best.keras
  * phase5/artifacts/vaani_v2_float32.keras
  * phase5/artifacts/training_metadata.json

Usage:
    # 2-epoch smoke test
    python train_v2.py --smoke_test

    # Full training (35 epochs)
    python train_v2.py --epochs 35
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.features import featurize_file, get_feature_shape  # noqa: E402
from model.architecture import LABELS, NUM_CLASSES, build_tiny_ds_cnn  # noqa: E402

LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}


def load_dataset_features(split_dir: Path, cache_file: Path = None):
    """
    Extracts MFCC features for all files in split_dir.
    Caches features to disk as .npz for fast reloading.
    """
    if cache_file and cache_file.exists():
        print(f"Loading cached features from: {cache_file.name}...")
        data = np.load(cache_file)
        return data["x"], data["y"], list(data["paths"])

    paths, labels = [], []
    for label in LABELS:
        label_dir = split_dir / label
        if not label_dir.exists():
            continue
        for wav in sorted(label_dir.glob("*.wav")):
            paths.append(str(wav))
            labels.append(LABEL_TO_IDX[label])

    print(f"Extracting features for {len(paths)} clips from {split_dir.name}...")
    features = []
    for p in paths:
        feat = featurize_file(p)
        features.append(feat)

    x = np.array(features, dtype=np.float32)  # (N, 63, 13)
    x = np.expand_dims(x, -1)                 # (N, 63, 13, 1)
    y = np.array(labels, dtype=np.int32)

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, x=x, y=y, paths=np.array(paths))
        print(f"Saved feature cache to: {cache_file}")

    return x, y, paths


def train_v2(
    data_dir: Path,
    out_dir: Path,
    epochs: int = 35,
    batch_size: int = 64,
    width: float = 1.0,
    smoke_test: bool = False,
    seed: int = 42,
):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    data_dir = data_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = data_dir / ".feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    tag = "smoke" if smoke_test else "v2"
    actual_epochs = 2 if smoke_test else epochs

    print("=" * 70)
    print(f"PHASE 5 - STEP 7: {'SMOKE TEST' if smoke_test else 'TRAIN MODEL V2'}")
    print("=" * 70)
    print(f"Data Source   : {data_dir}")
    print(f"Artifacts Dir : {out_dir}")
    print(f"Epochs        : {actual_epochs}")
    print(f"Batch Size    : {batch_size}")
    print(f"Model Width   : {width}\n")

    x_train, y_train, train_paths = load_dataset_features(
        data_dir / "training", cache_dir / f"train_feat_{width}.npz"
    )
    x_val, y_val, val_paths = load_dataset_features(
        data_dir / "validation", cache_dir / f"val_feat_{width}.npz"
    )
    x_test, y_test, test_paths = load_dataset_features(
        data_dir / "testing", cache_dir / f"test_feat_{width}.npz"
    )

    print(f"Train Set     : {x_train.shape[0]} samples (feature shape: {x_train.shape[1:]})")
    print(f"Val Set       : {x_val.shape[0]} samples")
    print(f"Test Set      : {x_test.shape[0]} samples\n")

    input_shape = get_feature_shape()
    model = build_tiny_ds_cnn(input_shape, NUM_CLASSES, width=width)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    best_model_path = out_dir / "vaani_v2_best.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(best_model_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=actual_epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks if not smoke_test else [],
    )

    # If best_model.keras exists, load best weights for evaluation
    if not smoke_test and best_model_path.exists():
        eval_model = tf.keras.models.load_model(best_model_path)
    else:
        eval_model = model

    test_loss, test_acc = eval_model.evaluate(x_test, y_test, verbose=0)
    preds = eval_model.predict(x_test, verbose=0)
    y_pred = np.argmax(preds, axis=1)

    print("\n" + "=" * 70)
    print("V2 TEST EVALUATION REPORT:")
    print("=" * 70)
    print(f"Test Accuracy: {test_acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=LABELS))
    print("Confusion Matrix (rows=true, cols=pred):")
    print(LABELS)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Save float32 model
    float32_path = out_dir / "vaani_v2_float32.keras"
    eval_model.save(float32_path)
    print(f"\nSaved model checkpoint to: {float32_path}")

    meta = {
        "smoke_test": smoke_test,
        "epochs_trained": len(history.history["loss"]),
        "total_parameters": int(eval_model.count_params()),
        "train_samples": int(x_train.shape[0]),
        "val_samples": int(x_val.shape[0]),
        "test_samples": int(x_test.shape[0]),
        "final_train_acc": float(history.history["accuracy"][-1]),
        "final_val_acc": float(history.history["val_accuracy"][-1]),
        "test_acc": float(test_acc),
        "test_loss": float(test_loss),
        "confusion_matrix": cm.tolist(),
        "labels": LABELS,
    }

    meta_file = out_dir / f"training_metadata_{tag}.json"
    with open(meta_file, "w") as fp:
        json.dump(meta, fp, indent=2)

    print(f"[SUCCESS] V2 training complete. Metadata saved to: {meta_file}\n")
    return eval_model, meta


def main():
    parser = argparse.ArgumentParser(description="Train V2 Tiny DS-CNN Model")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(ROOT_DIR / "dataset" / "combined"),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(ROOT_DIR / "model" / "checkpoints"),
    )
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--width", type=float, default=1.0)
    parser.add_argument("--smoke_test", action="store_true", help="Run 2-epoch smoke test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_v2(
        Path(args.data_dir),
        Path(args.out_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        width=args.width,
        smoke_test=args.smoke_test,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
