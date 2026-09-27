"""
Phase 1 - Step 2: Audio -> MFCC feature extraction.

Produces a fixed-size 2D feature map (time x mel-bins) per 1-second clip,
suitable as input to a tiny CNN. Designed to mirror what you'll eventually
compute on-device with a lightweight MFCC implementation (e.g. ESP-DSP),
so the PC-trained model transfers cleanly to the ESP32 later.
"""

import numpy as np
import librosa

SAMPLE_RATE = 16000
CLIP_SECONDS = 1.0
N_MFCC = 13          # keep small -- this directly drives on-device compute/RAM
N_FFT = 512           # 32ms window at 16kHz
HOP_LENGTH = 256      # 16ms hop -> ~63 frames per 1s clip
N_MELS = 40


def load_and_pad(path: str, target_len=int(SAMPLE_RATE * CLIP_SECONDS)) -> np.ndarray:
    audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]
    return audio


def extract_mfcc(audio: np.ndarray) -> np.ndarray:
    """Returns (n_frames, N_MFCC) float32 array, roughly (63, 13) for 1s @16kHz."""
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    return mfcc.T.astype(np.float32)  # (time, n_mfcc)


def featurize_file(path: str) -> np.ndarray:
    audio = load_and_pad(path)
    return extract_mfcc(audio)


def get_feature_shape():
    """Compute the (time, n_mfcc) shape once so model.py can build the right input layer."""
    dummy = np.zeros(int(SAMPLE_RATE * CLIP_SECONDS), dtype=np.float32)
    return extract_mfcc(dummy).shape


if __name__ == "__main__":
    shape = get_feature_shape()
    print(f"Feature map shape per 1s clip: {shape} (time_frames, n_mfcc)")
