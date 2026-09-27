"""
Phase 1 - Step 3: Tiny CNN for 3-class keyword spotting (silence/unknown/vaani).

Kept deliberately small so INT8 TFLite export lands well under the 256KB
budget with room to spare. Architecture is a scaled-down version of the
"DS-CNN" family that's standard for KWS on microcontrollers (depthwise-
separable convs trade a small accuracy hit for a big parameter/compute win).
"""

import tensorflow as tf
from tensorflow.keras import layers, models

LABELS = ["silence", "unknown", "vaani"]
NUM_CLASSES = len(LABELS)


def build_tiny_ds_cnn(input_shape, num_classes=NUM_CLASSES, width=1.0):
    """
    input_shape: (time_frames, n_mfcc) e.g. (63, 13)
    width: channel multiplier to trade size vs accuracy (0.5 = smaller/faster)
    """
    c1, c2, c3 = [max(8, int(n * width)) for n in (32, 32, 32)]

    inputs = layers.Input(shape=(*input_shape, 1))

    x = layers.Conv2D(c1, (10, 4), strides=(2, 2), padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    for filters in (c2, c3):
        x = layers.DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, (1, 1), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="tiny_ds_cnn_kws")
    return model


def build_micro_cnn(input_shape, num_classes=NUM_CLASSES):
    """
    Even smaller fallback architecture (plain conv, no depthwise) if you want
    the absolute floor on size/latency at some accuracy cost. Good baseline
    to compare the DS-CNN against.
    """
    inputs = layers.Input(shape=(*input_shape, 1))
    x = layers.Conv2D(16, (8, 4), strides=(2, 2), padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(24, (3, 3), padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="micro_cnn_kws")


if __name__ == "__main__":
    from features import get_feature_shape

    shape = get_feature_shape()
    model = build_tiny_ds_cnn(shape)
    model.summary()
    n_params = model.count_params()
    print(f"\nApprox float32 size: {n_params * 4 / 1024:.1f} KB "
          f"(INT8 post-quantization should be roughly 1/4 of this)")
