"""
Model package for Vaani Wake-Word Detection (DS-CNN).
"""

from .architecture import LABELS, NUM_CLASSES, build_tiny_ds_cnn, build_micro_cnn
from .features import extract_mfcc, featurize_file, get_feature_shape

__all__ = [
    "LABELS",
    "NUM_CLASSES",
    "build_tiny_ds_cnn",
    "build_micro_cnn",
    "extract_mfcc",
    "featurize_file",
    "get_feature_shape",
]
