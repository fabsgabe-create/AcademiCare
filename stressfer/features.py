"""Ekstraksi fitur: intensitas piksel yang diratakan / flattening (Bab 3.4 tahap 6)."""
import numpy as np


def flatten_features(face: np.ndarray) -> np.ndarray:
    """ROI grayscale (H x W) -> vektor 1-D float32 berskala 0..1 (H*W fitur)."""
    return face.astype(np.float32).ravel() / 255.0
