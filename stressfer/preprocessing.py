"""Pre-processing citra (Bab 3.7 poin 1): grayscale, resize, Gaussian Blur."""
import cv2
import numpy as np

from . import config


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """RGB/BGR -> grayscale. OpenCV memakai Gray = 0.299R + 0.587G + 0.114B."""
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def normalize_face(face_gray: np.ndarray) -> np.ndarray:
    """Resize ke IMAGE_SIZE lalu Gaussian Blur untuk mereduksi noise."""
    face = cv2.resize(face_gray, config.IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(face, config.GAUSSIAN_KERNEL, 0)
