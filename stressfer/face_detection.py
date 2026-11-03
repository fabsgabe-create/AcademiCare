"""Deteksi wajah dengan Haar Cascade Classifier (Bab 2.5 & 3.4 tahap 5)."""
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import config
from .preprocessing import normalize_face, to_grayscale

Box = Tuple[int, int, int, int]  # x, y, w, h


class FaceDetector:
    def __init__(self, cascade_path: Optional[Path] = None):
        path = Path(cascade_path or config.CASCADE_PATH)
        if not path.exists():  # fallback ke file bawaan OpenCV
            alt = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            if alt.exists():
                path = alt
        if not path.exists():
            raise FileNotFoundError(f"File Haar Cascade tidak ditemukan: {path}")
        self.cascade = cv2.CascadeClassifier(str(path))
        if self.cascade.empty():
            raise RuntimeError(f"Gagal memuat Haar Cascade: {path}")

    def detect(self, img: np.ndarray) -> List[Box]:
        """Kembalikan semua kotak wajah, terbesar lebih dulu."""
        gray = to_grayscale(img)
        gray = cv2.equalizeHist(gray)  # bantu variasi pencahayaan (hanya utk deteksi)
        boxes = self.cascade.detectMultiScale(
            gray,
            scaleFactor=config.HAAR_SCALE_FACTOR,
            minNeighbors=config.HAAR_MIN_NEIGHBORS,
            minSize=config.HAAR_MIN_SIZE,
        )
        boxes = [tuple(int(v) for v in b) for b in boxes]
        return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)

    def extract_roi(self, img: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Box]]:
        """Deteksi wajah terbesar -> crop ROI -> resize + blur. (None, None) bila tak ada wajah."""
        boxes = self.detect(img)
        if not boxes:
            return None, None
        x, y, w, h = boxes[0]
        roi = to_grayscale(img)[y:y + h, x:x + w]
        return normalize_face(roi), boxes[0]
