"""Inferensi: citra/frame -> Haar Cascade -> pre-processing -> SVM -> tingkat stres."""
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import config
from .face_detection import FaceDetector
from .features import flatten_features
from .model import load_model

# Warna BGR per kelas untuk anotasi
_COLORS = {"normal": (80, 175, 76), "ringan": (0, 200, 255), "sedang": (0, 140, 255), "berat": (60, 60, 230)}


class StressPredictor:
    def __init__(self, model_path: Path = config.MODEL_PATH, detector: Optional[FaceDetector] = None):
        self.bundle = load_model(model_path)
        self.model = self.bundle["model"]
        self.classes = self.bundle["classes"]
        self.detector = detector or FaceDetector()

    def predict(self, img: np.ndarray) -> dict:
        """Prediksi wajah terbesar pada citra BGR. Selalu mengembalikan dict."""
        face, box = self.detector.extract_roi(img)
        if face is None:
            return {"face_found": False}
        feats = flatten_features(face).reshape(1, -1)
        probs = self.model.predict_proba(feats)[0]
        # urutan kolom proba = self.model.classes_ (indeks kelas)
        prob_by_class = {c: 0.0 for c in self.classes}
        for col, cls_idx in enumerate(self.model.classes_):
            prob_by_class[self.classes[int(cls_idx)]] = float(probs[col])
        label = max(prob_by_class, key=prob_by_class.get)
        return {
            "face_found": True,
            "box": box,
            "label": label,
            "label_text": config.CLASS_LABELS[label],
            "description": config.CLASS_DESCRIPTIONS[label],
            "confidence": prob_by_class[label],
            "probabilities": prob_by_class,
        }

    @staticmethod
    def annotate(img: np.ndarray, result: dict) -> np.ndarray:
        out = img.copy()
        if not result.get("face_found"):
            cv2.putText(out, "Wajah tidak terdeteksi", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2, cv2.LINE_AA)
            return out
        x, y, w, h = result["box"]
        color = _COLORS.get(result["label"], (255, 255, 255))
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        text = f'{result["label_text"]} {result["confidence"] * 100:.0f}%'
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        top = max(y - th - 10, 0)
        cv2.rectangle(out, (x, top), (x + tw + 8, top + th + 10), color, -1)
        cv2.putText(out, text, (x + 4, top + th + 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return out
