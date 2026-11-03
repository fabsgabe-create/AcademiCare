"""Data SINTETIS hanya untuk menguji pipeline (bukan wajah asli, bukan data penelitian).

Setiap kelas memiliki pola intensitas berbeda + noise acak, agar alur
pre-processing -> fitur -> SVM -> evaluasi dapat diuji tanpa dataset nyata.
Hasil akurasi dari data ini TIDAK boleh dilaporkan sebagai hasil penelitian.
"""
from pathlib import Path

import cv2
import numpy as np

from . import config

DEMO_DIR = config.DATA_DIR / "demo_raw"


def _synthetic_face(label_idx: int, rng: np.random.Generator, size: int = 120) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    base = 120 + 25 * np.sin(xx / (14 - 2 * label_idx)) * np.cos(yy / (16 - 2 * label_idx))
    cx, cy = size / 2 + rng.normal(0, 3), size / 2 + rng.normal(0, 3)
    face = np.exp(-(((xx - cx) / 40) ** 2 + ((yy - cy) / 52) ** 2))
    img = base * (0.4 + 0.6 * face) + label_idx * 12 * (yy / size)  # gradasi makin gelap/tegang
    img += rng.normal(0, 14, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_demo_dataset(per_class: int = 40, out_dir: Path = DEMO_DIR, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    for i, c in enumerate(config.CLASSES):
        folder = out_dir / c
        folder.mkdir(parents=True, exist_ok=True)
        for n in range(per_class):
            cv2.imwrite(str(folder / f"demo_{n:03d}.png"), _synthetic_face(i, rng))
    return out_dir
