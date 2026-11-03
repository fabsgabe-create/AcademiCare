"""Membangun dataset fitur dari citra mentah: <folder>/<kelas>/*.jpg -> dataset.npz."""
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from . import config
from .face_detection import FaceDetector
from .features import flatten_features
from .preprocessing import normalize_face, to_grayscale

Progress = Optional[Callable[[int, int], None]]


def list_images(root: Path) -> List[Tuple[Path, str]]:
    items = []
    for label in config.CLASSES:
        folder = Path(root) / label
        if folder.is_dir():
            for p in sorted(folder.iterdir()):
                if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS:
                    items.append((p, label))
    return items


def count_images(root: Path) -> Dict[str, int]:
    counts = {c: 0 for c in config.CLASSES}
    for _, label in list_images(root):
        counts[label] += 1
    return counts


def build_dataset(raw_dir: Path, out_file: Path, already_cropped: bool = False,
                  faces_dir: Optional[Path] = None, detector: Optional[FaceDetector] = None,
                  progress: Progress = None) -> dict:
    """Pre-processing + deteksi wajah + flattening untuk seluruh citra di raw_dir.

    already_cropped=True melewati Haar Cascade (citra sudah berupa wajah).
    """
    items = list_images(raw_dir)
    if not items:
        raise FileNotFoundError(
            f"Tidak ada citra (.jpg/.png) di {raw_dir}/<kelas>/. Unggah data dulu lewat menu Data.")
    detector = None if already_cropped else (detector or FaceDetector())

    X, y, skipped = [], [], []
    for i, (path, label) in enumerate(items, 1):
        if progress:
            progress(i, len(items))
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            skipped.append(path.name)
            continue
        if already_cropped:
            face = normalize_face(to_grayscale(img))
        else:
            face, _ = detector.extract_roi(img)
            if face is None:
                skipped.append(path.name)
                continue
        X.append(flatten_features(face))
        y.append(config.CLASSES.index(label))
        if faces_dir is not None:
            dest = Path(faces_dir) / label
            dest.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dest / f"{path.stem}.png"), face)

    if not X:
        raise RuntimeError(
            f"Tidak ada wajah terdeteksi pada {len(items)} citra di {raw_dir}. "
            "Pastikan wajah frontal, atau centang 'citra sudah berupa wajah'.")
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_file, X=np.stack(X), y=np.array(y))
    ya = np.array(y)
    return {"total": len(X), "found": len(items), "skipped": skipped,
            "per_class": {c: int(np.sum(ya == i)) for i, c in enumerate(config.CLASSES)}}


def build_all(ws: config.Workspace, already_cropped: bool = False,
              progress: Optional[Callable[[int, int, str], None]] = None) -> dict:
    """Bangun dataset latih, dan dataset uji bila folder data uji berisi citra."""
    n_train = sum(count_images(ws.train_dir).values())
    n_test = sum(count_images(ws.test_dir).values())
    total = n_train + n_test
    detector = None if already_cropped else FaceDetector()

    def mk(offset, name):
        return (lambda i, n: progress(offset + i, total, name)) if progress else None

    out = {"train": build_dataset(ws.train_dir, ws.dataset_file, already_cropped,
                                  ws.processed / "faces" / "train", detector, mk(0, "data latih")),
           "test": None}
    if n_test:
        out["test"] = build_dataset(ws.test_dir, ws.dataset_test_file, already_cropped,
                                    ws.processed / "faces" / "test", detector, mk(n_train, "data uji"))
    elif ws.dataset_test_file.exists():
        ws.dataset_test_file.unlink()   # buang sisa dataset uji lama
    return out


def load_dataset(file: Path):
    file = Path(file)
    if not file.exists():
        raise FileNotFoundError(f"Dataset belum dibuat: {file}. Jalankan pre-processing / pelatihan dulu.")
    d = np.load(file, allow_pickle=False)
    return d["X"], d["y"]


def load_test_if_any(ws: config.Workspace):
    return load_dataset(ws.dataset_test_file) if ws.dataset_test_file.exists() else (None, None)
