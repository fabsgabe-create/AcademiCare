"""Unggah data latih/uji: banyak citra atau ZIP berfolder kelas. Semua masukan divalidasi.

Aturan keamanan: nama berkas dari pengguna TIDAK dipakai sebagai path; berkas disimpan dengan nama acak,
hanya JPG/PNG yang benar-benar dapat didekode, ukuran dibatasi, dan ZIP dibatasi jumlah/ukuran isinya.
"""
import io
import re
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from . import config

MAX_ZIP_ENTRIES = 20000
MAX_ZIP_UNCOMPRESSED = 1024 * 1024 * 1024   # 1 GB
MIN_SIDE, MAX_SIDE = 32, 12000
_TRAIN_WORDS = {"train", "training", "latih"}
_TEST_WORDS = {"test", "testing", "uji"}


def _ext_from_magic(raw: bytes) -> Optional[str]:
    if raw[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    return None


def _validate(raw: bytes) -> Tuple[Optional[str], Optional[str]]:
    """Kembalikan (ekstensi, alasan_penolakan)."""
    if not raw:
        return None, "berkas kosong"
    if len(raw) > config.MAX_IMAGE_BYTES:
        return None, "lebih dari 10 MB"
    ext = _ext_from_magic(raw)
    if ext is None:
        return None, "bukan JPG/PNG"
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None, "citra rusak"
    h, w = img.shape[:2]
    if min(h, w) < MIN_SIDE or max(h, w) > MAX_SIDE:
        return None, f"ukuran {w}x{h} di luar batas"
    return ext, None


def store_image(raw: bytes, ws: config.Workspace, split: str, label: str, prefix: str = "up") -> str:
    """Validasi lalu simpan satu citra. Melempar ValueError bila ditolak."""
    if label not in config.CLASSES:
        raise ValueError("kelas tidak valid")
    ext, reason = _validate(raw)
    if reason:
        raise ValueError(reason)
    folder = ws.split_dir(split) / label
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{prefix}_{uuid.uuid4().hex[:12]}{ext}"
    (folder / name).write_bytes(raw)
    return name


def save_images(items: Iterable[Tuple[str, bytes]], ws: config.Workspace, split: str, label: str) -> dict:
    report = {"saved": 0, "rejected": [], "per": {}}
    for name, raw in items:
        try:
            store_image(raw, ws, split, label)
            report["saved"] += 1
            report["per"][f"{split}/{label}"] = report["per"].get(f"{split}/{label}", 0) + 1
        except ValueError as e:
            report["rejected"].append({"name": _safe_name(name), "reason": str(e)})
    return report


def _safe_name(name: str) -> str:
    return PurePosixPath(str(name).replace("\\", "/")).name[:80] or "(tanpa nama)"


def _route(parts: List[str], split: Optional[str], label: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    lower = [p.lower() for p in parts]
    lbl = next((p for p in reversed(lower) if p in config.CLASSES), label)
    if split in ("train", "test"):
        sp = split
    elif any(p in _TEST_WORDS for p in lower):
        sp = "test"
    else:
        sp = "train"
    return sp, lbl


def save_zip(data: bytes, ws: config.Workspace, split: Optional[str] = None,
             label: Optional[str] = None) -> dict:
    """ZIP berisi folder kelas (normal/ringan/sedang/berat), boleh dibungkus folder train/ dan test/.

    split=None -> ikuti struktur ZIP (folder test/uji -> data uji, selain itu data latih).
    label diberikan -> dipakai untuk berkas yang tidak berada dalam folder kelas.
    """
    report = {"saved": 0, "rejected": [], "per": {}}
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise ValueError("Berkas bukan ZIP yang valid.")
    infos = [i for i in zf.infolist() if not i.is_dir()]
    if len(infos) > MAX_ZIP_ENTRIES:
        raise ValueError(f"ZIP berisi lebih dari {MAX_ZIP_ENTRIES} berkas.")
    if sum(i.file_size for i in infos) > MAX_ZIP_UNCOMPRESSED:
        raise ValueError("Isi ZIP setelah diekstrak melebihi 1 GB.")
    for info in infos:
        path = PurePosixPath(info.filename.replace("\\", "/"))
        name = path.name
        if "__MACOSX" in path.parts or name.startswith(".") or path.suffix.lower() not in config.IMAGE_EXTENSIONS:
            continue
        sp, lbl = _route(list(path.parts[:-1]), split, label)
        if lbl is None:
            report["rejected"].append({"name": name, "reason": "folder kelas tidak ditemukan"})
            continue
        if info.file_size > config.MAX_IMAGE_BYTES:
            report["rejected"].append({"name": name, "reason": "lebih dari 10 MB"})
            continue
        try:
            with zf.open(info) as fh:
                raw = fh.read(config.MAX_IMAGE_BYTES + 1)
            store_image(raw, ws, sp, lbl, prefix="zip")
            report["saved"] += 1
            key = f"{sp}/{lbl}"
            report["per"][key] = report["per"].get(key, 0) + 1
        except (ValueError, RuntimeError) as e:   # RuntimeError: ZIP berkata sandi
            report["rejected"].append({"name": name, "reason": str(e)})
    return report


def clear_images(ws: config.Workspace, split: str, label: Optional[str] = None) -> int:
    """Hapus citra pada data latih/uji (seluruh kelas atau satu kelas). Mengembalikan jumlah terhapus."""
    labels = [label] if label else config.CLASSES
    removed = 0
    for lb in labels:
        if lb not in config.CLASSES:
            raise ValueError("kelas tidak valid")
        folder = ws.split_dir(split) / lb
        if folder.is_dir():
            for p in folder.iterdir():
                if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS:
                    p.unlink()
                    removed += 1
    return removed


_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")


def image_path(ws: config.Workspace, split: str, label: str, name: str) -> Optional[Path]:
    """Path aman ke satu citra, atau None bila split/kelas/nama tidak valid atau berkas tidak ada."""
    if split not in ("train", "test") or label not in config.CLASSES or not _NAME_RE.match(name or ""):
        return None
    p = ws.split_dir(split) / label / name
    return p if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS else None


def list_images_page(ws: config.Workspace, split: str, label: str, offset: int = 0, limit: int = 24):
    """Daftar nama citra (terbaru dulu) beserta total, untuk galeri."""
    if split not in ("train", "test") or label not in config.CLASSES:
        raise ValueError("split atau kelas tidak valid")
    folder = ws.split_dir(split) / label
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS] \
        if folder.is_dir() else []
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in files[offset:offset + limit]], len(files)


def delete_image(ws: config.Workspace, split: str, label: str, name: str) -> bool:
    p = image_path(ws, split, label, name)
    if p is None:
        return False
    p.unlink()
    return True
