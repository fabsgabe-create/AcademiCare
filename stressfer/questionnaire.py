"""Kuesioner ground truth (Bab 3.2.2 & 3.6): subskala Stres gaya DASS-42, Likert 0-3.

PENTING: teks butir di bawah adalah TEMPLATE parafrase untuk keperluan prototipe.
Untuk penelitian, ganti dengan instrumen baku/tervalidasi (mis. DASS-42 versi Indonesia
yang berlisensi/izin penggunaan) sesuai ketentuan komite etik Anda. Logika skor & ambang
batas tidak perlu diubah bila jumlah butir tetap 14.
"""
import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from . import config

LIKERT = {
    0: "Tidak sesuai dengan saya sama sekali",
    1: "Sesuai dengan saya sampai tingkat tertentu / kadang-kadang",
    2: "Sesuai dengan saya sampai batas yang dapat dipertimbangkan / sering",
    3: "Sangat sesuai dengan saya / sering sekali",
}

# 14 butir subskala stres (template parafrase, selama satu minggu terakhir)
ITEMS: List[str] = [
    "Saya marah karena hal-hal yang sepele.",
    "Saya cenderung bereaksi berlebihan terhadap suatu situasi.",
    "Saya merasa menghabiskan banyak energi karena cemas.",
    "Saya merasa gelisah.",
    "Saya merasa sulit untuk bersantai.",
    "Saya tidak sabar terhadap hal yang menghalangi pekerjaan saya.",
    "Saya merasa mudah tersinggung.",
    "Saya sulit menenangkan diri setelah sesuatu mengganggu saya.",
    "Saya sulit mentoleransi gangguan terhadap hal yang sedang saya kerjakan.",
    "Saya berada dalam keadaan tegang.",
    "Saya mudah marah.",
    "Saya sulit untuk beristirahat/menurunkan ketegangan.",
    "Saya merasa gugup secara berlebihan.",
    "Saya merasa tidak sabar ketika harus menunggu.",
]

# Ambang skor Stres DASS-42 (jumlah 14 butir, tanpa pengali).
# Empat kelas penelitian: "Berat" menggabungkan kategori Berat dan Sangat Berat.
THRESHOLDS = [(14, "normal"), (18, "ringan"), (25, "sedang")]  # sisanya "berat"
MAX_SCORE = 3 * len(ITEMS)


def score(answers: List[int]) -> int:
    if len(answers) != len(ITEMS):
        raise ValueError(f"Jawaban harus berjumlah {len(ITEMS)} butir.")
    if any(a not in (0, 1, 2, 3) for a in answers):
        raise ValueError("Setiap jawaban harus bernilai 0-3.")
    return int(sum(answers))


def classify(total: int) -> str:
    for upper, label in THRESHOLDS:
        if total <= upper:
            return label
    return "berat"


def valid_subject_id(s: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_-]{1,32}", s or ""))


def log_response(subject_id: str, answers: List[int], out_dir: Path = config.QUESTIONNAIRE_DIR) -> Dict:
    """Simpan jawaban ke data/questionnaire/responses.csv dan kembalikan ringkasan."""
    total = score(answers)
    label = classify(total)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    file = out_dir / "responses.csv"
    new = not file.exists()
    with file.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["timestamp", "subject_id", *[f"q{i + 1}" for i in range(len(ITEMS))], "total", "label"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), subject_id, *answers, total, label])
    return {"subject_id": subject_id, "total": total, "label": label}
