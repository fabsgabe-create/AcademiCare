"""Petakan dataset emosi (Angry/Disgust/Fear/Happy/Neutral/Sad/Surprise) ke 4 kelas stres, lalu buat dataset.zip
yang siap diunggah di halaman Data (struktur train/ dan test/ berisi normal, ringan, sedang, berat).

Contoh (Windows):
  python emosi_ke_stres.py --src "D:\\Data Pribadi\\PROJECT TESIS\\Aplikasi\\dataset" --out dataset.zip

--src boleh berupa folder atau berkas .zip, struktur apa pun: skrip mencari sendiri folder emosi (Angry, Happy, dst.)
di kedalaman berapa pun, membuka ZIP di dalamnya, dan menentukan data latih/uji dari nama folder induknya
(train/test). Folder emosi yang tidak berada di bawah train/test dianggap data latih.

PENTING: pemetaan emosi -> tingkat stres adalah ASUMSI, bukan label stres sebenarnya. Dasarkan pada literatur dan
konsultasikan ke pembimbing. Citra FER berukuran kecil dan sudah berupa wajah: centang "Citra sudah berupa wajah" saat melatih.
"""
import argparse
import random
import tempfile
import zipfile
from pathlib import Path

DEFAULT_MAP = {
    "normal": ["happy", "neutral"],
    "ringan": ["surprise", "surprize"],
    "sedang": ["sad", "fear"],
    "berat": ["angry", "disgust"],
}
EXT = {".jpg", ".jpeg", ".png"}
TRAIN_NAMES = {"train", "training", "latih"}
TEST_NAMES = {"test", "testing", "validation", "val", "uji"}


def parse_map(items):
    if not items:
        return DEFAULT_MAP
    m = {}
    for it in items:                       # contoh: berat=angry,disgust
        k, v = it.split("=", 1)
        m[k.strip().lower()] = [x.strip().lower() for x in v.split(",") if x.strip()]
    return m


def extract_zips(root: Path, tmp: Path, seen=None, depth=0):
    """Ekstrak semua .zip di dalam root (juga zip di dalam zip, maks 2 tingkat) ke folder sementara."""
    seen = seen if seen is not None else set()
    for z in list(root.rglob("*.zip")):
        if z in seen or depth > 2:
            continue
        seen.add(z)
        out = tmp / f"z{len(seen)}_{z.stem[:30]}"
        print(f"Mengekstrak {z.name} ...")
        try:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(out)
        except zipfile.BadZipFile:
            print(f"  (dilewati, bukan ZIP valid: {z.name})")
            continue
        extract_zips(out, tmp, seen, depth + 1)


def split_of(d: Path, base: Path) -> str:
    for part in d.relative_to(base).parts[:-1] if d != base else []:
        low = part.lower()
        if low in TEST_NAMES:
            return "test"
        if low in TRAIN_NAMES:
            return "train"
    return "train"


def show_tree(base: Path, limit=40):
    print(f"\nIsi {base} (2 tingkat pertama):")
    n = 0
    for p in sorted(base.iterdir()):
        print("  " + p.name + ("/" if p.is_dir() else ""))
        if p.is_dir():
            for q in sorted(p.iterdir())[:8]:
                print("      " + q.name + ("/" if q.is_dir() else ""))
        n += 1
        if n >= limit:
            print("  ...")
            break


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder ATAU berkas .zip dataset")
    ap.add_argument("--out", default="dataset.zip")
    ap.add_argument("--map", nargs="*", help="ubah pemetaan, mis. --map normal=happy,neutral berat=angry")
    ap.add_argument("--max-train", type=int, default=1500, help="maks. citra per kelas stres untuk data latih (0 = semua)")
    ap.add_argument("--max-test", type=int, default=300, help="maks. citra per kelas stres untuk data uji (0 = semua)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    mapping = parse_map(a.map)
    emo2cls = {e: c for c, es in mapping.items() for e in es}
    src = Path(a.src)
    if not src.exists():
        raise SystemExit(f"Alamat tidak ditemukan: {src}")

    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        bases = []
        if src.is_file():
            if src.suffix.lower() != ".zip":
                raise SystemExit("--src harus folder atau berkas .zip")
            bases.append(src.parent)
            tmp_zip_dir = tmp / "src"
            tmp_zip_dir.mkdir()
            with zipfile.ZipFile(src) as zf:
                zf.extractall(tmp_zip_dir)
            bases = [tmp_zip_dir]
        else:
            bases = [src]
        for b in list(bases):
            extract_zips(b, tmp)
        bases.append(tmp)

        pool = {"train": {c: [] for c in mapping}, "test": {c: [] for c in mapping}}
        seen_dirs = set()
        for base in bases:
            for d in base.rglob("*"):
                if not d.is_dir() or d.name.lower() not in emo2cls or d in seen_dirs:
                    continue
                if any(p in seen_dirs for p in d.parents):          # emosi di dalam folder emosi
                    continue
                seen_dirs.add(d)
                sp, cls = split_of(d, base), emo2cls[d.name.lower()]
                pool[sp][cls] += [(d.name.lower(), p) for p in d.rglob("*") if p.suffix.lower() in EXT]

        total = sum(len(v) for sp in pool.values() for v in sp.values())
        if not total:
            print("Tidak ada citra dari folder emosi yang ditemukan.")
            print("Nama folder emosi yang dicari:", ", ".join(sorted(emo2cls)))
            show_tree(src if src.is_dir() else Path(t))
            raise SystemExit(1)

        rnd = random.Random(a.seed)
        report, written = [], 0
        with zipfile.ZipFile(a.out, "w", zipfile.ZIP_STORED) as z:
            for split in ("train", "test"):
                cap = a.max_train if split == "train" else a.max_test
                for cls, items in pool[split].items():
                    items.sort(key=lambda x: str(x[1]))
                    rnd.shuffle(items)
                    items = items[:cap] if cap else items
                    for i, (emo, p) in enumerate(items):
                        z.write(p, f"{split}/{cls}/{emo}_{i:05d}{p.suffix.lower()}")
                    if items:
                        report.append((split, cls, len(items)))
                    written += len(items)

    print(f"Dibuat: {a.out} ({written} citra)")
    for split, cls, n in report:
        print(f"  {split:5s} {cls:7s} {n}")
    if not any(s == "test" for s, _, _ in report):
        print("Catatan: tidak ada data uji terpisah; aplikasi akan membagi data latih 80/20 otomatis.")
    if written > 20000:
        print("PERINGATAN: lebih dari 20.000 berkas; aplikasi menolak ZIP sebesar itu. Turunkan --max-train/--max-test.")


if __name__ == "__main__":
    main()
