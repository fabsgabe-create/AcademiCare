"""Konfigurasi terpusat. Semua nilai mengacu pada Bab III penelitian."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- Direktori ---------------------------------------------------------------
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                  # citra mentah per kelas: raw/<kelas>/*.jpg
PROCESSED_DIR = DATA_DIR / "processed"      # wajah hasil crop + dataset.npz
QUESTIONNAIRE_DIR = DATA_DIR / "questionnaire"
MODELS_DIR = ROOT / "models"
CASCADE_PATH = MODELS_DIR / "cascades" / "haarcascade_frontalface_default.xml"
MODEL_PATH = MODELS_DIR / "svm_stress.joblib"
REPORTS_DIR = ROOT / "reports"

# --- Kelas target (variabel dependen, Bab 3.2.3) ----------------------------
CLASSES = ["normal", "ringan", "sedang", "berat"]
CLASS_LABELS = {
    "normal": "Tidak Stres (Normal)",
    "ringan": "Stres Ringan",
    "sedang": "Stres Sedang",
    "berat": "Stres Berat",
}
CLASS_DESCRIPTIONS = {
    "normal": "Ekspresi rileks dan stabil.",
    "ringan": "Mulai tampak tanda kelelahan emosional ringan.",
    "sedang": "Tampak indikator ketegangan psikologis yang cukup jelas.",
    "berat": "Tampak ekspresi kelelahan/ketegangan yang signifikan.",
}

# --- Pre-processing (Bab 3.7 poin 1) ---------------------------------------
IMAGE_SIZE = (100, 100)        # normalisasi ukuran, 100 x 100 piksel
GAUSSIAN_KERNEL = (3, 3)       # Gaussian Blur untuk reduksi noise
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}   # batasan masalah no. 1

# --- Haar Cascade ----------------------------------------------------------
HAAR_SCALE_FACTOR = 1.1
HAAR_MIN_NEIGHBORS = 5
HAAR_MIN_SIZE = (60, 60)

# --- Pelatihan SVM (Bab 3.4 & 3.7) -----------------------------------------
TEST_SIZE = 0.2                # 80% latih / 20% uji
CV_FOLDS = 5                   # K-Fold Cross-Validation, K = 5
RANDOM_STATE = 42
SVM_KERNEL = "rbf"
PARAM_GRID = {                 # Grid Search untuk C dan gamma
    "svm__C": [0.1, 1, 10, 100],
    "svm__gamma": ["scale", 1e-5, 1e-4, 1e-3],
}

# --- Akuisisi citra (Bab 3.3 & 3.6) ----------------------------------------
CAPTURE_DISTANCE_CM = "50-70"
CAPTURE_LUX = "300-500"

DISCLAIMER = (
    "Sistem ini adalah alat bantu analisis awal untuk keperluan penelitian dan "
    "bukan pengganti diagnosis profesional terhadap kondisi kesehatan mental. "
    "Jika Anda merasa tertekan, hubungi psikolog atau layanan konseling kampus."
)


class Workspace:
    """Kumpulan path kerja. Default = folder proyek; dapat diarahkan ke folder lain (mis. untuk tes)."""

    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.data = self.root / "data"
        self.train_dir = self.data / "raw"          # data latih: raw/<kelas>/*.jpg
        self.test_dir = self.data / "test"          # data uji (opsional): test/<kelas>/*.jpg
        self.processed = self.data / "processed"
        self.questionnaire = self.data / "questionnaire"
        self.model_path = self.root / "models" / "svm_stress.joblib"
        self.reports = self.root / "reports"
        self.dataset_file = self.processed / "dataset.npz"
        self.dataset_test_file = self.processed / "dataset_test.npz"
        self.history_dir = self.data / "history"       # riwayat hasil per mahasiswa (tanpa citra)
        self.submissions_dir = self.data / "submissions"  # kiriman foto mahasiswa (menunggu ditinjau admin)
        self.users_file = self.data / "users.json"      # akun login (password di-hash)
        self.secret_file = self.data / ".secret_key"    # kunci sesi

    def split_dir(self, split: str) -> Path:
        if split not in ("train", "test"):
            raise ValueError("split harus 'train' atau 'test'")
        return self.train_dir if split == "train" else self.test_dir


SPLIT_NAMES = {"train": "Data latih", "test": "Data uji"}
CLASS_SHORT = {"normal": "Normal", "ringan": "Ringan", "sedang": "Sedang", "berat": "Berat"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024


# Identitas pada halaman Kontak (ubah sesuai kebutuhan)
CONTACT = {
    "name": "Primayoga Ananda",
    "program": "Magister Teknik Informatika, Fakultas Ilmu Komputer",
    "university": "Universitas Putra Indonesia \"YPTK\" Padang",
    "email": "",   # isi bila ingin ditampilkan di halaman Kontak
}
