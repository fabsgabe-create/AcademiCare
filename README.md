# AcademiCare — Deteksi & Klasifikasi Tingkat Stres Akademik Mahasiswa
### Berbasis Citra Ekspresi Wajah — Haar Cascade + Support Vector Machine

Implementasi program & aplikasi untuk tesis *Primayoga Ananda (251321036), Magister Teknik Informatika UPI YPTK Padang*,
mengikuti Bab III (metodologi).

> Sistem ini alat bantu analisis awal untuk penelitian, **bukan diagnosis profesional**.

## Pipeline (sesuai Bab III)

| Tahap | Implementasi | Berkas |
|---|---|---|
| Akuisisi citra + kuesioner ground truth (DASS, Likert 0–3) | webcam via OpenCV / browser | `main.py capture`, `webapp`, `stressfer/questionnaire.py` |
| Pre-processing: RGB→Grayscale, resize 100×100, Gaussian Blur | `cv2.cvtColor` (0.299R+0.587G+0.114B) | `stressfer/preprocessing.py` |
| Deteksi wajah Haar Cascade → crop ROI | `haarcascade_frontalface_default.xml` | `stressfer/face_detection.py` |
| Ekstraksi fitur: flattening piksel (10.000 fitur) | | `stressfer/features.py` |
| Split 80/20 (stratified) + 5-fold CV | | `stressfer/model.py` |
| SVM kernel RBF + Grid Search (C, γ) | scikit-learn | `stressfer/model.py` |
| Evaluasi: confusion matrix, accuracy, precision, recall, F1 | | `stressfer/evaluation.py` |

Kelas: `normal`, `ringan`, `sedang`, `berat`.

## Struktur Direktori

```
stress-detection/
├── main.py                    # CLI: demo-data | capture | preprocess | train | evaluate | predict | webcam | serve
├── requirements.txt
├── README.md
├── stressfer/                 # paket inti
│   ├── config.py              # parameter (ukuran citra, grid, split, kelas) + Workspace (path kerja)
│   ├── preprocessing.py       # grayscale, resize, Gaussian Blur
│   ├── face_detection.py      # Haar Cascade + crop ROI
│   ├── features.py            # flattening
│   ├── dataset.py             # <folder>/<kelas>/*.jpg -> dataset.npz (latih) & dataset_test.npz (uji)
│   ├── uploads.py             # validasi & simpan unggahan citra/ZIP, galeri, hapus
│   ├── model.py               # Grid Search + 5-fold CV, SVM RBF, simpan/muat model
│   ├── jobs.py                # pelatihan di latar belakang (dipakai web)
│   ├── evaluation.py          # metrik + confusion matrix (JSON, TXT, PNG)
│   ├── predictor.py           # inferensi + anotasi citra
│   ├── questionnaire.py       # 14 butir stres, skor -> label kelas
│   └── demo_data.py           # data SINTETIS untuk uji pipeline saja
├── webapp/                    # aplikasi web Flask
│   ├── app.py
│   ├── templates/             # dashboard, data, train, evaluation, predict, capture, questionnaire
│   └── static/{css,js}/
├── data/
│   ├── raw/{normal,ringan,sedang,berat}/    # DATA LATIH
│   ├── test/{normal,ringan,sedang,berat}/   # DATA UJI (opsional)
│   ├── processed/             # dataset.npz, dataset_test.npz, wajah hasil crop
│   └── questionnaire/         # responses.csv (otomatis)
├── models/
│   ├── cascades/haarcascade_frontalface_default.xml
│   └── svm_stress.joblib      # dibuat oleh pelatihan
├── reports/                   # metrics.json, classification_report.txt, confusion_matrix.png
└── tests/                     # pytest (25 tes)
```

## Instalasi

Python 3.10+ (disarankan memakai virtual environment).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> `requirements.txt` membatasi `opencv-python<5` karena OpenCV 5.x tidak lagi menyertakan Haar Cascade,
> dan `scikit-learn<1.11` karena `SVC(probability=True)` akan dihapus di sana.

## Cara Pakai (aplikasi web, tanpa terminal selain menjalankan server)

```bash
python main.py serve        # lalu buka http://127.0.0.1:5000
```

0. **Login & peran**. `/` adalah beranda publik. Saat `python main.py serve` dijalankan pertama kali, akun **admin** dibuat
   otomatis dengan **password acak yang ditampilkan sekali di terminal** (tidak ada password bawaan yang bisa ditebak; atau
   tetapkan sendiri lewat env `STRESSFER_ADMIN_PASSWORD`). Ganti kapan saja: `python main.py set-password admin`.
   Akun admin tambahan: `python main.py create-user NAMA --role admin`. Password disimpan sebagai hash di `data/users.json`,
   kunci sesi di `data/.secret_key` (di-ignore git; jangan dibagikan). Login dibatasi percobaan (5 gagal/user atau 20/IP per 5 menit).
   Env opsional: `STRESSFER_SECRET`, `STRESSFER_HTTPS=1` (cookie Secure), `STRESSFER_ALLOW_REMOTE_SETUP=1`.

   - **Admin/peneliti**: akses penuh. Menu **Mahasiswa** punya dua tab: *Daftar mahasiswa* (hasil kuesioner dan prediksi, detail per orang,
     tombol hapus riwayat/foto/akun) dan *Foto mahasiswa* (galeri semua foto kiriman dengan filter status dan label; masukkan ke
     **data latih/uji** sesuai label, lihat hasil deteksi, atau hapus), serta unduh **CSV**. Menu **Pengguna**:
     atur ulang password dan hapus akun.
   - **Mahasiswa**: mendaftar sendiri di `/daftar` (NIM sebagai username, wajib centang persetujuan). Bisa mengisi **Kuesioner**,
     **Kirim Foto** ke peneliti (label otomatis dari kuesioner terakhir, maks. 20 foto) lalu langsung melihat **Hasil Deteksi** (kotak wajah, tingkat stres prediksi model beserta probabilitas, dan perbandingan dengan label kuesioner), mencoba **Prediksi** (foto tidak disimpan),
     dan melihat **Riwayat**. Mahasiswa **tidak dapat menghapus apa pun**; penghapusan hanya lewat admin. Halaman lain memberi 403.
   - Alur pengumpulan data: mahasiswa daftar → isi kuesioner (label) → kirim foto → admin tinjau → masukkan ke data latih/uji → Latih Model.
     Pastikan ada persetujuan etik/informed consent sebelum dipakai untuk penelitian sebenarnya.
1. **Data**: unggah citra lepas (pilih kelas) atau ZIP berfolder kelas, sebagai **data latih** atau **data uji**.
   Periksa di galeri dan hapus yang salah label. Bisa juga ambil dari menu **Kamera** (dengan *informed consent*).
2. **Latih Model**: satu tombol menjalankan pre-processing, Haar Cascade, Grid Search SVM (5-fold CV), dan evaluasi.
   Centang *Citra sudah berupa wajah* bila dataset berisi foto wajah yang sudah dipotong.
3. **Evaluasi**: confusion matrix, accuracy, precision, recall, F1.
4. **Prediksi**: unggah citra atau webcam real-time.

**Data uji terpisah atau tidak?** Bila `data/test/` berisi citra, model dilatih pada seluruh data latih dan
dievaluasi pada data uji tersebut. Bila kosong, data latih dibagi 80/20 (stratified).

**Format ZIP** (nama folder kelas harus `normal`, `ringan`, `sedang`, `berat`; nama file bebas):

```
dataset.zip
├── train/   (atau latih/)  → data latih
│   └── normal/ ringan/ sedang/ berat/
└── test/    (atau uji/)    → data uji
    └── normal/ ringan/ sedang/ berat/
```
Tanpa folder train/test, semua isi ZIP masuk ke tujuan yang Anda pilih (atau data latih bila "Ikuti folder ZIP").

Keamanan unggahan: nama file dari pengguna tidak dipakai (disimpan dengan nama acak), hanya JPG/PNG yang benar-benar
bisa didekode (maks 10 MB/citra), isi ZIP dibatasi jumlah dan ukurannya.

## Cara Pakai (CLI)

```bash
python main.py capture --subject R001 --label normal --split train   # webcam; 1-4 ganti kelas, SPASI simpan, Q keluar
python main.py preprocess      # data/raw (+ data/test bila ada) -> data/processed/*.npz
python main.py train           # latih + evaluasi
python main.py evaluate        # evaluasi ulang
python main.py predict foto.jpg --output hasil.jpg
python main.py webcam          # real-time
```

## Mencoba pipeline tanpa data asli

```bash
python main.py demo-data       # membuat data/demo_raw/<kelas>/*.png (SINTETIS)
```
lalu di aplikasi web: Data → Unggah → pilih semua berkas di `data/demo_raw/`, atau lewat CLI:
`python main.py preprocess --raw data/demo_raw --already-cropped && python main.py train`.
Data ini **bukan wajah** dan akurasinya tidak bermakna; **jangan dilaporkan sebagai hasil penelitian**.
Hapus `models/svm_stress.joblib`, `reports/*`, dan `data/processed/*` sebelum melatih dengan data sebenarnya.

## Tes
```bash
python -m pytest -q
```

## Catatan penting untuk penelitian

- **Butir kuesioner** di `stressfer/questionnaire.py` adalah template parafrase. Ganti dengan instrumen baku
  (DASS-42 versi Indonesia, dengan izin penggunaan) sebelum pengambilan data. Ambang skor stres DASS-42:
  normal 0–14, ringan 15–18, sedang 19–25, berat ≥26 (kategori "berat" dan "sangat berat" digabung menjadi 4 kelas).
- **Etika & privasi**: citra wajah adalah data pribadi. Gunakan ID anonim, simpan *informed consent*, dan jangan
  meng-commit `data/` (sudah di `.gitignore`). Sesuaikan dengan ketentuan komite etik kampus.
- **Keterbatasan metodologis** yang perlu dibahas di Bab IV: fitur piksel mentah sensitif terhadap pencahayaan/pose;
  pastikan satu responden tidak muncul di data latih *dan* uji sekaligus (kebocoran data) — bila satu responden
  punya banyak foto, pisahkan split per responden; label berasal dari kuesioner, bukan dari ekspresi itu sendiri,
  sehingga akurasi mengukur korelasi citra–skor pada sampel Anda.
- Bab III menyebut "Depresi" pada beberapa bagian (3.7 poin 4) dan "Stres" di bagian lain; program ini konsisten
  memakai istilah **stres akademik**.
- Parameter (ukuran citra, grid C/γ, rasio split, jumlah fold) ada di `stressfer/config.py`.
