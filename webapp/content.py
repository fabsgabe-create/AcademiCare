"""Konten halaman publik (Pengumuman, FAQ). Ubah teks di sini; tidak perlu menyentuh template."""

ANNOUNCEMENTS = [
    {"tag": "Panduan", "title": "Standar pengambilan citra wajah",
     "text": "Jarak kamera 50-70 cm, pencahayaan 300-500 lux, wajah menghadap depan tanpa penghalang "
             "(masker, kacamata hitam, topi yang menutupi dahi). Konsistensi kondisi pengambilan sangat "
             "memengaruhi kualitas data."},
    {"tag": "Etika", "title": "Persetujuan responden wajib",
     "text": "Citra wajah adalah data pribadi. Ambil citra hanya setelah responden memberi persetujuan "
             "(informed consent), gunakan ID anonim, dan jangan membagikan data mentah di luar tim penelitian."},
    {"tag": "Penting", "title": "Hasil sistem bukan diagnosis",
     "text": "Keluaran sistem adalah estimasi awal berbasis pola citra wajah untuk keperluan penelitian. "
             "Bukan pengganti penilaian psikolog atau tenaga kesehatan profesional."},
    {"tag": "Info", "title": "Format data yang didukung",
     "text": "Citra JPG/PNG (maks. 10 MB per citra) atau ZIP berisi folder kelas normal, ringan, sedang, berat. "
             "Data uji boleh diunggah terpisah dari data latih."},
]

FAQ = [
    {"q": "Apa itu AcademiCare?",
     "a": "Aplikasi untuk menganalisis dan mengklasifikasikan tingkat stres akademik mahasiswa dari citra ekspresi "
          "wajah. Wajah dideteksi dengan Haar Cascade, lalu tingkat stres (normal, ringan, sedang, berat) "
          "diklasifikasikan dengan Support Vector Machine."},
    {"q": "Apakah hasil prediksi sama dengan diagnosis?",
     "a": "Tidak. Hasil hanya estimasi awal untuk penelitian. Jika Anda merasa tertekan, hubungi psikolog "
          "atau layanan konseling kampus."},
    {"q": "Bagaimana cara mulai memakai aplikasi?",
     "a": "Masuk dengan akun Anda, unggah data latih di menu Data (dan data uji bila ada), lalu jalankan "
          "pelatihan di menu Latih Model. Setelah itu lihat Evaluasi atau coba Prediksi."},
    {"q": "Berapa jumlah data yang dibutuhkan?",
     "a": "Minimal 2 citra per kelas agar pelatihan bisa berjalan, tetapi hasil yang layak dipertanggungjawabkan "
          "membutuhkan jauh lebih banyak (puluhan hingga ratusan per kelas, dari banyak responden berbeda)."},
    {"q": "Mengapa wajah saya tidak terdeteksi?",
     "a": "Haar Cascade paling baik pada wajah frontal dengan pencahayaan merata. Hindari wajah miring, "
          "tertutup, atau terlalu kecil di dalam gambar (minimal sekitar 60x60 piksel)."},
    {"q": "Bagaimana data saya dilindungi?",
     "a": "Seluruh halaman data dan prediksi memerlukan login. Password disimpan dalam bentuk hash, dan "
          "citra tersimpan di server aplikasi ini saja, tidak dikirim ke layanan lain."},
    {"q": "Saya lupa password, bagaimana?",
     "a": "Minta administrator aplikasi mengatur ulang lewat perintah di server: python main.py set-password USERNAME."},
]
