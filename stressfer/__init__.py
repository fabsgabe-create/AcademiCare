"""stressfer - Deteksi & klasifikasi tingkat stres akademik berbasis citra ekspresi wajah.

Pipeline (sesuai Bab III proposal tesis):
    citra -> grayscale -> deteksi wajah (Haar Cascade) -> crop ROI -> resize 100x100
    -> Gaussian Blur -> flattening -> SVM (kernel RBF, Grid Search, 5-fold CV)
"""

__version__ = "1.0.0"
