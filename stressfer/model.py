"""Pelatihan SVM (kernel RBF) dengan Grid Search + K-Fold CV (Bab 3.4 tahap 7, 3.7 poin 2-3)."""
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config


def _make_pipeline(**svm_kwargs) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel=config.SVM_KERNEL, class_weight="balanced",
                    random_state=config.RANDOM_STATE, **svm_kwargs)),
    ])


def train_model(X: np.ndarray, y: np.ndarray, model_path: Path = config.MODEL_PATH,
                test_size: float = config.TEST_SIZE, cv_folds: int = config.CV_FOLDS,
                param_grid: Optional[dict] = None, has_separate_test: bool = False) -> dict:
    """Grid Search (5-fold CV) pada data latih -> simpan model.

    Bila has_separate_test=False, X dibagi 80/20 (stratified) dan sisanya jadi data uji.
    Bila True, seluruh X adalah data latih (data uji diberikan terpisah saat evaluasi).
    """
    classes_present, counts = np.unique(y, return_counts=True)
    if len(classes_present) < 2:
        raise ValueError("Butuh minimal 2 kelas berbeda untuk pelatihan.")
    if counts.min() < 2:
        raise ValueError(
            f"Setiap kelas butuh minimal 2 citra (kelas dengan data tersedikit hanya {counts.min()})."
        )

    idx = np.arange(len(y))
    if has_separate_test:
        tr_idx, te_idx = idx, np.array([], dtype=int)
    else:
        try:
            tr_idx, te_idx = train_test_split(
                idx, test_size=test_size, stratify=y, random_state=config.RANDOM_STATE)
        except ValueError:  # data terlalu sedikit untuk stratifikasi
            tr_idx, te_idx = train_test_split(idx, test_size=test_size, random_state=config.RANDOM_STATE)

    X_tr, y_tr = X[tr_idx], y[tr_idx]
    min_in_train = int(np.bincount(y_tr)[np.unique(y_tr)].min())
    folds = max(2, min(cv_folds, min_in_train))

    search = GridSearchCV(
        _make_pipeline(),
        param_grid or config.PARAM_GRID,
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.RANDOM_STATE),
        scoring="f1_macro",
        n_jobs=-1,
        refit=False,
    )
    search.fit(X_tr, y_tr)
    best = {k.split("__", 1)[1]: v for k, v in search.best_params_.items()}

    # Refit akhir dengan probability=True agar aplikasi dapat menampilkan tingkat keyakinan.
    with warnings.catch_warnings():  # deprecation sklearn>=1.9; requirements.txt membatasi <1.11
        warnings.simplefilter("ignore", FutureWarning)
        final = _make_pipeline(probability=True, **best).fit(X_tr, y_tr)

    bundle = {
        "model": final,
        "classes": config.CLASSES,
        "image_size": config.IMAGE_SIZE,
        "best_params": best,
        "cv_folds": folds,
        "cv_f1_macro": float(search.best_score_),
        "train_idx": tr_idx,
        "test_idx": te_idx,
        "test_source": "upload" if has_separate_test else "split",
    }
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    return bundle


def load_model(model_path: Path = config.MODEL_PATH) -> dict:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model belum dilatih: {model_path}. Jalankan 'python main.py train'.")
    return joblib.load(model_path)
