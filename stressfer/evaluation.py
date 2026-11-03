"""Evaluasi: Confusion Matrix, Accuracy, Precision, Recall, F1-Score (Bab 3.7 poin 4)."""
import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

from . import config


def evaluate(bundle: dict, X: np.ndarray, y: np.ndarray, out_dir: Path = config.REPORTS_DIR,
             X_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None) -> dict:
    """Evaluasi pada data uji: (X_test, y_test) bila diberikan, jika tidak pada bagian uji hasil split."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if X_test is not None:
        X_te, y_te = X_test, y_test
    else:
        te = bundle["test_idx"]
        X_te, y_te = X[te], y[te]
    if len(y_te) == 0:
        raise ValueError("Data uji kosong.")
    pred = bundle["model"].predict(X_te)

    labels = list(range(len(config.CLASSES)))
    cm = confusion_matrix(y_te, pred, labels=labels)
    p, r, f, s = precision_recall_fscore_support(y_te, pred, labels=labels, zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(
        y_te, pred, labels=labels, average="macro", zero_division=0)

    metrics = {
        "n_train": int(len(bundle["train_idx"])),
        "n_test": int(len(y_te)),
        "test_source": bundle.get("test_source", "split"),
        "best_params": {k: (v if isinstance(v, str) else float(v)) for k, v in bundle["best_params"].items()},
        "cv_folds": bundle["cv_folds"],
        "cv_f1_macro": bundle["cv_f1_macro"],
        "accuracy": float(accuracy_score(y_te, pred)),
        "precision_macro": float(pm),
        "recall_macro": float(rm),
        "f1_macro": float(fm),
        "per_class": {
            c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, c in enumerate(config.CLASSES)
        },
        "confusion_matrix": cm.tolist(),
        "classes": config.CLASSES,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "classification_report.txt").write_text(
        classification_report(y_te, pred, labels=labels, target_names=config.CLASSES, zero_division=0),
        encoding="utf-8")
    _plot_confusion_matrix(cm, out_dir / "confusion_matrix.png")
    return metrics


def _plot_confusion_matrix(cm: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=130)
    im = ax.imshow(cm, cmap="Blues")
    names = [config.CLASS_LABELS[c].replace(" (Normal)", "") for c in config.CLASSES]
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title("Confusion Matrix")
    thresh = cm.max() / 2 if cm.max() else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def load_metrics(out_dir: Path = config.REPORTS_DIR):
    f = Path(out_dir) / "metrics.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
