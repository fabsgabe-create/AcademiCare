"""Pelatihan di latar belakang (dipakai aplikasi web): pre-processing -> Grid Search SVM -> evaluasi."""
import threading
import time
from typing import Callable, Optional

from . import config
from .dataset import build_all, load_dataset, load_test_if_any
from .evaluation import evaluate
from .model import train_model


class TrainingJob:
    def __init__(self, ws: config.Workspace, on_done: Optional[Callable[[], None]] = None):
        self.ws = ws
        self.on_done = on_done
        self._lock = threading.Lock()
        self._state = self._fresh()

    @staticmethod
    def _fresh():
        return {"status": "idle", "stage": None, "progress": 0, "log": [], "error": None,
                "metrics": None, "started": None, "finished": None}

    def snapshot(self) -> dict:
        with self._lock:
            s = dict(self._state)
            s["log"] = list(s["log"])
            return s

    def _set(self, **kw):
        with self._lock:
            self._state.update(kw)

    def _log(self, msg: str):
        with self._lock:
            self._state["log"].append(msg)

    def start(self, already_cropped: bool = False) -> bool:
        with self._lock:
            if self._state["status"] == "running":
                return False
            self._state = self._fresh()
            self._state.update(status="running", stage="preprocess", started=time.time())
        threading.Thread(target=self._run, args=(already_cropped,), daemon=True).start()
        return True

    def _run(self, already_cropped: bool):
        ws = self.ws
        try:
            self._log("Pre-processing: grayscale, resize 100x100, Gaussian Blur" +
                      ("" if already_cropped else ", deteksi wajah Haar Cascade") + ".")
            last = {"t": 0.0}

            def prog(i, n, name):
                if time.time() - last["t"] > 0.25 or i == n:
                    last["t"] = time.time()
                    self._set(progress=int(60 * i / max(n, 1)))

            info = build_all(ws, already_cropped, prog)
            tr, te = info["train"], info["test"]
            self._log(f"Data latih: {tr['total']} citra dipakai" +
                      (f", {len(tr['skipped'])} dilewati (wajah tidak terdeteksi/rusak)." if tr["skipped"] else "."))
            if te:
                self._log(f"Data uji: {te['total']} citra dipakai" +
                          (f", {len(te['skipped'])} dilewati." if te["skipped"] else "."))
            else:
                self._log("Tidak ada data uji terpisah: data dibagi 80% latih / 20% uji secara acak (stratified).")

            self._set(stage="train", progress=62)
            self._log("Grid Search C dan gamma (kernel RBF) dengan 5-fold cross-validation. Tahap ini paling lama.")
            X, y = load_dataset(ws.dataset_file)
            Xt, yt = load_test_if_any(ws)
            bundle = train_model(X, y, model_path=ws.model_path, has_separate_test=Xt is not None)
            self._log(f"Parameter terbaik: C={bundle['best_params']['C']}, gamma={bundle['best_params']['gamma']} "
                      f"(CV F1 macro {bundle['cv_f1_macro'] * 100:.1f}%).")

            self._set(stage="evaluate", progress=92)
            metrics = evaluate(bundle, X, y, ws.reports, Xt, yt)
            self._log(f"Evaluasi selesai: akurasi {metrics['accuracy'] * 100:.1f}% pada {metrics['n_test']} citra uji.")
            self._set(status="done", stage="done", progress=100, metrics=metrics, finished=time.time())
            if self.on_done:
                self.on_done()
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            self._log(f"Gagal: {e}")
            self._set(status="error", error=str(e), finished=time.time())
        except Exception as e:  # noqa: BLE001 - jangan biarkan thread mati tanpa laporan
            self._log(f"Kesalahan tak terduga: {type(e).__name__}: {e}")
            self._set(status="error", error=f"{type(e).__name__}: {e}", finished=time.time())
