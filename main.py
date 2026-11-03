#!/usr/bin/env python3
"""CLI utama.  Contoh:  python main.py train   |   python main.py serve

Alur penelitian:  capture -> preprocess -> train -> evaluate -> predict/webcam/serve
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

from stressfer import config


def cmd_demo_data(a):
    from stressfer.demo_data import make_demo_dataset
    out = make_demo_dataset(per_class=a.per_class)
    print(f"Data SINTETIS dibuat di {out}")
    print("Hanya untuk uji pipeline. Lanjut:  python main.py preprocess --raw data/demo_raw --already-cropped")
    print("(atau unggah ZIP data/demo_raw lewat menu Data di aplikasi web)")


def cmd_capture(a):
    """Akuisisi citra via webcam. Tombol 1-4 pilih kelas, SPASI simpan, Q keluar."""
    from stressfer.face_detection import FaceDetector
    from stressfer.questionnaire import valid_subject_id
    if not valid_subject_id(a.subject):
        sys.exit("ID subjek harus 1-32 karakter: huruf/angka/_/-")
    det = FaceDetector()
    cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        sys.exit("Webcam tidak dapat dibuka.")
    label = a.label
    n_saved = 0
    print(f"Jarak {config.CAPTURE_DISTANCE_CM} cm, pencahayaan {config.CAPTURE_LUX} lux, wajah frontal.")
    print("Tombol: 1=normal 2=ringan 3=sedang 4=berat | SPASI=simpan | Q=keluar")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        view = frame.copy()
        for (x, y, w, h) in det.detect(frame)[:1]:
            cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(view, f"Kelas: {label} | tersimpan: {n_saved}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Akuisisi Citra", view)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k in (ord("1"), ord("2"), ord("3"), ord("4")):
            label = config.CLASSES[k - ord("1")]
        if k == 32:
            if not det.detect(frame):
                print("Wajah tidak terdeteksi - tidak disimpan.")
                continue
            folder = config.Workspace().split_dir(a.split) / label
            folder.mkdir(parents=True, exist_ok=True)
            name = f"{a.subject}_{int(time.time() * 1000)}.jpg"
            cv2.imwrite(str(folder / name), frame)
            n_saved += 1
            print(f"Disimpan: {folder / name}")
    cap.release()
    cv2.destroyAllWindows()


def cmd_preprocess(a):
    from stressfer.dataset import build_all, build_dataset
    ws = config.Workspace()
    if a.raw:   # folder lain, hanya data latih
        info = {"train": build_dataset(Path(a.raw), ws.dataset_file, a.already_cropped,
                                       ws.processed / "faces" / "train"), "test": None}
    else:
        info = build_all(ws, a.already_cropped)
    for name in ("train", "test"):
        i = info[name]
        if i:
            print(f"{config.SPLIT_NAMES[name]}: {i['total']} citra dipakai | " +
                  ", ".join(f"{c}={n}" for c, n in i["per_class"].items()))
            if i["skipped"]:
                print(f"  dilewati (tanpa wajah/rusak): {len(i['skipped'])}")
    if not info["test"]:
        print("Tidak ada data uji terpisah (data/test/<kelas>/): akan dibagi 80/20 saat pelatihan.")


def cmd_train(a):
    from stressfer.dataset import load_dataset, load_test_if_any
    from stressfer.evaluation import evaluate
    from stressfer.model import train_model
    ws = config.Workspace()
    X, y = load_dataset(ws.dataset_file)
    Xt, yt = load_test_if_any(ws)
    print(f"Melatih SVM-RBF pada {len(y)} sampel x {X.shape[1]} fitur (Grid Search, 5-fold CV)...")
    bundle = train_model(X, y, model_path=ws.model_path, has_separate_test=Xt is not None)
    print("Parameter terbaik:", bundle["best_params"], f"| CV F1-macro={bundle['cv_f1_macro']:.4f}")
    _print_metrics(evaluate(bundle, X, y, ws.reports, Xt, yt))
    print(f"Model: {ws.model_path}\nLaporan: {ws.reports}")


def cmd_evaluate(a):
    from stressfer.dataset import load_dataset, load_test_if_any
    from stressfer.evaluation import evaluate
    from stressfer.model import load_model
    ws = config.Workspace()
    X, y = load_dataset(ws.dataset_file)
    Xt, yt = load_test_if_any(ws)
    _print_metrics(evaluate(load_model(ws.model_path), X, y, ws.reports, Xt, yt))


def _print_metrics(m):
    src = "data uji terpisah" if m.get("test_source") == "upload" else "split 80/20"
    print(f"\nData uji ({src}): {m['n_test']} | Accuracy={m['accuracy']:.4f} "
          f"Precision={m['precision_macro']:.4f} Recall={m['recall_macro']:.4f} F1={m['f1_macro']:.4f} (macro)")
    print("Confusion matrix (baris=aktual, kolom=prediksi):")
    for c, row in zip(m["classes"], m["confusion_matrix"]):
        print(f"  {c:7s}", row)


def cmd_predict(a):
    from stressfer.predictor import StressPredictor
    img = cv2.imread(a.image)
    if img is None:
        sys.exit(f"Tidak dapat membaca citra: {a.image}")
    p = StressPredictor()
    r = p.predict(img)
    if not r["face_found"]:
        sys.exit("Wajah tidak terdeteksi.")
    print(f"{r['label_text']}  (keyakinan {r['confidence'] * 100:.1f}%)")
    for c, v in r["probabilities"].items():
        print(f"  {c:7s} {v * 100:5.1f}%")
    if a.output:
        cv2.imwrite(a.output, p.annotate(img, r))
        print("Hasil anotasi:", a.output)


def cmd_webcam(a):
    from stressfer.predictor import StressPredictor
    p = StressPredictor()
    cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        sys.exit("Webcam tidak dapat dibuka.")
    print("Tekan Q untuk keluar.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imshow("Deteksi Stres Akademik (real-time)", p.annotate(frame, p.predict(frame)))
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()


def _ask_password():
    from getpass import getpass
    pw = getpass("Password: ")
    if pw != getpass("Ulangi password: "):
        sys.exit("Error: password tidak sama.")
    return pw


def cmd_create_user(a):
    from webapp.auth import UserStore
    store = UserStore(config.Workspace().users_file)
    u = store.create(a.username, _ask_password(), a.name or a.username, a.role)
    print(f"Akun '{u['username']}' ({u['role']}) dibuat. Login di halaman web (python main.py serve).")


def cmd_set_password(a):
    from webapp.auth import UserStore
    UserStore(config.Workspace().users_file).set_password(a.username, _ask_password())
    print(f"Password '{a.username.lower()}' diperbarui.")


def _bootstrap_admin():
    """Pertama kali dijalankan: buat akun admin 'admin' dengan password ACAK (ditampilkan sekali di terminal).
    Tidak ada password bawaan yang bisa ditebak. Bisa ditetapkan lewat STRESSFER_ADMIN_PASSWORD."""
    import os
    import secrets
    from webapp.auth import UserStore
    store = UserStore(config.Workspace().users_file)
    if store.has_users():
        return
    pw = os.environ.get("STRESSFER_ADMIN_PASSWORD") or secrets.token_urlsafe(9)
    store.create("admin", pw, "Administrator", "admin")
    shown = "(sesuai STRESSFER_ADMIN_PASSWORD)" if os.environ.get("STRESSFER_ADMIN_PASSWORD") else pw
    print("=" * 62)
    print(" Akun admin dibuat otomatis (hanya ditampilkan sekali):")
    print(f"   username : admin")
    print(f"   password : {shown}")
    print(" Ganti kapan saja: python main.py set-password admin")
    print("=" * 62, flush=True)


def cmd_serve(a):
    from webapp.app import create_app
    _bootstrap_admin()
    create_app().run(host=a.host, port=a.port, debug=a.debug)


def main():
    ap = argparse.ArgumentParser(description="Deteksi & klasifikasi tingkat stres akademik (Haar Cascade + SVM)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("demo-data", help="buat data sintetis untuk uji pipeline")
    s.add_argument("--per-class", type=int, default=40)
    s.set_defaults(fn=cmd_demo_data)

    s = sub.add_parser("capture", help="akuisisi citra dari webcam")
    s.add_argument("--subject", required=True, help="ID responden (anonim), mis. R001")
    s.add_argument("--label", choices=config.CLASSES, default="normal")
    s.add_argument("--split", choices=["train", "test"], default="train", help="simpan sebagai data latih/uji")
    s.add_argument("--camera", type=int, default=0)
    s.set_defaults(fn=cmd_capture)

    s = sub.add_parser("preprocess", help="pre-processing + Haar Cascade + flattening -> dataset.npz")
    s.add_argument("--raw", default=None, help="folder data latih lain (default: data/raw dan data/test)")
    s.add_argument("--already-cropped", action="store_true", help="lewati deteksi wajah (citra sudah berupa wajah)")
    s.set_defaults(fn=cmd_preprocess)

    sub.add_parser("train", help="latih SVM + evaluasi").set_defaults(fn=cmd_train)
    sub.add_parser("evaluate", help="evaluasi ulang model pada data uji").set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("predict", help="prediksi satu citra")
    s.add_argument("image")
    s.add_argument("--output", help="simpan citra beranotasi")
    s.set_defaults(fn=cmd_predict)

    s = sub.add_parser("webcam", help="prediksi real-time dari webcam")
    s.add_argument("--camera", type=int, default=0)
    s.set_defaults(fn=cmd_webcam)

    s = sub.add_parser("create-user", help="buat akun login (password diminta tanpa tampil)")
    s.add_argument("username")
    s.add_argument("--name", help="nama tampilan")
    s.add_argument("--role", choices=["admin", "mahasiswa"], default="admin", help="peran akun (default admin)")
    s.set_defaults(fn=cmd_create_user)

    s = sub.add_parser("set-password", help="atur ulang password akun")
    s.add_argument("username")
    s.set_defaults(fn=cmd_set_password)

    s = sub.add_parser("serve", help="jalankan aplikasi web")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=5000)
    s.add_argument("--debug", action="store_true")
    s.set_defaults(fn=cmd_serve)

    a = ap.parse_args()
    try:
        a.fn(a)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
