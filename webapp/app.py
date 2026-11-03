"""Aplikasi web Flask: dashboard, data (unggah/kamera), pelatihan, evaluasi, prediksi, kuesioner."""
import base64
import math
import os
import re
import time
from datetime import timedelta
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from flask import (Flask, flash, get_flashed_messages, Response, abort, jsonify, redirect, render_template, request, send_from_directory,
                   session, url_for)

from stressfer import config, questionnaire, uploads
from stressfer.dataset import count_images
from stressfer.evaluation import load_metrics
from stressfer.face_detection import FaceDetector
from stressfer.jobs import TrainingJob
from stressfer.predictor import StressPredictor
from webapp import auth, content
from webapp.history import HistoryStore
from webapp.submissions import SubmissionStore

_DATA_URL = re.compile(r"^data:image/(jpeg|png);base64,(.+)$", re.S)


def _decode_bytes(raw: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR) if arr.size else None


def _decode_data_url(s: str) -> Optional[np.ndarray]:
    m = _DATA_URL.match(s or "")
    if not m:
        return None
    try:
        return _decode_bytes(base64.b64decode(m.group(2)))
    except Exception:
        return None


def _encode_jpeg_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def create_app(model_path=None, workspace: Optional[config.Workspace] = None) -> Flask:
    ws = workspace or config.Workspace()
    if model_path:
        ws.model_path = Path(model_path)

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 700 * 1024 * 1024   # ZIP dataset
    app.config["MAX_FORM_PARTS"] = 5000
    app.secret_key = auth.load_secret(ws.secret_file)
    app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                      SESSION_COOKIE_SECURE=bool(os.environ.get("STRESSFER_HTTPS")),
                      PERMANENT_SESSION_LIFETIME=timedelta(hours=8))
    store = auth.UserStore(ws.users_file)
    history = HistoryStore(ws.history_dir)
    subs = SubmissionStore(ws.submissions_dir)
    throttle = auth.LoginThrottle()
    reg_throttle = auth.LoginThrottle(max_user=10, max_ip=10, window=3600)   # pendaftaran: maks 10/jam/IP
    PUBLIC = {"home", "pengumuman", "faq", "kontak", "login", "setup", "daftar", "logout", "static"}
    PUBLIC_PAGES = {"home", "pengumuman", "faq", "kontak", "login", "setup", "daftar"}
    # Mahasiswa hanya boleh endpoint ini; semua endpoint lain (data, latih, evaluasi, kamera, laporan, pengguna) khusus admin.
    STUDENT = {"mahasiswa_home", "predict_page", "api_predict", "questionnaire_page", "riwayat", "kirim", "hasil_kiriman"}
    state = {"predictor": None, "detector": None}

    def detector() -> FaceDetector:
        if state["detector"] is None:
            state["detector"] = FaceDetector()
        return state["detector"]

    def predictor() -> Optional[StressPredictor]:
        if state["predictor"] is None:
            try:
                state["predictor"] = StressPredictor(ws.model_path, detector())
            except FileNotFoundError:
                return None
        return state["predictor"]

    def reset_predictor():
        state["predictor"] = None

    job = TrainingJob(ws, on_done=reset_predictor)

    def all_counts():
        return {"train": count_images(ws.train_dir), "test": count_images(ws.test_dir)}

    @app.context_processor
    def inject():
        u = current_user()
        return {"user": u, "dash_url": url_for("mahasiswa_home" if u and u["role"] == "mahasiswa" else "index"),
                "public_pages": PUBLIC_PAGES, "classes": config.CLASSES, "class_labels": config.CLASS_LABELS,
                "class_short": config.CLASS_SHORT, "class_desc": config.CLASS_DESCRIPTIONS,
                "split_names": config.SPLIT_NAMES, "disclaimer": config.DISCLAIMER}

    def current_user():
        uid = session.get("uid")
        return store.get(uid) if uid else None

    @app.before_request
    def gate():
        """Default-deny: semua endpoint selain halaman publik wajib login."""
        if request.endpoint is None or request.endpoint in PUBLIC:
            return None
        u = current_user()
        if u is not None:
            if u["role"] == "admin" or request.endpoint in STUDENT:
                return None
            if request.path.startswith("/api/"):
                return jsonify(error="Fitur ini khusus admin."), 403
            return render_template("forbidden.html"), 403
        session.pop("uid", None)
        if request.path.startswith("/api/"):
            return jsonify(error="Sesi berakhir atau belum masuk. Silakan login lagi."), 401
        return redirect(url_for("login", next=request.full_path.rstrip("?")))

    @app.after_request
    def secure_headers(resp):
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "same-origin")
        if request.endpoint not in ("static", None):
            resp.headers.setdefault("Cache-Control", "no-store")
        return resp

    def loopback() -> bool:
        return request.remote_addr in ("127.0.0.1", "::1") or bool(os.environ.get("STRESSFER_ALLOW_REMOTE_SETUP"))

    def log_in(username: str):
        session.clear()                      # cegah session fixation
        session["uid"] = username
        session.permanent = True

    # ---- Halaman publik ---------------------------------------------------------
    @app.get("/")
    def home():
        return render_template("home.html", announcements=content.ANNOUNCEMENTS[:3], faq=content.FAQ[:4],
                               setup_needed=not store.has_users(), can_setup=loopback())

    @app.get("/pengumuman")
    def pengumuman():
        return render_template("pengumuman.html", announcements=content.ANNOUNCEMENTS)

    @app.get("/faq")
    def faq():
        return render_template("faq.html", faq=content.FAQ)

    @app.get("/kontak")
    def kontak():
        return render_template("kontak.html", contact=config.CONTACT)

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if not store.has_users():
            return redirect(url_for("setup"))
        nxt = auth.safe_next(request.values.get("next"))
        if current_user() is not None and request.method == "GET":
            return redirect(nxt or url_for("mahasiswa_home" if current_user()["role"] == "mahasiswa" else "index"))
        error, username, status = None, "", 200
        if request.method == "POST":
            username = request.form.get("username", "").strip()[:64]
            password = request.form.get("password", "")
            ip = request.remote_addr or "?"
            wait = throttle.wait_seconds(ip, username)
            if wait:
                error, status = f"Terlalu banyak percobaan gagal. Coba lagi dalam {math.ceil(wait / 60)} menit.", 429
            elif store.verify(username, password):
                throttle.reset(ip, username)
                log_in(username.lower())
                role = store.get(username)["role"]
                if role == "mahasiswa":
                    return redirect(url_for("mahasiswa_home"))      # tujuan ?next= tidak dipakai (hampir semua khusus admin)
                return redirect(nxt or url_for("index"))
            else:
                throttle.fail(ip, username)
                error, status = "Username atau password salah.", 401
        return render_template("login.html", error=error, username=username, next=nxt), status

    @app.post("/logout")
    def logout():
        session.clear()
        return redirect(url_for("home"))

    @app.route("/setup", methods=["GET", "POST"])
    def setup():
        """Pembuatan akun pertama. Hanya bila belum ada akun, dan hanya dari komputer yang sama."""
        if store.has_users():
            return redirect(url_for("login"))
        if not loopback():
            return render_template("setup.html", blocked=True, error=None, form={}), 403
        error, form = None, {}
        if request.method == "POST":
            form = {"username": request.form.get("username", "").strip(), "name": request.form.get("name", "").strip()}
            pw, pw2 = request.form.get("password", ""), request.form.get("password2", "")
            try:
                if pw != pw2:
                    raise ValueError("Konfirmasi password tidak sama.")
                u = store.create(form["username"], pw, form["name"])
            except ValueError as e:
                error = str(e)
            else:
                log_in(u["username"])
                return redirect(url_for("index"))
        return render_template("setup.html", blocked=False, error=error, form=form), (400 if error else 200)

    @app.route("/daftar", methods=["GET", "POST"])
    def daftar():
        """Pendaftaran mandiri mahasiswa (NIM sebagai username). Akun mahasiswa tidak bisa mengakses data penelitian."""
        if not store.has_users():
            return redirect(url_for("setup"))
        if current_user() is not None:
            return redirect(dash())
        error, form, status = None, {}, 200
        if request.method == "POST":
            form = {"name": request.form.get("name", "").strip()[:60], "nim": request.form.get("nim", "").strip()[:32]}
            pw, pw2 = request.form.get("password", ""), request.form.get("password2", "")
            ip = request.remote_addr or "?"
            try:
                if reg_throttle.wait_seconds(ip, ""):
                    status = 429
                    raise ValueError("Terlalu banyak pendaftaran dari alamat ini. Coba lagi nanti.")
                if not form["name"]:
                    raise ValueError("Nama lengkap wajib diisi.")
                if not questionnaire.valid_subject_id(form["nim"]):
                    raise ValueError("NIM 3-32 karakter: hanya huruf, angka, _ atau -.")
                if not request.form.get("consent"):
                    raise ValueError("Centang persetujuan penggunaan data terlebih dahulu.")
                if pw != pw2:
                    raise ValueError("Konfirmasi password tidak sama.")
                reg_throttle.fail(ip, "")
                u = store.create(form["nim"], pw, form["name"], role="mahasiswa")
            except ValueError as e:
                error, status = str(e), (status if status != 200 else 400)
            else:
                log_in(u["username"])
                return redirect(url_for("mahasiswa_home"))
        return render_template("daftar.html", error=error, form=form), status

    def dash() -> str:
        u = current_user()
        return url_for("mahasiswa_home" if u and u["role"] == "mahasiswa" else "index")

    # ---- Area mahasiswa -------------------------------------------------------
    @app.get("/mahasiswa")
    def mahasiswa_home():
        u = current_user()
        return render_template("mahasiswa.html", model_ready=ws.model_path.exists(),
                               last_face=history.last(u["username"], "wajah"),
                               last_q=history.last(u["username"], "kuesioner"),
                               n_hist=len(history.list(u["username"])),
                               n_sub=len(subs.list(u["username"])),
                               last_sub=(subs.list(u["username"]) or [None])[0])

    @app.get("/riwayat")
    def riwayat():
        return render_template("riwayat.html", items=history.list(current_user()["username"]))

    # ---- Pengelolaan pengguna (admin) -------------------------------------------
    @app.get("/pengguna")
    def pengguna():
        rows = []
        for u in store.list():
            rows.append({**u, "n": len(history.list(u["username"])) if u["role"] == "mahasiswa" else None,
                         "last": history.last(u["username"]) if u["role"] == "mahasiswa" else None})
        return render_template("pengguna.html", rows=rows, msgs=get_flashed_messages(with_categories=True))

    @app.post("/pengguna/hapus")
    def pengguna_hapus():
        target = request.form.get("username", "")
        if target == current_user()["username"]:
            flash("Anda tidak dapat menghapus akun yang sedang dipakai.", "err")
        else:
            try:
                if store.delete(target):
                    history.clear(target.strip().lower())
                    subs.delete_user(target.strip().lower())
                    flash(f"Akun {target} dihapus.", "ok")
                else:
                    flash("Akun tidak ditemukan.", "err")
            except ValueError as e:
                flash(str(e), "err")
        return redirect(url_for("pengguna"))

    @app.post("/pengguna/password")
    def pengguna_password():
        target = request.form.get("username", "")
        try:
            store.set_password(target, request.form.get("password", ""))
            flash(f"Password {target} diperbarui. Sampaikan ke pemilik akun lewat jalur aman.", "ok")
        except ValueError as e:
            flash(str(e), "err")
        return redirect(url_for("pengguna"))

    @app.route("/kirim", methods=["GET", "POST"])
    def kirim():
        """Mahasiswa mengirim foto wajah untuk penelitian. Label berasal dari kuesioner terakhir; admin yang meninjau."""
        u = current_user()
        q = history.last(u["username"], "kuesioner")
        error = ok = None
        if request.method == "POST":
            f = request.files.get("image")
            raw = f.read(config.MAX_IMAGE_BYTES + 1) if f and f.filename else b""
            try:
                if q is None:
                    raise ValueError("Isi kuesioner dulu agar foto Anda punya label.")
                if not request.form.get("consent"):
                    raise ValueError("Centang persetujuan pengiriman foto terlebih dahulu.")
                if not raw:
                    raise ValueError("Pilih foto terlebih dahulu.")
                if len(raw) > config.MAX_IMAGE_BYTES:
                    raise ValueError("Ukuran foto maksimal 10 MB.")
                img = _decode_bytes(raw)
                if img is None:
                    raise ValueError("Berkas bukan citra JPG/PNG yang valid.")
                if not detector().detect(img):
                    raise ValueError("Wajah tidak terdeteksi. Gunakan foto frontal dengan pencahayaan baik.")
                h, w = img.shape[:2]
                k = 1280 / max(h, w)
                if k < 1:
                    img = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
                ok_enc, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                entry = subs.add(u["username"], q["label"], buf.tobytes())
                return redirect(url_for("hasil_kiriman", sid=entry["id"], baru=1))
            except ValueError as e:
                error = str(e)
        return render_template("kirim.html", error=error, ok=ok, q=q, items=subs.list(u["username"])), \
            (400 if error else 200)

    # ---- Data mahasiswa (admin) -------------------------------------------------
    def students():
        return [u for u in store.list() if u["role"] == "mahasiswa"]

    def purge(nim: str, what: str) -> None:
        """Hapus data satu mahasiswa. what: 'riwayat' | 'foto' | 'semua' (riwayat+foto)."""
        if what in ("riwayat", "semua"):
            history.clear(nim)
        if what in ("foto", "semua"):
            subs.delete_user(nim)

    def back(sub_user: str):
        if request.form.get("kembali") == "foto":
            return redirect(url_for("foto_mahasiswa"))
        return redirect(url_for("mahasiswa_detail", nim=sub_user))

    @app.get("/data-mahasiswa")
    def data_mahasiswa():
        all_subs = subs.list()
        rows = []
        for u in students():
            mine = [s for s in all_subs if s["username"] == u["username"]]
            rows.append({**u, "q": history.last(u["username"], "kuesioner"), "face": history.last(u["username"], "wajah"),
                         "n_sub": len(mine), "n_new": sum(1 for s in mine if s["status"] == "baru")})
        return render_template("data_mahasiswa.html", rows=rows, n_new=sum(1 for s in all_subs if s["status"] == "baru"))

    @app.get("/data-mahasiswa/<nim>")
    def mahasiswa_detail(nim):
        u = store.get(nim)
        if not u or u["role"] != "mahasiswa":
            abort(404)
        return render_template("mahasiswa_detail.html", s=u, items=history.list(nim), kiriman=subs.list(nim),
                               msgs=get_flashed_messages(with_categories=True))

    @app.get("/kiriman/<sid>.jpg")
    def kiriman_img(sid):
        p = subs.path(sid)
        if p is None:
            abort(404)
        return send_from_directory(p.parent, p.name, mimetype="image/jpeg", max_age=0)

    @app.post("/data-mahasiswa/impor")
    def mahasiswa_impor():
        sid, split, label = request.form.get("id", ""), request.form.get("split", ""), request.form.get("label", "")
        sub, p = subs.get(sid), subs.path(sid)
        if not sub or p is None:
            abort(404)
        try:
            if split not in ("train", "test"):
                raise ValueError("Pilih data latih atau data uji.")
            uploads.store_image(p.read_bytes(), ws, split, label, prefix="mhs")
            subs.mark_used(sid, split, label)
            flash(f"Foto dimasukkan ke {config.SPLIT_NAMES[split]} kelas {config.CLASS_LABELS[label]}.", "ok")
        except ValueError as e:
            flash(str(e), "err")
        return back(sub["username"])

    @app.post("/data-mahasiswa/hapus-kiriman")
    def kiriman_hapus():
        sub = subs.get(request.form.get("id", ""))
        if not sub:
            abort(404)
        subs.delete(sub["id"])
        flash("Kiriman dihapus.", "ok")
        return back(sub["username"])

    @app.post("/data-mahasiswa/hapus-data")
    def mahasiswa_hapus_data():
        nim, what = request.form.get("nim", "").strip().lower(), request.form.get("apa", "")
        u = store.get(nim)
        if not u or u["role"] != "mahasiswa" or what not in ("riwayat", "foto", "semua", "akun"):
            abort(404)
        if what == "akun":
            purge(nim, "semua")
            store.delete(nim)
            flash(f"Akun {nim} beserta seluruh datanya dihapus.", "ok")
            return redirect(url_for("data_mahasiswa"))
        purge(nim, what)
        flash({"riwayat": "Riwayat hasil dihapus.", "foto": "Semua foto kiriman dihapus.",
               "semua": "Riwayat dan foto dihapus (akun tetap ada)."}[what], "ok")
        return redirect(url_for("mahasiswa_detail", nim=nim))

    @app.get("/foto-mahasiswa")
    def foto_mahasiswa():
        status = request.args.get("status", "semua")
        label = request.args.get("label", "")
        offset = max(0, request.args.get("offset", 0, type=int))
        names = {u["username"]: u["name"] for u in store.list()}
        items = [x for x in subs.list()
                 if (status == "semua" or x["status"] == status) and (not label or x["label"] == label)]
        page = [{**x, "name": names.get(x["username"], x["username"])} for x in items[offset:offset + 24]]
        return render_template("foto_mahasiswa.html", items=page, total=len(items), offset=offset, status=status, label=label,
                               msgs=get_flashed_messages(with_categories=True))

    @app.get("/kiriman/<sid>/thumb.jpg")
    def kiriman_thumb(sid):
        p = subs.path(sid)
        img = cv2.imread(str(p)) if p else None
        if img is None:
            abort(404)
        h, w = img.shape[:2]
        k = 260 / max(h, w)
        if k < 1:
            img = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return Response(buf.tobytes(), mimetype="image/jpeg", headers={"Cache-Control": "private, max-age=60"})

    @app.get("/hasil/<sid>")
    def hasil_kiriman(sid):
        """Hasil deteksi untuk satu foto kiriman: kotak wajah, tingkat stres model, dan perbandingan dengan label kuesioner."""
        u = current_user()
        sub, p = subs.get(sid), subs.path(sid)
        if not sub or p is None or (u["role"] != "admin" and sub["username"] != u["username"]):
            abort(404)                       # mahasiswa lain tidak bisa melihat (dan tidak tahu ada/tidaknya)
        img = _decode_bytes(p.read_bytes())
        if img is None:
            abort(404)
        pr, result = predictor(), None
        if pr is not None:
            result = pr.predict(img)
            annotated = _encode_jpeg_b64(pr.annotate(img, result))
        else:                                # model belum ada: tetap tampilkan deteksi wajah (kotak saja)
            _, box = detector().extract_roi(img)
            out = img.copy()
            if box is not None:
                x, y, w, h = box
                cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), (50, 155, 255), 2)
            annotated, result = _encode_jpeg_b64(out), {"face_found": box is not None}
        owner = store.get(sub["username"]) or {"name": sub["username"], "username": sub["username"]}
        return render_template("hasil.html", sub=sub, owner=owner, result=result, annotated=annotated,
                               model_ready=pr is not None, baru=bool(request.args.get("baru")))

    @app.get("/data-mahasiswa.csv")
    def mahasiswa_csv():
        import csv
        import io

        def cell(v):                      # cegah CSV/formula injection dari nama yang diisi pengguna
            v = str(v if v is not None else "")
            return "'" + v if v[:1] in ("=", "+", "-", "@", "\t", "\r") else v
        out = io.StringIO()
        w = csv.writer(out)
        w.writerow(["nim", "nama", "waktu", "jenis", "hasil", "keterangan"])
        for u in students():
            for e in reversed(history.list(u["username"])):
                w.writerow([cell(u["username"]), cell(u["name"]), e["time"], e["jenis"], e["label"], cell(e["detail"])])
        return Response(out.getvalue(), mimetype="text/csv; charset=utf-8",
                        headers={"Content-Disposition": "attachment; filename=data_mahasiswa.csv"})

    # ---- Dashboard -----------------------------------------------------------
    @app.get("/dashboard")
    def index():
        counts = all_counts()
        model_ready = ws.model_path.exists()
        n_train, n_test = sum(counts["train"].values()), sum(counts["test"].values())
        if n_train == 0:
            nxt = {"href": "/data", "title": "Unggah data latih",
                   "text": "Belum ada citra. Unggah beberapa citra wajah per tingkat stres atau ambil lewat kamera."}
        elif not model_ready:
            nxt = {"href": "/train", "title": "Latih model",
                   "text": f"{n_train} citra latih siap. Jalankan pre-processing dan pelatihan SVM."}
        else:
            nxt = {"href": "/predict", "title": "Coba prediksi",
                   "text": "Model sudah dilatih. Unggah citra atau nyalakan webcam."}
        return render_template("index.html", counts=counts, n_train=n_train, n_test=n_test,
                               model_ready=model_ready, metrics=load_metrics(ws.reports), nxt=nxt,
                               n_new_subs=sum(1 for x in subs.list() if x["status"] == "baru"))

    # ---- Data: unggah, kamera, hapus ------------------------------------------
    @app.get("/data")
    def data_page():
        return render_template("data.html", counts=all_counts())

    @app.get("/api/dataset")
    def api_dataset():
        return jsonify(all_counts())

    @app.post("/api/upload")
    def api_upload():
        split = request.form.get("split", "train")
        label = request.form.get("label") or None
        if split not in ("train", "test", "auto"):
            return jsonify(error="Tujuan harus data latih, data uji, atau otomatis."), 400
        if label is not None and label not in config.CLASSES:
            return jsonify(error="Kelas tidak valid."), 400
        files = request.files.getlist("files")
        if not files:
            return jsonify(error="Belum ada berkas yang dipilih."), 400
        report = {"saved": 0, "rejected": [], "per": {}}
        image_items = []
        for f in files:
            raw = f.read()
            if (f.filename or "").lower().endswith(".zip") or raw[:4] == b"PK\x03\x04":
                try:
                    r = uploads.save_zip(raw, ws, None if split == "auto" else split, label)
                except ValueError as e:
                    report["rejected"].append({"name": (f.filename or "zip")[:80], "reason": str(e)})
                    continue
                report["saved"] += r["saved"]
                report["rejected"] += r["rejected"]
                for k, v in r["per"].items():
                    report["per"][k] = report["per"].get(k, 0) + v
            else:
                image_items.append((f.filename or "", raw))
        if image_items:
            if split == "auto":
                return jsonify(error="Mode otomatis hanya untuk ZIP. Pilih data latih atau data uji."), 400
            if label is None:
                return jsonify(error="Pilih kelas untuk citra yang diunggah."), 400
            r = uploads.save_images(image_items, ws, split, label)
            report["saved"] += r["saved"]
            report["rejected"] += r["rejected"]
            for k, v in r["per"].items():
                report["per"][k] = report["per"].get(k, 0) + v
        report["rejected_total"] = len(report["rejected"])
        report["rejected"] = report["rejected"][:20]
        report["counts"] = all_counts()
        return jsonify(report)

    @app.post("/api/dataset/clear")
    def api_clear():
        data = request.get_json(silent=True) or {}
        try:
            n = uploads.clear_images(ws, data.get("split", ""), data.get("label") or None)
        except ValueError as e:
            return jsonify(error=str(e)), 400
        return jsonify(removed=n, counts=all_counts())

    @app.get("/api/dataset/list")
    def api_dataset_list():
        try:
            names, total = uploads.list_images_page(
                ws, request.args.get("split", ""), request.args.get("label", ""),
                max(0, request.args.get("offset", 0, type=int)),
                min(60, max(1, request.args.get("limit", 24, type=int))))
        except ValueError as e:
            return jsonify(error=str(e)), 400
        return jsonify(names=names, total=total)

    @app.get("/thumb/<split>/<label>/<name>")
    def thumb(split, label, name):
        p = uploads.image_path(ws, split, label, name)
        if p is None:
            abort(404)
        img = cv2.imread(str(p))
        if img is None:
            abort(404)
        h, w = img.shape[:2]
        k = 220 / max(h, w)
        if k < 1:
            img = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return Response(buf.tobytes(), mimetype="image/jpeg", headers={"Cache-Control": "private, max-age=300"})

    @app.post("/api/dataset/delete")
    def api_delete_one():
        d = request.get_json(silent=True) or {}
        if not uploads.delete_image(ws, d.get("split", ""), d.get("label", ""), d.get("name", "")):
            return jsonify(error="Citra tidak ditemukan."), 404
        return jsonify(ok=True, counts=all_counts())

    @app.get("/capture")
    def capture_page():
        return render_template("capture.html", subject=request.args.get("subject", ""),
                               label=request.args.get("label", "normal"),
                               split=request.args.get("split", "train"),
                               counts=all_counts(), lux=config.CAPTURE_LUX, dist=config.CAPTURE_DISTANCE_CM)

    @app.post("/api/capture")
    def api_capture():
        data = request.get_json(silent=True) or {}
        sid, label = str(data.get("subject_id", "")), data.get("label")
        split = data.get("split", "train")
        if not data.get("consent"):
            return jsonify(error="Centang persetujuan responden (informed consent) dulu."), 400
        if not questionnaire.valid_subject_id(sid):
            return jsonify(error="ID responden harus 1-32 karakter: huruf, angka, _ atau -."), 400
        if label not in config.CLASSES or split not in ("train", "test"):
            return jsonify(error="Kelas atau tujuan tidak valid."), 400
        img = _decode_data_url(data.get("image", ""))
        if img is None:
            return jsonify(error="Citra tidak valid."), 400
        if not detector().detect(img):
            return jsonify(error="Wajah tidak terdeteksi, citra tidak disimpan. Hadapkan wajah ke kamera."), 422
        folder = ws.split_dir(split) / label
        folder.mkdir(parents=True, exist_ok=True)
        name = f"{sid}_{int(time.time() * 1000)}.jpg"
        cv2.imwrite(str(folder / name), img)
        return jsonify(saved=name, counts=all_counts())

    # ---- Pelatihan ------------------------------------------------------------
    @app.get("/train")
    def train_page():
        return render_template("train.html", counts=all_counts(), metrics=load_metrics(ws.reports),
                               job=job.snapshot())

    @app.post("/api/train")
    def api_train():
        data = request.get_json(silent=True) or {}
        if not job.start(bool(data.get("already_cropped"))):
            return jsonify(error="Pelatihan sedang berjalan."), 409
        return jsonify(started=True)

    @app.get("/api/train/status")
    def api_train_status():
        s = job.snapshot()
        s.pop("metrics", None)
        return jsonify(s)

    # ---- Prediksi ---------------------------------------------------------------
    @app.route("/predict", methods=["GET", "POST"])
    def predict_page():
        result = annotated = error = None
        if request.method == "POST":
            f = request.files.get("image")
            img = _decode_bytes(f.read()) if f and f.filename else None
            p = predictor()
            if p is None:
                error = "Model belum dilatih. Latih model di menu Latih."
            elif img is None:
                error = "Berkas bukan citra JPG/PNG yang valid."
            else:
                result = p.predict(img)
                annotated = _encode_jpeg_b64(p.annotate(img, result))
                u = current_user()
                if u["role"] == "mahasiswa" and result.get("face_found"):
                    history.add(u["username"], "wajah", result["label"], f"keyakinan {result['confidence'] * 100:.1f}%")
        return render_template("predict.html", result=result, annotated=annotated, error=error,
                               model_ready=ws.model_path.exists())

    @app.post("/api/predict")
    def api_predict():
        data = request.get_json(silent=True) or {}
        img = _decode_data_url(data.get("image", ""))
        if img is None:
            return jsonify(error="Citra tidak valid"), 400
        p = predictor()
        if p is None:
            return jsonify(error="Model belum dilatih"), 503
        r = p.predict(img)
        if r.get("box"):
            r["box"] = list(r["box"])
        u = current_user()
        if data.get("save") and u["role"] == "mahasiswa" and r.get("face_found"):
            history.add(u["username"], "wajah", r["label"], f"keyakinan {r['confidence'] * 100:.1f}% (webcam)")
            r["saved"] = True
        return jsonify(r)

    # ---- Kuesioner --------------------------------------------------------------
    @app.route("/questionnaire", methods=["GET", "POST"])
    def questionnaire_page():
        summary = error = None
        if request.method == "POST":
            u = current_user()
            student = u["role"] == "mahasiswa"
            sid = u["username"] if student else request.form.get("subject_id", "").strip()
            try:
                if not questionnaire.valid_subject_id(sid):
                    raise ValueError("ID responden harus 1-32 karakter: huruf, angka, _ atau -.")
                answers = [int(request.form[f"q{i + 1}"]) for i in range(len(questionnaire.ITEMS))]
                summary = questionnaire.log_response(sid, answers, ws.questionnaire)
                summary["label_text"] = config.CLASS_LABELS[summary["label"]]
                if student:
                    history.add(sid, "kuesioner", summary["label"], f"skor {summary['total']} dari {questionnaire.MAX_SCORE}")
            except (KeyError, ValueError) as e:
                error = "Jawab semua butir (0-3)." if isinstance(e, KeyError) else str(e)
        return render_template("questionnaire.html", items=questionnaire.ITEMS, likert=questionnaire.LIKERT,
                               summary=summary, error=error, max_score=questionnaire.MAX_SCORE)

    # ---- Evaluasi ---------------------------------------------------------------
    @app.get("/evaluation")
    def evaluation_page():
        return render_template("evaluation.html", m=load_metrics(ws.reports))

    @app.get("/reports/<path:name>")
    def reports(name):
        if name not in {"confusion_matrix.png", "classification_report.txt", "metrics.json"}:
            abort(404)
        return send_from_directory(ws.reports, name)

    return app


if __name__ == "__main__":
    create_app().run(debug=True)
