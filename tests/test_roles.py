"""Peran mahasiswa: daftar mandiri, pembatasan akses, riwayat tanpa citra, manajemen pengguna oleh admin."""
import pytest

from stressfer import config, questionnaire
from webapp.app import create_app
from webapp.auth import UserStore

PW = "rahasia-123"
ADMIN_ONLY = ["/dashboard", "/data-mahasiswa", "/data-mahasiswa.csv", "/data-mahasiswa/peneliti", "/kiriman/0123456789abcdef.jpg", "/data", "/train", "/evaluation", "/capture", "/pengguna", "/reports/metrics.json",
              "/thumb/train/normal/x.jpg"]
ADMIN_API_GET = ["/api/dataset", "/api/dataset/list?split=train&label=normal", "/api/train/status"]


@pytest.fixture
def ws(tmp_path):
    return config.Workspace(tmp_path)


@pytest.fixture
def app(ws):
    UserStore(ws.users_file).create("peneliti", PW, "Peneliti", "admin")
    return create_app(workspace=ws)


def register(c, nim="251321036", name="Budi", pw=PW, **extra):
    data = {"name": name, "nim": nim, "password": pw, "password2": pw, "consent": "1"}
    data.update(extra)
    return c.post("/daftar", data=data)


@pytest.fixture
def mhs(app):
    c = app.test_client()
    assert register(c).status_code == 302
    return c


@pytest.fixture
def admin(app):
    c = app.test_client()
    assert c.post("/login", data={"username": "peneliti", "password": PW}).status_code == 302
    return c


def test_register_creates_student_and_logs_in(app, ws):
    c = app.test_client()
    r = register(c)
    assert r.status_code == 302 and r.headers["Location"].endswith("/mahasiswa")
    assert UserStore(ws.users_file).get("251321036")["role"] == "mahasiswa"
    assert c.get("/mahasiswa").status_code == 200


@pytest.mark.parametrize("extra,msg", [({"consent": ""}, "persetujuan"), ({"password2": "lain-lain-1"}, "Konfirmasi"),
                                       ({"nim": "a b"}, "NIM"), ({"nim": "12"}, "NIM"), ({"pw": "pendek"}, "minimal"),
                                       ({"name": " "}, "Nama")])
def test_register_validation(app, extra, msg):
    c = app.test_client()
    data = {"name": "Budi", "nim": "251321036", "password": PW, "password2": PW, "consent": "1"}
    if "pw" in extra:
        data["password"] = data["password2"] = extra["pw"]
    else:
        data.update(extra)
    r = c.post("/daftar", data=data)
    assert r.status_code == 400 and msg in r.get_data(as_text=True)


def test_register_duplicate_and_admin_username_taken(app):
    assert register(app.test_client()).status_code == 302
    assert register(app.test_client()).status_code == 400
    assert register(app.test_client(), nim="peneliti").status_code == 400


def test_register_rate_limited(app):
    for i in range(10):
        assert register(app.test_client(), nim=f"nim{i:04d}").status_code == 302
    assert register(app.test_client(), nim="nim9999").status_code == 429


def test_register_unavailable_without_setup(tmp_path):
    c = create_app(workspace=config.Workspace(tmp_path)).test_client()
    assert c.get("/daftar").status_code == 302 and c.get("/daftar").headers["Location"].endswith("/setup")


def test_student_cannot_reach_admin_pages(mhs):
    for url in ADMIN_ONLY:
        assert mhs.get(url).status_code == 403, url
    for url in ADMIN_API_GET:
        r = mhs.get(url)
        assert r.status_code == 403 and r.is_json, url
    for url in ["/api/upload", "/api/dataset/clear", "/api/dataset/delete", "/api/train", "/api/capture",
                "/pengguna/hapus", "/pengguna/password", "/data-mahasiswa/impor"]:
        assert mhs.post(url, json={}).status_code == 403, url


def test_student_pages_ok_and_login_redirect(app, mhs):
    for url in ["/mahasiswa", "/questionnaire", "/predict", "/riwayat", "/kirim", "/", "/faq"]:
        assert mhs.get(url).status_code == 200, url
    c = app.test_client()
    r = c.post("/login", data={"username": "251321036", "password": PW})
    assert r.headers["Location"].endswith("/mahasiswa")


def test_student_questionnaire_uses_own_nim_and_history(mhs, ws):
    data = {f"q{i + 1}": "3" for i in range(len(questionnaire.ITEMS))}
    data["subject_id"] = "orang-lain"        # harus diabaikan
    r = mhs.post("/questionnaire", data=data)
    assert r.status_code == 200 and "251321036" in r.get_data(as_text=True)
    csv = (ws.questionnaire / "responses.csv").read_text()
    assert "251321036" in csv and "orang-lain" not in csv
    page = mhs.get("/riwayat").get_data(as_text=True)
    assert "Kuesioner" in page and f"skor {len(questionnaire.ITEMS) * 3}" in page


def test_student_histories_are_isolated(app, mhs):
    data = {f"q{i + 1}": "0" for i in range(len(questionnaire.ITEMS))}
    mhs.post("/questionnaire", data=data)
    other = app.test_client()
    register(other, nim="999000111", name="Siti")
    assert "Belum ada riwayat" in other.get("/riwayat").get_data(as_text=True)
    assert "skor 0" in mhs.get("/riwayat").get_data(as_text=True)


def test_student_cannot_delete_anything(mhs):
    data = {f"q{i + 1}": "1" for i in range(len(questionnaire.ITEMS))}
    mhs.post("/questionnaire", data=data)
    assert mhs.post("/riwayat/hapus").status_code in (404, 405)
    page = mhs.get("/riwayat").get_data(as_text=True)
    assert "skor 14" in page and "Hapus" not in page
    for url in ["/data-mahasiswa/hapus-kiriman", "/data-mahasiswa/impor", "/pengguna/hapus", "/api/dataset/delete"]:
        assert mhs.post(url, data={"id": "x"}, json=None).status_code == 403, url


def test_student_predict_without_model_does_not_log_or_crash(mhs):
    r = mhs.post("/api/predict", json={"image": "data:image/png;base64,AAAA", "save": True})
    assert r.status_code == 400
    assert "Belum ada riwayat" in mhs.get("/riwayat").get_data(as_text=True)


def test_admin_has_full_access_and_questionnaire_field(admin):
    for url in ["/dashboard", "/data", "/train", "/evaluation", "/capture", "/pengguna", "/predict", "/questionnaire"]:
        assert admin.get(url).status_code == 200, url
    assert 'name="subject_id"' in admin.get("/questionnaire").get_data(as_text=True)


def test_admin_manage_users(app, admin, ws):
    register(app.test_client(), nim="777000111", name="Andi")
    page = admin.get("/pengguna").get_data(as_text=True)
    assert "777000111" in page and "Andi" in page
    admin.post("/pengguna/password", data={"username": "777000111", "password": "baru-sekali-1"})
    store = UserStore(ws.users_file)
    assert store.verify("777000111", "baru-sekali-1") and not store.verify("777000111", PW)
    admin.post("/pengguna/password", data={"username": "777000111", "password": "x"})
    assert store.verify("777000111", "baru-sekali-1")           # password lemah ditolak
    admin.post("/pengguna/hapus", data={"username": "777000111"})
    assert store.get("777000111") is None


def test_admin_cannot_delete_self_or_last_admin(admin, ws):
    admin.post("/pengguna/hapus", data={"username": "peneliti"})
    assert UserStore(ws.users_file).get("peneliti") is not None
    with pytest.raises(ValueError):
        UserStore(ws.users_file).delete("peneliti")


def test_deleted_student_session_invalid(app, mhs, admin):
    admin.post("/pengguna/hapus", data={"username": "251321036"})
    assert mhs.get("/mahasiswa").status_code == 302


def test_legacy_user_without_role_is_admin(tmp_path):
    ws = config.Workspace(tmp_path)
    ws.users_file.parent.mkdir(parents=True, exist_ok=True)
    s = UserStore(ws.users_file)
    s.create("lama", PW, "Lama")
    import json
    d = json.loads(ws.users_file.read_text()); d["lama"].pop("role"); ws.users_file.write_text(json.dumps(d))
    assert s.get("lama")["role"] == "admin"


def test_forbidden_page_has_no_stack_trace(mhs):
    body = mhs.get("/data").get_data(as_text=True)
    assert "khusus admin" in body.lower() and "Traceback" not in body


class _FakePredictor:
    def __init__(self, *a, **k):
        pass

    def predict(self, img):
        return {"face_found": True, "label": "sedang", "confidence": 0.8123, "box": (1, 2, 3, 4),
                "probabilities": {c: 0.25 for c in config.CLASSES}, "label_text": "Sedang", "description": "x"}

    def annotate(self, img, r):
        return img


@pytest.fixture
def fake_model(monkeypatch, ws):
    monkeypatch.setattr("webapp.app.StressPredictor", _FakePredictor)
    class _D:
        def detect(self, img):
            return [(0, 0, 10, 10)]

        def extract_roi(self, img):
            return img, (5, 5, 30, 30)
    monkeypatch.setattr("webapp.app.FaceDetector", lambda *a, **k: _D())
    ws.model_path.parent.mkdir(parents=True, exist_ok=True)
    ws.model_path.write_bytes(b"x")


def test_student_webcam_save_logs_only_when_asked(mhs, fake_model):
    img = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    assert mhs.post("/api/predict", json={"image": img}).get_json().get("saved") is None
    assert "Belum ada riwayat" in mhs.get("/riwayat").get_data(as_text=True)
    assert mhs.post("/api/predict", json={"image": img, "save": True}).get_json()["saved"] is True
    page = mhs.get("/riwayat").get_data(as_text=True)
    assert "Prediksi wajah" in page and "81.2%" in page


def test_admin_predict_not_logged(admin, fake_model, ws):
    img = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    assert admin.post("/api/predict", json={"image": img, "save": True}).get_json().get("saved") is None
    assert not ws.history_dir.exists()


# ---------------------------------------------------------------- kiriman foto ke admin
import cv2
import numpy as np


def _jpg():
    img = np.full((120, 100, 3), 180, np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


def _send(c, consent=True, data=None):
    import io
    form = {"image": (io.BytesIO(data if data is not None else _jpg()), "foto.jpg")}
    if consent:
        form["consent"] = "1"
    return c.post("/kirim", data=form, content_type="multipart/form-data")


def _fill_q(c, v="3"):
    c.post("/questionnaire", data={f"q{i + 1}": v for i in range(len(questionnaire.ITEMS))})


@pytest.fixture
def face_ok(monkeypatch):
    class D:
        def detect(self, img):
            return [(0, 0, 10, 10)]
    monkeypatch.setattr("webapp.app.FaceDetector", lambda *a, **k: D())


def test_kirim_requires_questionnaire_consent_and_face(mhs, face_ok, monkeypatch):
    assert "kuesioner" in _send(mhs).get_data(as_text=True).lower() and _send(mhs).status_code == 400
    _fill_q(mhs)
    assert _send(mhs, consent=False).status_code == 400
    assert _send(mhs, data=b"bukan gambar").status_code == 400
    assert _send(mhs).status_code == 302
    class NoFace:
        def detect(self, img):
            return []
    monkeypatch.setattr("webapp.app.FaceDetector", lambda *a, **k: NoFace())
    # detektor sudah di-cache pada app; buat app baru untuk memastikan jalur 'tanpa wajah'
    # (cukup memastikan kiriman sukses di atas tidak menambah apa pun pada kegagalan)


def test_kirim_no_face_rejected(app, monkeypatch):
    class NoFace:
        def detect(self, img):
            return []
    monkeypatch.setattr("webapp.app.FaceDetector", lambda *a, **k: NoFace())
    c = app.test_client(); register(c); _fill_q(c)
    r = _send(c)
    assert r.status_code == 400 and "Wajah tidak terdeteksi" in r.get_data(as_text=True)


def test_kirim_limit_per_student(mhs, face_ok):
    _fill_q(mhs)
    for _ in range(20):
        assert _send(mhs).status_code == 302
    assert _send(mhs).status_code == 400


def test_admin_sees_student_data_and_imports_photo(app, mhs, admin, ws, face_ok):
    _fill_q(mhs, "3")                                  # skor 42 -> berat
    assert _send(mhs).status_code == 302
    page = admin.get("/data-mahasiswa").get_data(as_text=True)
    assert "251321036" in page and "Budi" in page and "1 baru" in page
    detail = admin.get("/data-mahasiswa/251321036").get_data(as_text=True)
    sid = detail.split("/kiriman/")[1].split("/")[0]
    img = admin.get(f"/kiriman/{sid}.jpg")
    assert img.status_code == 200 and img.mimetype == "image/jpeg"
    r = admin.post("/data-mahasiswa/impor", data={"id": sid, "split": "train", "label": "berat"})
    assert r.status_code == 302
    assert len(list((ws.train_dir / "berat").glob("mhs_*.jpg"))) == 1
    assert "Masuk" in admin.get("/data-mahasiswa/251321036").get_data(as_text=True)
    assert admin.post("/data-mahasiswa/impor", data={"id": sid, "split": "salah", "label": "berat"}).status_code == 302
    assert admin.post("/data-mahasiswa/impor", data={"id": "../../etc", "split": "train", "label": "berat"}).status_code == 404


def test_student_photo_hidden_from_other_students(app, mhs, face_ok):
    _fill_q(mhs); _send(mhs)
    other = app.test_client(); register(other, nim="999000111", name="Siti")
    assert other.get("/kiriman/0123456789abcdef.jpg").status_code == 403
    assert "Belum ada foto" in other.get("/kirim").get_data(as_text=True)


def test_admin_delete_submission_and_user_cleanup(app, mhs, admin, ws, face_ok):
    _fill_q(mhs); _send(mhs)
    sid = admin.get("/data-mahasiswa/251321036").get_data(as_text=True).split("/kiriman/")[1].split("/")[0]
    assert admin.post("/data-mahasiswa/hapus-kiriman", data={"id": sid}).status_code == 302
    assert admin.get(f"/kiriman/{sid}.jpg").status_code == 404
    _send(mhs)
    admin.post("/pengguna/hapus", data={"username": "251321036"})
    assert not list(ws.submissions_dir.glob("*.jpg"))
    assert not (ws.history_dir / "251321036.json").exists()


def test_csv_export_admin_only_and_sanitized(app, admin, mhs):
    c = app.test_client(); register(c, nim="888000111", name="=HYPERLINK(1)"); _fill_q(c)
    body = admin.get("/data-mahasiswa.csv").get_data(as_text=True)
    assert "nim,nama,waktu" in body and "'=HYPERLINK" in body and ",=HYPERLINK" not in body
    assert mhs.get("/data-mahasiswa.csv").status_code == 403


def test_bootstrap_admin_random_password(tmp_path, monkeypatch, capsys):
    import main
    real = config.Workspace
    monkeypatch.setattr(main.config, "Workspace", lambda *a, **k: real(tmp_path))
    monkeypatch.delenv("STRESSFER_ADMIN_PASSWORD", raising=False)
    main._bootstrap_admin()
    out = capsys.readouterr().out
    pw = out.split("password : ")[1].split()[0]
    store = UserStore(real(tmp_path).users_file)
    assert store.get("admin")["role"] == "admin" and store.verify("admin", pw) and len(pw) >= 12
    main._bootstrap_admin()                       # tidak mengubah apa pun bila sudah ada akun
    assert capsys.readouterr().out == "" and store.verify("admin", pw)


# ---------------------------------------------------------------- hasil deteksi, foto mahasiswa, hapus data
def _sid_of(client):
    return client.get("/foto-mahasiswa").get_data(as_text=True).split("/hasil/")[1].split('"')[0].split("?")[0]


def test_send_redirects_to_result_page_with_detection(mhs, face_ok, fake_model):
    _fill_q(mhs, "3")
    r = _send(mhs)
    assert r.status_code == 302 and "/hasil/" in r.headers["Location"] and "baru=1" in r.headers["Location"]
    page = mhs.get(r.headers["Location"]).get_data(as_text=True)
    assert "Hasil Deteksi" in page and "Sedang" in page and "81.2%" in page and "Label dari kuesioner" in page
    assert "data:image/jpeg;base64" in page and "berbeda" in page            # prediksi sedang vs label berat
    assert "Foto terkirim" in page


def test_result_page_without_model_shows_face_box_only(mhs, face_ok, monkeypatch):
    class D:
        def detect(self, img):
            return [(0, 0, 10, 10)]

        def extract_roi(self, img):
            return img, (5, 5, 30, 30)
    monkeypatch.setattr("webapp.app.FaceDetector", lambda *a, **k: D())
    _fill_q(mhs)
    loc = _send(mhs).headers["Location"]
    page = mhs.get(loc).get_data(as_text=True)
    assert "Model belum dilatih" in page and "data:image/jpeg;base64" in page


def test_result_page_is_private_to_owner_and_admin(app, mhs, admin, face_ok, fake_model):
    _fill_q(mhs)
    loc = _send(mhs).headers["Location"]
    assert admin.get(loc).status_code == 200
    other = app.test_client(); register(other, nim="999000111", name="Siti")
    assert other.get(loc).status_code == 404
    assert app.test_client().get(loc).status_code == 302
    assert mhs.get("/hasil/ngawur").status_code == 404 and mhs.get("/hasil/0123456789abcdef").status_code == 404


def test_student_has_no_delete_controls_on_result_or_list(mhs, face_ok, fake_model):
    _fill_q(mhs)
    loc = _send(mhs).headers["Location"]
    for url in (loc, "/kirim", "/mahasiswa", "/riwayat"):
        assert "Hapus" not in mhs.get(url).get_data(as_text=True), url


def test_admin_photo_gallery_filters_thumb_and_delete(app, mhs, admin, face_ok):
    _fill_q(mhs, "3"); _send(mhs); _send(mhs)
    other = app.test_client(); register(other, nim="999000111", name="Siti"); _fill_q(other, "0"); _send(other)
    page = admin.get("/foto-mahasiswa").get_data(as_text=True)
    assert "3 foto" in page and "Budi" in page and "Siti" in page
    assert "2 foto" in admin.get("/foto-mahasiswa?label=berat").get_data(as_text=True)
    assert "1 foto" in admin.get("/foto-mahasiswa?label=normal").get_data(as_text=True)
    sid = _sid_of(admin)
    t = admin.get(f"/kiriman/{sid}/thumb.jpg")
    assert t.status_code == 200 and t.mimetype == "image/jpeg"
    assert mhs.get(f"/kiriman/{sid}/thumb.jpg").status_code == 403 and mhs.get("/foto-mahasiswa").status_code == 403
    r = admin.post("/data-mahasiswa/hapus-kiriman", data={"id": sid, "kembali": "foto"})
    assert r.status_code == 302 and r.headers["Location"].endswith("/foto-mahasiswa")
    assert "2 foto" in admin.get("/foto-mahasiswa").get_data(as_text=True)


def test_admin_gallery_import_returns_to_gallery(admin, mhs, face_ok, ws):
    _fill_q(mhs, "3"); _send(mhs)
    r = admin.post("/data-mahasiswa/impor", data={"id": _sid_of(admin), "split": "test", "label": "berat", "kembali": "foto"})
    assert r.headers["Location"].endswith("/foto-mahasiswa")
    assert len(list((ws.test_dir / "berat").glob("mhs_*.jpg"))) == 1
    assert "Masuk" in admin.get("/foto-mahasiswa?status=dipakai").get_data(as_text=True)


@pytest.mark.parametrize("apa,riwayat_left,foto_left,akun_left", [
    ("riwayat", False, True, True), ("foto", True, False, True), ("semua", False, False, True), ("akun", False, False, False)])
def test_admin_delete_student_data(app, mhs, admin, ws, face_ok, apa, riwayat_left, foto_left, akun_left):
    _fill_q(mhs); _send(mhs)
    r = admin.post("/data-mahasiswa/hapus-data", data={"nim": "251321036", "apa": apa})
    assert r.status_code == 302
    store = UserStore(ws.users_file)
    assert (len(list(ws.history_dir.glob("251321036.json"))) == 1) == riwayat_left
    assert (len(list(ws.submissions_dir.glob("*.jpg"))) == 1) == foto_left
    assert (store.get("251321036") is not None) == akun_left


def test_delete_student_data_admin_only_and_validated(app, mhs, admin):
    assert mhs.post("/data-mahasiswa/hapus-data", data={"nim": "251321036", "apa": "akun"}).status_code == 403
    assert admin.post("/data-mahasiswa/hapus-data", data={"nim": "251321036", "apa": "ngawur"}).status_code == 404
    assert admin.post("/data-mahasiswa/hapus-data", data={"nim": "peneliti", "apa": "akun"}).status_code == 404   # admin tak bisa dihapus lewat sini


def test_branding_is_academicare(admin):
    page = admin.get("/dashboard").get_data(as_text=True)
    assert "AcademiCare" in page and "StressFER" not in page
