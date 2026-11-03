import base64
import io
import time
import zipfile

import cv2
import numpy as np
import pytest

from stressfer import config, questionnaire, uploads
from stressfer.dataset import build_all, count_images, load_dataset, load_test_if_any
from stressfer.demo_data import make_demo_dataset
from stressfer.evaluation import evaluate
from stressfer.face_detection import FaceDetector
from stressfer.features import flatten_features
from stressfer.model import load_model, train_model
from stressfer.preprocessing import normalize_face, to_grayscale

SMALL_GRID = {"svm__C": [1, 10], "svm__gamma": ["scale", 1e-4]}


def png_bytes(seed=0, size=80, fmt=".png"):
    img = np.random.default_rng(seed).integers(0, 255, (size, size, 3), dtype=np.uint8)
    return cv2.imencode(fmt, img)[1].tobytes()


def make_zip(entries):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in entries.items():
            z.writestr(name, data)
    return buf.getvalue()


@pytest.fixture
def ws(tmp_path):
    return config.Workspace(tmp_path)


# ---------------------------------------------------------------- dasar
def test_grayscale_and_normalize():
    bgr = np.random.default_rng(0).integers(0, 255, (150, 130, 3), dtype=np.uint8)
    g = to_grayscale(bgr)
    assert g.shape == (150, 130)
    face = normalize_face(g)
    assert face.shape == config.IMAGE_SIZE and face.dtype == np.uint8
    f = flatten_features(face)
    assert f.shape == (100 * 100,) and 0 <= f.min() and f.max() <= 1


@pytest.mark.parametrize("total,label", [
    (0, "normal"), (14, "normal"), (15, "ringan"), (18, "ringan"),
    (19, "sedang"), (25, "sedang"), (26, "berat"), (42, "berat")])
def test_questionnaire_thresholds(total, label):
    assert questionnaire.classify(total) == label


def test_questionnaire_validation():
    assert questionnaire.score([1] * 14) == 14
    with pytest.raises(ValueError):
        questionnaire.score([1] * 13)
    with pytest.raises(ValueError):
        questionnaire.score([4] + [0] * 13)
    assert questionnaire.valid_subject_id("R001") and not questionnaire.valid_subject_id("../x")


def test_detector_loads_and_handles_blank():
    det = FaceDetector()
    assert det.detect(np.zeros((200, 200, 3), np.uint8)) == []
    assert det.extract_roi(np.zeros((200, 200, 3), np.uint8)) == (None, None)


# ---------------------------------------------------------------- unggah
def test_store_image_validates(ws):
    name = uploads.store_image(png_bytes(), ws, "train", "normal")
    assert name.endswith(".png") and (ws.train_dir / "normal" / name).exists()
    jpg = uploads.store_image(png_bytes(fmt=".jpg"), ws, "test", "berat")
    assert jpg.endswith(".jpg") and (ws.test_dir / "berat" / jpg).exists()
    for bad, why in [(b"", "kosong"), (b"GIF89a....", "bukan JPG/PNG"),
                     (b"\x89PNG\r\n\x1a\nxxxx", "rusak"), (png_bytes(size=10), "ukuran")]:
        with pytest.raises(ValueError, match=why):
            uploads.store_image(bad, ws, "train", "normal")
    with pytest.raises(ValueError):
        uploads.store_image(png_bytes(), ws, "train", "../etc")
    with pytest.raises(ValueError):
        uploads.store_image(png_bytes(), ws, "val", "normal")


def test_upload_ignores_user_filename(ws):
    r = uploads.save_images([("../../evil.png", png_bytes(1)), ("x.txt", b"hello")], ws, "train", "ringan")
    assert r["saved"] == 1 and r["rejected"][0]["name"] == "x.txt"
    files = list((ws.train_dir / "ringan").iterdir())
    assert len(files) == 1 and "evil" not in files[0].name
    assert not (ws.root / "evil.png").exists()


def test_zip_routing(ws):
    z = make_zip({
        "train/normal/a.png": png_bytes(1), "train/berat/b.png": png_bytes(2),
        "uji/ringan/c.png": png_bytes(3), "Test/Sedang/d.jpg": png_bytes(4, fmt=".jpg"),
        "train/unknown/e.png": png_bytes(5), "__MACOSX/train/normal/._a.png": b"x",
        "train/normal/readme.txt": b"hi", "train/normal/broken.png": b"not an image",
    })
    r = uploads.save_zip(z, ws)
    assert r["saved"] == 4
    assert count_images(ws.train_dir)["normal"] == 1 and count_images(ws.train_dir)["berat"] == 1
    assert count_images(ws.test_dir)["ringan"] == 1 and count_images(ws.test_dir)["sedang"] == 1
    reasons = {x["name"]: x["reason"] for x in r["rejected"]}
    assert "e.png" in reasons and "broken.png" in reasons and "readme.txt" not in reasons


def test_zip_forced_split_and_bad_zip(ws):
    z = make_zip({"normal/a.png": png_bytes(1), "test/ringan/b.png": png_bytes(2)})
    r = uploads.save_zip(z, ws, split="test")
    assert r["saved"] == 2 and sum(count_images(ws.train_dir).values()) == 0
    with pytest.raises(ValueError):
        uploads.save_zip(b"not a zip", ws)


def test_gallery_and_delete(ws):
    names = [uploads.store_image(png_bytes(i), ws, "train", "normal") for i in range(5)]
    page, total = uploads.list_images_page(ws, "train", "normal", 0, 3)
    assert total == 5 and len(page) == 3
    assert uploads.delete_image(ws, "train", "normal", names[0])
    assert not uploads.delete_image(ws, "train", "normal", "../../etc/passwd")
    assert not uploads.delete_image(ws, "train", "normal", names[0])
    assert uploads.clear_images(ws, "train", "normal") == 4


# ---------------------------------------------------------------- pelatihan
def _fill(ws, split, per_class, seed):
    src = make_demo_dataset(per_class=per_class, out_dir=ws.root / f"_demo_{split}", seed=seed)
    for label in config.CLASSES:
        for p in (src / label).iterdir():
            uploads.store_image(p.read_bytes(), ws, split, label, prefix="d")


def test_train_with_split_80_20(ws):
    _fill(ws, "train", 30, 0)
    build_all(ws, already_cropped=True)
    assert load_test_if_any(ws) == (None, None)
    X, y = load_dataset(ws.dataset_file)
    b = train_model(X, y, ws.model_path, param_grid=SMALL_GRID)
    assert b["test_source"] == "split" and len(b["train_idx"]) == 96 and len(b["test_idx"]) == 24
    m = evaluate(b, X, y, ws.reports)
    assert m["n_test"] == 24 and np.array(m["confusion_matrix"]).sum() == 24
    assert (ws.reports / "confusion_matrix.png").exists()
    assert load_model(ws.model_path)["best_params"] == b["best_params"]


def test_train_with_uploaded_test_set(ws):
    _fill(ws, "train", 25, 0)
    _fill(ws, "test", 8, 99)
    info = build_all(ws, already_cropped=True)
    assert info["train"]["total"] == 100 and info["test"]["total"] == 32
    X, y = load_dataset(ws.dataset_file)
    Xt, yt = load_test_if_any(ws)
    b = train_model(X, y, ws.model_path, param_grid=SMALL_GRID, has_separate_test=True)
    assert b["test_source"] == "upload" and len(b["train_idx"]) == 100
    m = evaluate(b, X, y, ws.reports, Xt, yt)
    assert m["n_test"] == 32 and m["test_source"] == "upload"
    # data uji dihapus -> dataset uji lama ikut dibuang pada pre-processing berikutnya
    uploads.clear_images(ws, "test")
    build_all(ws, already_cropped=True)
    assert not ws.dataset_test_file.exists()


def test_too_few_samples_rejected():
    with pytest.raises(ValueError):
        train_model(np.zeros((3, 10), np.float32), np.array([0, 1, 1]))


def test_no_faces_gives_clear_error(ws):
    uploads.store_image(png_bytes(), ws, "train", "normal")
    uploads.store_image(png_bytes(1), ws, "train", "ringan")
    with pytest.raises(RuntimeError, match="Tidak ada wajah"):
        build_all(ws, already_cropped=False)


# ---------------------------------------------------------------- web
@pytest.fixture
def client(ws):
    """Klien web yang sudah login (seluruh halaman aplikasi wajib login)."""
    from webapp.app import create_app
    from webapp.auth import UserStore
    UserStore(ws.users_file).create("peneliti", "rahasia-123", "Peneliti Uji")
    c = create_app(workspace=ws).test_client()
    assert c.post("/login", data={"username": "peneliti", "password": "rahasia-123"}).status_code == 302
    return c


def post_files(client, files, **form):
    data = {"files": [(io.BytesIO(b), n) for n, b in files], **form}
    return client.post("/api/upload", data=data, content_type="multipart/form-data")


def test_pages_render(client):
    for url in ["/", "/dashboard", "/data", "/train", "/predict", "/questionnaire", "/capture", "/evaluation",
                "/pengumuman", "/faq", "/kontak"]:
        assert client.get(url).status_code == 200, url


def test_web_upload_images_and_zip(client, ws):
    r = post_files(client, [("a.png", png_bytes(1)), ("b.jpg", png_bytes(2, fmt=".jpg")), ("c.txt", b"x")],
                   split="train", label="sedang")
    d = r.get_json()
    assert r.status_code == 200 and d["saved"] == 2 and d["rejected_total"] == 1
    assert d["counts"]["train"]["sedang"] == 2
    z = make_zip({"train/normal/a.png": png_bytes(3), "test/berat/b.png": png_bytes(4)})
    d = post_files(client, [("set.zip", z)], split="auto").get_json()
    assert d["saved"] == 2 and d["counts"]["test"]["berat"] == 1 and d["counts"]["train"]["normal"] == 1
    # validasi
    assert post_files(client, [("a.png", png_bytes())], split="train").status_code == 400            # tanpa kelas
    assert post_files(client, [("a.png", png_bytes())], split="auto", label="normal").status_code == 400
    assert post_files(client, [("a.png", png_bytes())], split="val", label="normal").status_code == 400
    assert client.post("/api/upload", data={"split": "train"}).status_code == 400                    # tanpa berkas


def test_web_gallery_thumb_delete(client, ws):
    post_files(client, [("a.png", png_bytes(1, size=400))], split="test", label="ringan")
    d = client.get("/api/dataset/list?split=test&label=ringan").get_json()
    assert d["total"] == 1
    name = d["names"][0]
    t = client.get(f"/thumb/test/ringan/{name}")
    assert t.status_code == 200 and t.mimetype == "image/jpeg"
    img = cv2.imdecode(np.frombuffer(t.data, np.uint8), cv2.IMREAD_COLOR)
    assert max(img.shape[:2]) <= 220
    assert client.get("/thumb/test/ringan/..%2F..%2Fmain.py").status_code == 404
    assert client.get("/thumb/val/ringan/x.png").status_code == 404
    assert client.get("/api/dataset/list?split=zzz&label=ringan").status_code == 400
    assert client.post("/api/dataset/delete", json={"split": "test", "label": "ringan", "name": name}).status_code == 200
    assert client.post("/api/dataset/delete", json={"split": "test", "label": "ringan", "name": name}).status_code == 404
    post_files(client, [("a.png", png_bytes(1))], split="train", label="berat")
    r = client.post("/api/dataset/clear", json={"split": "train", "label": "berat"})
    assert r.get_json()["removed"] == 1
    assert client.post("/api/dataset/clear", json={"split": "x"}).status_code == 400


def test_web_train_end_to_end(client, ws, monkeypatch):
    import stressfer.config as cfg
    monkeypatch.setattr(cfg, "PARAM_GRID", SMALL_GRID)
    # tanpa data -> gagal dengan pesan jelas
    assert client.post("/api/train", json={"already_cropped": True}).status_code == 200
    for _ in range(100):
        s = client.get("/api/train/status").get_json()
        if s["status"] != "running":
            break
        time.sleep(0.1)
    assert s["status"] == "error" and "Tidak ada citra" in s["error"]

    _fill(ws, "train", 25, 0)
    _fill(ws, "test", 8, 99)
    assert client.post("/api/train", json={"already_cropped": True}).status_code == 200
    for _ in range(300):
        s = client.get("/api/train/status").get_json()
        if s["status"] != "running":
            break
        time.sleep(0.2)
    assert s["status"] == "done", s
    assert "metrics" not in s and s["progress"] == 100
    m = client.get("/reports/metrics.json").get_json()
    assert m["n_test"] == 32 and m["test_source"] == "upload"
    for url in ["/evaluation", "/dashboard", "/train", "/predict"]:
        assert client.get(url).status_code == 200
    assert client.get("/reports/confusion_matrix.png").status_code == 200
    assert client.get("/reports/../main.py").status_code == 404


def test_web_predict_and_capture_validation(client):
    ok, buf = cv2.imencode(".png", np.full((200, 200, 3), 90, np.uint8))
    uri = "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()
    assert client.post("/api/predict", json={"image": uri}).status_code == 503     # model belum ada
    assert client.post("/api/predict", json={"image": "bukan-citra"}).status_code == 400
    base = {"image": uri, "subject_id": "R001", "label": "normal", "split": "train", "consent": True}
    assert client.post("/api/capture", json={**base, "consent": False}).status_code == 400
    assert client.post("/api/capture", json={**base, "subject_id": "../../etc"}).status_code == 400
    assert client.post("/api/capture", json={**base, "label": "x"}).status_code == 400
    assert client.post("/api/capture", json={**base, "split": "val"}).status_code == 400
    assert client.post("/api/capture", json=base).status_code == 422                # tak ada wajah
