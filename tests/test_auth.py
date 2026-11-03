import json

import pytest

from stressfer import config
from webapp.app import create_app
from webapp.auth import LoginThrottle, UserStore, load_secret, safe_next, validate_password, validate_username

PW = "rahasia-123"


@pytest.fixture
def ws(tmp_path):
    return config.Workspace(tmp_path)


@pytest.fixture
def app(ws):
    return create_app(workspace=ws)


@pytest.fixture
def anon(app):
    return app.test_client()


@pytest.fixture
def store(ws):
    return UserStore(ws.users_file)


def login(c, user="peneliti", pw=PW, **extra):
    return c.post("/login", data={"username": user, "password": pw, **extra})


# ---------------------------------------------------------------- UserStore
def test_store_hashes_and_validates(store, ws):
    assert not store.has_users()
    store.create("Peneliti", PW, "Nama Panjang")
    assert store.has_users() and store.get("PENELITI") == {"username": "peneliti", "name": "Nama Panjang", "role": "admin"}
    raw = ws.users_file.read_text()
    assert PW not in raw and ("pbkdf2" in raw or "scrypt" in raw)
    assert store.verify("peneliti", PW) and store.verify(" PENELITI ", PW)
    assert not store.verify("peneliti", "salah") and not store.verify("tidakada", PW) and not store.verify("", "")
    with pytest.raises(ValueError, match="sudah dipakai"):
        store.create("peneliti", PW)
    for bad in ["ab", "a b c", "../x", "x" * 33, "", "né"]:
        with pytest.raises(ValueError):
            validate_username(bad)
    for bad in ["pendek", "x" * 129]:
        with pytest.raises(ValueError):
            validate_password(bad)
    with pytest.raises(ValueError):
        validate_password("peneliti01", "peneliti01")
    store.set_password("peneliti", "password-baru-9")
    assert store.verify("peneliti", "password-baru-9") and not store.verify("peneliti", PW)
    with pytest.raises(ValueError):
        store.set_password("tidakada", PW)


def test_store_corrupt_file_is_reported(store, ws):
    ws.users_file.parent.mkdir(parents=True, exist_ok=True)
    ws.users_file.write_text("{bukan json")
    with pytest.raises(RuntimeError, match="rusak"):
        store.has_users()


def test_secret_persisted_and_env(tmp_path, monkeypatch):
    monkeypatch.delenv("STRESSFER_SECRET", raising=False)
    a = load_secret(tmp_path / "s")
    assert len(a) >= 32 and load_secret(tmp_path / "s") == a
    monkeypatch.setenv("STRESSFER_SECRET", "dari-env")
    assert load_secret(tmp_path / "x") == "dari-env"


def test_safe_next():
    assert safe_next("/data") == "/data" and safe_next("/predict?a=1") == "/predict?a=1"
    for bad in [None, "", "//evil.com", "https://evil.com", "evil.com", "/\\evil.com", "/login", "/logout", "/setup?x=1"]:
        assert safe_next(bad) is None, bad


# ---------------------------------------------------------------- throttle
def test_throttle_window_and_reset():
    t = [0.0]
    th = LoginThrottle(max_user=3, max_ip=5, window=100, clock=lambda: t[0])
    for _ in range(3):
        assert th.wait_seconds("1.1.1.1", "a") == 0
        th.fail("1.1.1.1", "a")
    assert th.wait_seconds("1.1.1.1", "a") > 0
    assert th.wait_seconds("1.1.1.1", "b") == 0          # username lain belum terkunci
    assert th.wait_seconds("2.2.2.2", "a") == 0          # IP lain tidak terpengaruh
    t[0] = 101
    assert th.wait_seconds("1.1.1.1", "a") == 0          # jendela habis
    for u in "abcde":                                    # penyemprotan username dari satu IP
        th.fail("3.3.3.3", u)
    assert th.wait_seconds("3.3.3.3", "baru") > 0
    th2 = LoginThrottle(max_user=3, clock=lambda: t[0])
    th2.fail("1.1.1.1", "a"); th2.reset("1.1.1.1", "a")
    assert th2.wait_seconds("1.1.1.1", "a") == 0


# ---------------------------------------------------------------- akses anonim
PROTECTED = ["/dashboard", "/data", "/train", "/evaluation", "/predict", "/questionnaire", "/capture",
             "/reports/metrics.json", "/thumb/train/normal/x.jpg", "/api/dataset", "/api/dataset/list?split=train&label=normal"]


def test_anonymous_is_redirected_or_401(anon, store):
    store.create("peneliti", PW)
    for url in PROTECTED:
        r = anon.get(url)
        if url.startswith("/api/"):
            assert r.status_code == 401 and "login" in r.get_json()["error"].lower(), url
        else:
            assert r.status_code == 302 and "/login" in r.headers["Location"], url
    assert "next=%2Fdata" in anon.get("/data").headers["Location"] or "next=/data" in anon.get("/data").headers["Location"]
    for url, body in [("/api/upload", {}), ("/api/train", {}), ("/api/predict", {}), ("/api/capture", {}),
                      ("/api/dataset/clear", {}), ("/api/dataset/delete", {})]:
        assert anon.post(url, json=body).status_code == 401, url
    assert anon.post("/predict").status_code == 302 and anon.post("/questionnaire").status_code == 302


def test_public_pages_open(anon, store):
    store.create("peneliti", PW)
    for url in ["/", "/pengumuman", "/faq", "/kontak", "/login", "/static/css/style.css"]:
        assert anon.get(url).status_code == 200, url
    home = anon.get("/").get_data(as_text=True)
    assert "Masuk ke Akun Anda" in home and "Masuk Sekarang" in home
    assert anon.get("/tidak-ada").status_code == 404


# ---------------------------------------------------------------- setup akun pertama
def test_setup_flow(anon, store):
    assert "Buat Akun Pertama" in anon.get("/").get_data(as_text=True)
    assert anon.get("/login").status_code == 302 and "/setup" in anon.get("/login").headers["Location"]
    assert anon.get("/setup").status_code == 200
    for form, msg in [
        ({"username": "admin1", "password": "pendek", "password2": "pendek"}, "minimal 8"),
        ({"username": "admin1", "password": PW, "password2": "beda-beda-1"}, "tidak sama"),
        ({"username": "a b", "password": PW, "password2": PW}, "Username"),
        ({"username": "admin1", "password": "admin1", "password2": "admin1"}, "minimal 8"),
    ]:
        r = anon.post("/setup", data=form)
        assert r.status_code == 400 and msg in r.get_data(as_text=True), form
        assert not store.has_users()
    r = anon.post("/setup", data={"username": "Admin1", "name": "Admin Satu", "password": PW, "password2": PW})
    assert r.status_code == 302 and r.headers["Location"].endswith("/dashboard")
    assert anon.get("/dashboard").status_code == 200                       # langsung login
    assert store.verify("admin1", PW)
    assert anon.get("/setup").status_code == 302                           # tidak bisa dipakai lagi
    other = anon.application.test_client()
    assert other.post("/setup", data={"username": "penyusup", "password": PW, "password2": PW}).status_code == 302
    assert store.get("penyusup") is None


def test_setup_blocked_from_remote_host(app, store):
    c = app.test_client()
    kw = {"environ_overrides": {"REMOTE_ADDR": "203.0.113.9"}}
    assert c.get("/setup", **kw).status_code == 403
    assert c.post("/setup", data={"username": "x1y2z3", "password": PW, "password2": PW}, **kw).status_code == 403
    assert not store.has_users()
    assert "create-user" in c.get("/", **kw).get_data(as_text=True)


# ---------------------------------------------------------------- login / logout
def test_login_logout_cycle(anon, store):
    store.create("peneliti", PW, "Peneliti Uji")
    r = login(anon)
    assert r.status_code == 302 and r.headers["Location"].endswith("/dashboard")
    cookie = r.headers["Set-Cookie"]
    assert "HttpOnly" in cookie and "SameSite=Lax" in cookie
    page = anon.get("/dashboard")
    assert page.status_code == 200 and "Peneliti Uji" in page.get_data(as_text=True) and "Keluar" in page.get_data(as_text=True)
    assert anon.get("/api/dataset").status_code == 200
    assert anon.get("/login").status_code == 302                          # sudah login
    assert anon.get("/logout").status_code == 405                         # logout hanya POST
    assert anon.post("/logout").status_code == 302
    assert anon.get("/dashboard").status_code == 302 and anon.get("/api/dataset").status_code == 401


def test_login_errors_are_generic_and_rejected(anon, store):
    store.create("peneliti", PW)
    a = login(anon, "peneliti", "salah-salah")
    b = login(anon, "tidakada", "salah-salah")
    assert a.status_code == b.status_code == 401
    ta, tb = a.get_data(as_text=True), b.get_data(as_text=True)
    assert "Username atau password salah." in ta and "Username atau password salah." in tb
    assert anon.get("/dashboard").status_code == 302                      # tetap anonim
    assert login(anon, "", "").status_code == 401
    assert login(anon, "peneliti", "x" * 100000).status_code in (401, 413)


def test_login_next_and_open_redirect(anon, store):
    store.create("peneliti", PW)
    r = login(anon, next="/data")
    assert r.status_code == 302 and r.headers["Location"].endswith("/data")
    anon.post("/logout")
    for evil in ["//evil.com", "https://evil.com/x", "/\\evil.com", "evil.com"]:
        r = login(anon, next=evil)
        assert r.status_code == 302 and r.headers["Location"].endswith("/dashboard"), evil
        anon.post("/logout")
    # alur nyata: ditolak -> login -> kembali ke halaman tujuan
    loc = anon.get("/train").headers["Location"]
    r = anon.post(loc, data={"username": "peneliti", "password": PW})
    assert r.status_code == 302 and r.headers["Location"].endswith("/train")


def test_login_lockout(anon, store):
    store.create("peneliti", PW)
    for _ in range(5):
        assert login(anon, pw="salah-salah").status_code == 401
    r = login(anon)                                                       # password benar pun ditolak saat terkunci
    assert r.status_code == 429 and "Terlalu banyak" in r.get_data(as_text=True)
    assert anon.get("/dashboard").status_code == 302
    other_ip = {"environ_overrides": {"REMOTE_ADDR": "198.51.100.7"}}
    assert anon.post("/login", data={"username": "peneliti", "password": PW}, **other_ip).status_code == 302


def test_session_dies_when_user_removed(app, store, ws):
    store.create("peneliti", PW)
    store.create("lain", PW)
    c = app.test_client()
    login(c, "lain")
    assert c.get("/dashboard").status_code == 200
    data = json.loads(ws.users_file.read_text())
    del data["lain"]
    ws.users_file.write_text(json.dumps(data))
    assert c.get("/dashboard").status_code == 302 and c.get("/api/dataset").status_code == 401


def test_forged_session_cookie_rejected(app, store):
    store.create("peneliti", PW)
    c = app.test_client()
    c.set_cookie("session", "eyJ1aWQiOiJwZW5lbGl0aSJ9.aaaaaa.bbbbbbbbbbbbbbbbbbbbbbbbbbb")
    assert c.get("/dashboard").status_code == 302


def test_security_headers(anon, store):
    store.create("peneliti", PW)
    login(anon)
    r = anon.get("/dashboard")
    assert r.headers["X-Frame-Options"] == "DENY" and r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["Cache-Control"] == "no-store" and r.headers["Referrer-Policy"] == "same-origin"
