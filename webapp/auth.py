"""Autentikasi sederhana untuk aplikasi lokal/kampus: akun di file JSON, password di-hash, pembatasan percobaan login.

Prinsip: tidak ada akun/password bawaan. Akun pertama dibuat saat pertama kali dibuka (/setup dari komputer yang
sama) atau lewat CLI (`python main.py create-user`). Password tidak pernah disimpan polos.
"""
import json
import os
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from werkzeug.security import check_password_hash, generate_password_hash

ROLES = ("admin", "mahasiswa")
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
MIN_PASSWORD, MAX_PASSWORD = 8, 128
# Hash palsu agar waktu verifikasi sama untuk username yang tidak ada (cegah enumerasi lewat timing).
_DUMMY_HASH = generate_password_hash("tidak-dipakai-" + secrets.token_hex(8))


def validate_username(username: str) -> str:
    username = (username or "").strip().lower()
    if not USERNAME_RE.match(username):
        raise ValueError("Username 3-32 karakter: huruf, angka, titik, garis bawah, atau strip.")
    return username


def validate_password(password: str, username: str = "") -> None:
    if len(password or "") < MIN_PASSWORD:
        raise ValueError(f"Password minimal {MIN_PASSWORD} karakter.")
    if len(password) > MAX_PASSWORD:
        raise ValueError(f"Password maksimal {MAX_PASSWORD} karakter.")
    if username and password.lower() == username.lower():
        raise ValueError("Password tidak boleh sama dengan username.")


class UserStore:
    """Penyimpanan akun: {username: {name, hash, created}} dalam satu berkas JSON (ditulis atomik)."""

    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def _read(self) -> Dict[str, dict]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise RuntimeError(f"Berkas akun rusak: {self.path}. Perbaiki atau hapus berkas ini.")

    def _write(self, data: Dict[str, dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, self.path)

    def has_users(self) -> bool:
        with self._lock:
            return bool(self._read())

    def get(self, username: str) -> Optional[dict]:
        """Data publik akun (tanpa hash), atau None."""
        with self._lock:
            u = self._read().get((username or "").strip().lower())
        if not u:
            return None
        # Akun lama tanpa field role dianggap admin (akun peneliti dari versi sebelumnya).
        return {"username": (username or "").strip().lower(), "name": u["name"], "role": u.get("role", "admin")}

    def list(self) -> List[dict]:
        with self._lock:
            return [{"username": k, "name": v["name"], "role": v.get("role", "admin"), "created": v.get("created")}
                    for k, v in self._read().items()]

    def create(self, username: str, password: str, name: str = "", role: str = "admin") -> dict:
        if role not in ROLES:
            raise ValueError("Peran tidak valid.")
        username = validate_username(username)
        validate_password(password, username)
        name = (name or username).strip()[:60]
        with self._lock:
            data = self._read()
            if username in data:
                raise ValueError("Username sudah dipakai.")
            data[username] = {"name": name, "role": role, "hash": generate_password_hash(password),
                              "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
            self._write(data)
        return {"username": username, "name": name, "role": role}

    def delete(self, username: str) -> bool:
        username = (username or "").strip().lower()
        with self._lock:
            data = self._read()
            if username not in data:
                return False
            if data[username].get("role", "admin") == "admin" and \
                    sum(1 for v in data.values() if v.get("role", "admin") == "admin") <= 1:
                raise ValueError("Admin terakhir tidak dapat dihapus.")
            del data[username]
            self._write(data)
        return True

    def set_password(self, username: str, password: str) -> None:
        username = validate_username(username)
        validate_password(password, username)
        with self._lock:
            data = self._read()
            if username not in data:
                raise ValueError("Username tidak ditemukan.")
            data[username]["hash"] = generate_password_hash(password)
            self._write(data)

    def verify(self, username: str, password: str) -> bool:
        username = (username or "").strip().lower()
        with self._lock:
            u = self._read().get(username)
        ok = check_password_hash(u["hash"] if u else _DUMMY_HASH, password or "")
        return bool(u) and ok


class LoginThrottle:
    """Batasi percobaan login gagal: MAX_USER per (IP, username) dan MAX_IP per IP dalam WINDOW detik."""

    def __init__(self, max_user: int = 5, max_ip: int = 20, window: int = 300, clock=time.monotonic):
        self.max_user, self.max_ip, self.window, self.clock = max_user, max_ip, window, clock
        self._fails: Dict[tuple, List[float]] = {}
        self._lock = threading.Lock()

    def _recent(self, key) -> List[float]:
        now = self.clock()
        fails = [t for t in self._fails.get(key, []) if now - t < self.window]
        if fails:
            self._fails[key] = fails
        else:
            self._fails.pop(key, None)
        return fails

    def wait_seconds(self, ip: str, username: str) -> int:
        """0 bila boleh mencoba; selain itu sisa detik penguncian."""
        with self._lock:
            now, wait = self.clock(), 0
            for key, limit in (((ip, username.lower()[:32]), self.max_user), ((ip, None), self.max_ip)):
                fails = self._recent(key)
                if len(fails) >= limit:
                    wait = max(wait, int(self.window - (now - fails[0])) + 1)
            return wait

    def fail(self, ip: str, username: str) -> None:
        with self._lock:
            now = self.clock()
            for key in ((ip, username.lower()[:32]), (ip, None)):
                self._fails.setdefault(key, []).append(now)

    def reset(self, ip: str, username: str) -> None:
        with self._lock:
            self._fails.pop((ip, username.lower()[:32]), None)


def load_secret(path) -> str:
    """Kunci sesi: dari env STRESSFER_SECRET, atau dibuat acak dan disimpan (izin 600) di path."""
    env = os.environ.get("STRESSFER_SECRET")
    if env:
        return env
    path = Path(path)
    if path.exists():
        s = path.read_text(encoding="utf-8").strip()
        if len(s) >= 32:
            return s
    path.parent.mkdir(parents=True, exist_ok=True)
    s = secrets.token_hex(32)
    path.write_text(s, encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return s


def safe_next(target: Optional[str]) -> Optional[str]:
    """Hanya izinkan path relatif di situs ini (cegah open redirect)."""
    if not target or not target.startswith("/") or target.startswith("//") or "\\" in target:
        return None
    if re.match(r"^/+(login|setup|logout)\b", target):
        return None
    return target
