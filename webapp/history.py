"""Riwayat hasil per mahasiswa. Hanya hasil (label, skor/keyakinan, waktu) yang disimpan, TIDAK ada citra wajah."""
import json
import os
import threading
import time
from pathlib import Path
from typing import List, Optional

MAX_ENTRIES = 200


class HistoryStore:
    def __init__(self, directory):
        self.dir = Path(directory)
        self._lock = threading.Lock()

    def _file(self, username: str) -> Path:
        # username sudah divalidasi (^[A-Za-z0-9_.-]{3,32}$); tolak nama berbahaya sekali lagi di sini.
        if not username or "/" in username or "\\" in username or username.startswith("."):
            raise ValueError("username tidak valid")
        return self.dir / f"{username}.json"

    def _read(self, username: str) -> List[dict]:
        f = self._file(username)
        if not f.exists():
            return []
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []

    def add(self, username: str, jenis: str, label: str, detail: str) -> dict:
        entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "jenis": jenis, "label": label, "detail": detail}
        with self._lock:
            items = self._read(username)
            items.append(entry)
            items = items[-MAX_ENTRIES:]
            self.dir.mkdir(parents=True, exist_ok=True)
            f = self._file(username)
            tmp = f.with_suffix(".tmp")
            tmp.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
            try:
                os.chmod(tmp, 0o600)
            except OSError:
                pass
            os.replace(tmp, f)
        return entry

    def list(self, username: str) -> List[dict]:
        """Terbaru lebih dulu."""
        with self._lock:
            return list(reversed(self._read(username)))

    def last(self, username: str, jenis: Optional[str] = None) -> Optional[dict]:
        for e in self.list(username):
            if jenis is None or e["jenis"] == jenis:
                return e
        return None

    def clear(self, username: str) -> None:
        with self._lock:
            f = self._file(username)
            if f.exists():
                f.unlink()
