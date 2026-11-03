"""Kiriman foto dari mahasiswa untuk admin. Citra disimpan ulang sebagai JPEG (metadata EXIF hilang), bernama acak."""
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional

MAX_PER_STUDENT = 20
_ID = re.compile(r"^[0-9a-f]{16}$")


class SubmissionStore:
    def __init__(self, directory):
        self.dir = Path(directory)
        self.index = self.dir / "index.json"
        self._lock = threading.Lock()

    def _read(self) -> List[dict]:
        if not self.index.exists():
            return []
        try:
            return json.loads(self.index.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []

    def _write(self, items: List[dict]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.index.with_suffix(".tmp")
        tmp.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, self.index)

    def add(self, username: str, label: str, jpeg: bytes) -> dict:
        with self._lock:
            items = self._read()
            if sum(1 for i in items if i["username"] == username) >= MAX_PER_STUDENT:
                raise ValueError(f"Batas {MAX_PER_STUDENT} kiriman per mahasiswa tercapai.")
            sid = uuid.uuid4().hex[:16]
            self.dir.mkdir(parents=True, exist_ok=True)
            (self.dir / f"{sid}.jpg").write_bytes(jpeg)
            entry = {"id": sid, "username": username, "label": label, "status": "baru",
                     "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
            items.append(entry)
            self._write(items)
        return entry

    def list(self, username: Optional[str] = None) -> List[dict]:
        with self._lock:
            items = self._read()
        items = [i for i in items if username is None or i["username"] == username]
        return list(reversed(items))

    def get(self, sid: str) -> Optional[dict]:
        if not _ID.match(sid or ""):
            return None
        with self._lock:
            return next((i for i in self._read() if i["id"] == sid), None)

    def path(self, sid: str) -> Optional[Path]:
        if self.get(sid) is None:
            return None
        p = self.dir / f"{sid}.jpg"
        return p if p.exists() else None

    def mark_used(self, sid: str, split: str, label: str) -> None:
        with self._lock:
            items = self._read()
            for i in items:
                if i["id"] == sid:
                    i.update(status="dipakai", split=split, used_label=label)
            self._write(items)

    def delete(self, sid: str) -> bool:
        if not _ID.match(sid or ""):
            return False
        with self._lock:
            items = self._read()
            keep = [i for i in items if i["id"] != sid]
            if len(keep) == len(items):
                return False
            self._write(keep)
            (self.dir / f"{sid}.jpg").unlink(missing_ok=True)
        return True

    def delete_user(self, username: str) -> int:
        n = 0
        for i in self.list(username):
            n += self.delete(i["id"])
        return n
