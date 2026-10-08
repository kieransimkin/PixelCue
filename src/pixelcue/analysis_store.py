from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from platformdirs import user_data_dir

from .logging_utils import log_event, log_exception


SCHEMA_VERSION = 1


def default_analysis_db_path() -> Path:
    override = os.environ.get("PIXELCUE_ANALYSIS_DB", "").strip()
    if override:
        return Path(override).expanduser()
    return Path(user_data_dir("PixelCue", "DanceFlow")) / "analysis-cache.sqlite3"


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _jsonable(item())
        except Exception:
            pass
    return str(value)


def normalized_path(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def file_signature(path: str | Path) -> tuple[str, int, int]:
    p = Path(path)
    st = p.stat()
    return normalized_path(p), int(st.st_size), int(st.st_mtime_ns)


class AnalysisStore:
    """Permanent stage-oriented analysis cache backed by SQLite.

    Results are immutable by file signature: if a path changes size or mtime, a
    new row is written while the old result remains as historical analysis.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or default_analysis_db_path()).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.db_path,
            timeout=30.0,
            check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._create_schema()
        log_event(
            "analysis-store",
            "ready",
            "persistent analysis cache opened",
            db_path=str(self.db_path),
        )

    def _create_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    engine TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(path, size_bytes, mtime_ns, kind, engine)
                );
                CREATE INDEX IF NOT EXISTS idx_analysis_lookup
                    ON analysis_results(path, kind, engine, size_bytes, mtime_ns);
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    def load(
        self,
        path: str | Path,
        *,
        kind: str,
        engine: str,
    ) -> dict[str, Any] | None:
        try:
            normalized, size_bytes, mtime_ns = file_signature(path)
            with self._lock:
                row = self._conn.execute(
                    """
                    SELECT payload_json
                    FROM analysis_results
                    WHERE path=? AND size_bytes=? AND mtime_ns=?
                      AND kind=? AND engine=?
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                    (normalized, size_bytes, mtime_ns, kind, engine),
                ).fetchone()
            if row is None:
                return None
            payload = json.loads(row[0])
            log_event(
                "analysis-store",
                "cache-hit",
                "restored persisted analysis",
                path=normalized,
                kind=kind,
                engine=engine,
            )
            return payload
        except FileNotFoundError:
            return None
        except Exception as exc:
            log_exception(
                "analysis-store",
                "load-failure",
                exc,
                path=str(path),
                kind=kind,
                engine=engine,
            )
            return None

    def save(
        self,
        path: str | Path,
        *,
        kind: str,
        engine: str,
        payload: dict[str, Any],
    ) -> bool:
        try:
            normalized, size_bytes, mtime_ns = file_signature(path)
            encoded = json.dumps(
                _jsonable(payload),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            now = time.time()
            with self._lock, self._conn:
                self._conn.execute(
                    """
                    INSERT INTO analysis_results(
                        path, size_bytes, mtime_ns, kind, engine,
                        payload_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(path, size_bytes, mtime_ns, kind, engine)
                    DO UPDATE SET payload_json=excluded.payload_json,
                                  updated_at=excluded.updated_at
                    """,
                    (
                        normalized,
                        size_bytes,
                        mtime_ns,
                        kind,
                        engine,
                        encoded,
                        now,
                        now,
                    ),
                )
            log_event(
                "analysis-store",
                "cache-save",
                "persisted analysis result",
                path=normalized,
                kind=kind,
                engine=engine,
            )
            return True
        except FileNotFoundError:
            return False
        except Exception as exc:
            log_exception(
                "analysis-store",
                "save-failure",
                exc,
                path=str(path),
                kind=kind,
                engine=engine,
            )
            return False

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM analysis_results").fetchone()
        return int(row[0] if row else 0)

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            finally:
                log_event(
                    "analysis-store",
                    "finish",
                    "persistent analysis cache closed",
                    db_path=str(self.db_path),
                )
