from __future__ import annotations

import json
import mimetypes
import os
import stat as statmod
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import ExifTags, Image, UnidentifiedImageError


def _iso(ts: float | None) -> str:
    if ts is None:
        return ""
    try:
        return datetime.fromtimestamp(ts).astimezone().isoformat()
    except (OSError, OverflowError, ValueError):
        return ""


def _jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def exif_json(path: Path) -> str:
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            if not exif:
                return ""
            decoded: dict[str, Any] = {}
            for key, value in exif.items():
                name = ExifTags.TAGS.get(key, str(key))
                decoded[name] = _jsonable(value)

            # Pillow exposes GPS as a nested IFD on newer versions.
            try:
                gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
                if gps_ifd:
                    decoded["GPSInfo"] = {
                        ExifTags.GPSTAGS.get(k, str(k)): _jsonable(v)
                        for k, v in gps_ifd.items()
                    }
            except Exception:
                pass
            return json.dumps(decoded, ensure_ascii=False, sort_keys=True)
    except Exception:
        return ""


class MagicDetector:
    """Thread-local-ish wrapper: create one per scanner worker."""

    def __init__(self) -> None:
        self._mime = None
        self._desc = None
        try:
            import magic
            self._mime = magic.Magic(mime=True)
            self._desc = magic.Magic()
        except Exception:
            self._mime = None
            self._desc = None

    def identify(self, path: Path) -> tuple[str, str]:
        if self._mime is not None and self._desc is not None:
            try:
                return (
                    str(self._mime.from_file(str(path))),
                    str(self._desc.from_file(str(path))),
                )
            except Exception:
                pass

        # Fall back to the system `file` command when available.
        try:
            mime = subprocess.run(
                ["file", "-b", "--mime-type", "--", str(path)],
                capture_output=True, text=True, check=True, timeout=15
            ).stdout.strip()
            desc = subprocess.run(
                ["file", "-b", "--", str(path)],
                capture_output=True, text=True, check=True, timeout=15
            ).stdout.strip()
            return mime, desc
        except Exception:
            guessed, _ = mimetypes.guess_type(path.name)
            return guessed or "application/octet-stream", ""


def stat_record(path: Path, detector: MagicDetector, *, follow_symlinks: bool = False) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "filename": path.name,
        "path": str(path),
        "tags": [],
        "exif": "",
        "extension": path.suffix.lower(),
        "mime_type": "",
        "file_type": "",
        "size_bytes": "",
        "modified": "",
        "accessed": "",
        "created_or_changed": "",
        "birth_time": "",
        "permissions": "",
        "uid": "",
        "gid": "",
        "symlink_target": "",
        "category": "",
        "extra_metadata": {},
        "error": "",
    }
    try:
        st = path.stat() if follow_symlinks else path.lstat()
        rec["size_bytes"] = st.st_size
        rec["modified"] = _iso(st.st_mtime)
        rec["accessed"] = _iso(st.st_atime)
        rec["created_or_changed"] = _iso(st.st_ctime)
        rec["birth_time"] = _iso(getattr(st, "st_birthtime", None))
        rec["permissions"] = statmod.filemode(st.st_mode)
        rec["uid"] = getattr(st, "st_uid", "")
        rec["gid"] = getattr(st, "st_gid", "")
    except Exception as e:
        rec["error"] = f"stat: {e}"
        return rec

    if path.is_symlink() and not follow_symlinks:
        try:
            rec["symlink_target"] = os.readlink(path)
            rec["category"] = "symlink"
            rec["mime_type"] = "inode/symlink"
            rec["file_type"] = "symbolic link"
        except Exception as e:
            rec["error"] = f"readlink: {e}"
        return rec

    if path.is_file():
        mime, desc = detector.identify(path)
        rec["mime_type"] = mime
        rec["file_type"] = desc
        rec["exif"] = exif_json(path)

    return rec
