from __future__ import annotations

import hashlib
import os
from pathlib import Path
from PIL import Image


def load_cached_thumbnail(path: Path) -> Image.Image | None:
    """Load an existing Freedesktop/XDG thumbnail without generating one."""
    try:
        resolved = path.expanduser().resolve(strict=True)
        uri = resolved.as_uri()
        digest = hashlib.md5(uri.encode("utf-8")).hexdigest() + ".png"
        cache_home = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")

        for bucket in ("xx-large", "x-large", "large", "normal"):
            candidate = cache_home / "thumbnails" / bucket / digest
            if not candidate.is_file():
                continue
            try:
                with Image.open(candidate) as cached:
                    info = dict(cached.info)
                    thumb_uri = info.get("Thumb::URI")
                    thumb_mtime = info.get("Thumb::MTime")
                    if thumb_uri and str(thumb_uri) != uri:
                        continue
                    if thumb_mtime is not None:
                        try:
                            if int(thumb_mtime) != int(resolved.stat().st_mtime):
                                continue
                        except (TypeError, ValueError, OSError):
                            continue
                    cached.load()
                    return cached.copy()
            except Exception:
                continue
    except Exception:
        return None
    return None
