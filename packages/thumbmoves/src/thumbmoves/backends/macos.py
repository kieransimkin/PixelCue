from __future__ import annotations

from pathlib import Path
from PIL import Image


def load_cached_thumbnail(path: Path) -> Image.Image | None:
    """macOS has no public strict cache-only Quick Look query.

    The cross-platform cache-only API therefore returns a clean miss instead of
    causing Quick Look to generate a thumbnail. A future backend can add a
    documented cache-only mechanism if Apple exposes one.
    """
    return None
