from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from PIL import Image

from .common import DEFAULT_MAX_SIZE, encode_image, fit_image, normalize_size


@dataclass(frozen=True)
class BackendInfo:
    name: str
    platform: str
    cache_only_supported: bool
    notes: str = ""


@dataclass(frozen=True)
class ThumbnailResult:
    data: bytes
    image: Image.Image
    backend: str
    path: Path
    output_format: str
    cache_only: bool = True

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size


def backend_info(platform: str | None = None) -> BackendInfo:
    platform = platform or sys.platform
    if platform == "win32":
        return BackendInfo("windows-shell", platform, True,
                           "IShellItemImageFactory with INCACHEONLY + THUMBNAILONLY")
    if platform == "darwin":
        return BackendInfo("macos-quicklook", platform, False,
                           "Quick Look has no public strict cache-only query")
    return BackendInfo("freedesktop-xdg", platform, True,
                       "Freedesktop thumbnail-spec cache lookup")


def get_cached_thumbnail_image(
    path: str | Path,
    max_size: tuple[int, int] = DEFAULT_MAX_SIZE,
) -> Image.Image | None:
    """Return an existing OS-managed thumbnail as a Pillow image.

    This function never opens the original media file and never intentionally
    asks the OS to generate a thumbnail. On macOS, where no public strict
    cache-only Quick Look API exists, it returns ``None``.
    """
    p = Path(path)
    max_size = normalize_size(max_size)

    if sys.platform == "win32":
        from .backends.windows import load_cached_thumbnail
        image = load_cached_thumbnail(p, max(max_size))
    elif sys.platform == "darwin":
        from .backends.macos import load_cached_thumbnail
        image = load_cached_thumbnail(p)
    else:
        from .backends.freedesktop import load_cached_thumbnail
        image = load_cached_thumbnail(p)

    return fit_image(image, max_size) if image is not None else None


def get_cached_thumbnail(
    path: str | Path,
    max_size: tuple[int, int] = DEFAULT_MAX_SIZE,
    *,
    output_format: str = "JPEG",
    quality: int = 82,
) -> ThumbnailResult | None:
    image = get_cached_thumbnail_image(path, max_size=max_size)
    if image is None:
        return None
    fmt = str(output_format).upper()
    return ThumbnailResult(
        data=encode_image(image, output_format=fmt, quality=quality),
        image=image,
        backend=backend_info().name,
        path=Path(path),
        output_format=fmt,
        cache_only=True,
    )


def get_cached_thumbnail_bytes(
    path: str | Path,
    max_size: tuple[int, int] = DEFAULT_MAX_SIZE,
    *,
    output_format: str = "JPEG",
    quality: int = 82,
) -> bytes | None:
    result = get_cached_thumbnail(
        path, max_size=max_size, output_format=output_format, quality=quality
    )
    return result.data if result is not None else None
