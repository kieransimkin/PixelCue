from __future__ import annotations

import io
from PIL import Image

DEFAULT_MAX_SIZE = (320, 240)


def normalize_size(max_size: tuple[int, int]) -> tuple[int, int]:
    width, height = int(max_size[0]), int(max_size[1])
    if width <= 0 or height <= 0:
        raise ValueError("max_size dimensions must be positive")
    return width, height


def fit_image(image: Image.Image, max_size: tuple[int, int]) -> Image.Image:
    max_size = normalize_size(max_size)
    out = image.convert("RGB").copy()
    out.thumbnail(max_size)
    return out


def encode_image(
    image: Image.Image,
    *,
    output_format: str = "JPEG",
    quality: int = 82,
) -> bytes:
    fmt = str(output_format).upper()
    out = io.BytesIO()
    kwargs = {}
    if fmt in {"JPEG", "JPG", "WEBP"}:
        kwargs["quality"] = int(quality)
    if fmt in {"JPEG", "JPG"}:
        kwargs["optimize"] = True
    image.save(out, format=fmt, **kwargs)
    return out.getvalue()
