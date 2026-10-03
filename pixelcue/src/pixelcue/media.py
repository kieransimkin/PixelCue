from __future__ import annotations

import io
from pathlib import Path
from typing import Iterator

from PIL import Image

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff",
    ".heic", ".heif", ".avif", ".jxl",
}
VIDEO_EXTS = {
    ".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm", ".wmv", ".flv",
    ".mpeg", ".mpg", ".mts", ".m2ts", ".3gp", ".ogv",
}


def is_image(path: Path, mime: str = "") -> bool:
    return mime.startswith("image/") or path.suffix.lower() in IMAGE_EXTS


def is_video(path: Path, mime: str = "") -> bool:
    return mime.startswith("video/") or path.suffix.lower() in VIDEO_EXTS


def open_image(path: Path) -> Image.Image:
    with Image.open(path) as im:
        im.load()
        return im.convert("RGB")


def video_duration_seconds(path: Path) -> float | None:
    import av
    try:
        with av.open(str(path)) as container:
            if container.duration is not None:
                return float(container.duration / av.time_base)
            if container.streams.video:
                stream = container.streams.video[0]
                if stream.duration is not None and stream.time_base is not None:
                    return float(stream.duration * stream.time_base)
    except Exception:
        return None
    return None


def _frame_at(path: Path, seconds: float) -> Image.Image | None:
    import av
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                return None
            stream = container.streams.video[0]
            if stream.time_base is not None and seconds > 0:
                offset = int(seconds / float(stream.time_base))
                container.seek(offset, stream=stream, any_frame=False, backward=True)

            candidate = None
            for frame in container.decode(stream):
                candidate = frame
                frame_t = frame.time
                if seconds <= 0 or frame_t is None or frame_t >= seconds - 0.25:
                    break
            if candidate is None:
                return None
            return candidate.to_image().convert("RGB")
    except Exception:
        return None


def sample_video_frames(path: Path, interval_seconds: int = 30) -> Iterator[tuple[float, Image.Image]]:
    duration = video_duration_seconds(path)
    if duration is None:
        frame = _frame_at(path, 0.0)
        if frame is not None:
            yield 0.0, frame
        return

    # Include first frame and then 30, 60, ... while inside the duration.
    timestamps = [0.0]
    t = float(interval_seconds)
    while t <= duration + 0.01:
        timestamps.append(t)
        t += interval_seconds

    for ts in timestamps:
        frame = _frame_at(path, ts)
        if frame is not None:
            yield ts, frame


def first_video_frame(path: Path) -> Image.Image | None:
    return _frame_at(path, 0.0)


def thumbnail_jpeg(image: Image.Image, max_size: tuple[int, int] = (320, 240)) -> bytes:
    im = image.convert("RGB").copy()
    im.thumbnail(max_size)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=82, optimize=True)
    return buf.getvalue()
