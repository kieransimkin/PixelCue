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
    # When a MIME result is available, trust content detection completely.
    # Extension fallback is only for contexts where no magic result exists.
    mime = (mime or "").strip().lower()
    if mime:
        return mime.startswith("image/")
    return path.suffix.lower() in IMAGE_EXTS


def is_video(path: Path, mime: str = "") -> bool:
    mime = (mime or "").strip().lower()
    if mime:
        return mime.startswith("video/")
    return path.suffix.lower() in VIDEO_EXTS


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


def _frame_fingerprint(image: Image.Image) -> tuple[float, ...]:
    tiny = image.convert("L").resize((16, 9))
    return tuple(float(v) / 255.0 for v in tiny.getdata())


def _fingerprint_distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if not a or len(a) != len(b):
        return 1.0
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def _frames_at_timestamps(path: Path, timestamps: list[float]) -> list[tuple[float, Image.Image]]:
    """Seek several timestamps while opening the container only once."""
    import av

    output: list[tuple[float, Image.Image]] = []
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                return output
            stream = container.streams.video[0]
            for seconds in timestamps:
                try:
                    if stream.time_base is not None and seconds > 0:
                        offset = int(seconds / float(stream.time_base))
                        container.seek(offset, stream=stream, any_frame=False, backward=True)

                    candidate = None
                    for frame in container.decode(stream):
                        candidate = frame
                        frame_t = frame.time
                        if seconds <= 0 or frame_t is None or frame_t >= seconds - 0.25:
                            break
                    if candidate is not None:
                        output.append((seconds, candidate.to_image().convert("RGB")))
                except Exception:
                    # A bad seek should lose one candidate, not the whole video.
                    continue
    except Exception:
        return []
    return output


def sample_video_frames(
    path: Path,
    interval_seconds: int = 30,
    max_frames: int = 5,
) -> Iterator[tuple[float, Image.Image]]:
    """Select <=5 diverse frames with one container open and <=7 sparse seeks."""
    max_frames = max(1, min(int(max_frames), 5))
    duration = video_duration_seconds(path)
    if duration is None or duration <= 0:
        frame = _frame_at(path, 0.0)
        if frame is not None:
            yield 0.0, frame
        return

    # Seven candidates is enough to improve diversity without paying for nine
    # independent seeks. The container itself is opened only once below.
    candidate_count = min(7, max(max_frames, max_frames + 2))
    end = max(0.0, duration - min(0.25, duration * 0.01))
    timestamps = (
        [end * i / (candidate_count - 1) for i in range(candidate_count)]
        if candidate_count > 1 else [0.0]
    )

    candidates = [
        (ts, frame, _frame_fingerprint(frame))
        for ts, frame in _frames_at_timestamps(path, timestamps)
    ]

    if len(candidates) <= max_frames:
        for ts, frame, _fp in candidates:
            yield ts, frame
        return

    selected = [0]
    remaining = set(range(1, len(candidates)))
    while remaining and len(selected) < max_frames:
        best = max(
            remaining,
            key=lambda idx: min(
                _fingerprint_distance(candidates[idx][2], candidates[j][2])
                for j in selected
            ),
        )
        selected.append(best)
        remaining.remove(best)

    for idx in sorted(selected, key=lambda i: candidates[i][0]):
        ts, frame, _fp = candidates[idx]
        yield ts, frame


def first_video_frame(path: Path) -> Image.Image | None:
    return _frame_at(path, 0.0)


def thumbnail_jpeg(image: Image.Image, max_size: tuple[int, int] = (320, 240)) -> bytes:
    im = image.convert("RGB").copy()
    im.thumbnail(max_size)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=82, optimize=True)
    return buf.getvalue()
