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


def _frames_at_timestamps(path: Path, timestamps: list[float], *, evidence=False,
                          max_edge=None, failures=None) -> list:
    """Seek several timestamps while opening the container only once."""
    import av

    output: list[tuple[float, Image.Image]] = []
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                return output
            stream = container.streams.video[0]
            origin = float((stream.start_time or 0) * stream.time_base) if stream.time_base else 0.0
            for seconds in timestamps:
                try:
                    if stream.time_base is not None and seconds > 0:
                        offset = int((seconds + origin) / float(stream.time_base))
                        container.seek(offset, stream=stream, any_frame=False, backward=True)

                    candidate = None
                    reached = False
                    exhausted_budget = False
                    for index, frame in enumerate(container.decode(stream)):
                        if evidence and index >= 240:
                            exhausted_budget = True
                            break
                        candidate = frame
                        frame_t = None if frame.time is None else frame.time - origin
                        tolerance = 0.0 if evidence else 0.25
                        if seconds <= 0 or frame_t is None or frame_t >= seconds - tolerance:
                            reached = True
                            break
                    if evidence and (candidate is None or (not reached and exhausted_budget)):
                        if failures is not None:
                            failures.append({"requested_seconds": seconds, "reason": "No frame reached within the 240-frame seek budget"})
                        continue
                    if candidate is not None:
                        if candidate.width * candidate.height > 40_000_000:
                            raise ValueError("Video frame exceeds pixel limit")
                        image = candidate.to_image().convert("RGB")
                        dimensions = list(image.size)
                        if max_edge is not None:
                            image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
                        clock = {"requested_seconds": seconds,
                                 "timestamp_seconds": None if candidate.time is None else candidate.time - origin,
                                 "pts": candidate.pts,
                                 "time_base": str(candidate.time_base) if candidate.time_base else None,
                                 "stream_origin_seconds": origin, "dimensions": dimensions}
                        output.append((clock if evidence else seconds, image))
                except Exception as exc:
                    if failures is not None:
                        failures.append({"requested_seconds": seconds, "reason": type(exc).__name__})
                    # A bad seek should lose one candidate, not the whole video.
                    continue
    except Exception as exc:
        for _, image in output:
            image.close()
        if failures is not None:
            failures.append({"requested_seconds": None, "reason": type(exc).__name__})
        return []
    return output


def sample_video_evidence(path: Path, *, max_frames=5, max_edge=768):
    """Bounded diverse samples with decoded PTS, rather than requested seek times.

    Caller owns returned images. Missing clocks stay unknown; the failure ledger
    retains every unsuccessful seek. This is sparse inspection, not action tracking.
    """
    if type(max_frames) is not int or not 1 <= max_frames <= 5:
        raise ValueError("max_frames must be 1..5")
    if type(max_edge) is not int or not 32 <= max_edge <= 2048:
        raise ValueError("max_edge must be 32..2048")
    duration = video_duration_seconds(path)
    failures = []
    count = min(7, max_frames + 2) if duration and duration > 0 else 1
    end = max(0.0, duration - min(0.25, duration * 0.01)) if duration else 0.0
    timestamps = [end * i / (count - 1) for i in range(count)] if count > 1 else [0.0]
    candidates = _frames_at_timestamps(path, timestamps, evidence=True, max_edge=max_edge,
                                     failures=failures)
    unique, seen = [], set()
    for clock, image in candidates:
        key = (clock["pts"], clock["time_base"])
        if clock["pts"] is not None and key in seen:
            image.close()
        else:
            seen.add(key)
            unique.append((clock, image, _frame_fingerprint(image)))
    selected = [0] if unique else []
    remaining = set(range(1, len(unique)))
    while remaining and len(selected) < max_frames:
        chosen = max(sorted(remaining), key=lambda i: min(
            _fingerprint_distance(unique[i][2], unique[j][2]) for j in selected))
        selected.append(chosen)
        remaining.remove(chosen)
    for index in remaining:
        unique[index][1].close()
    return {"frames": [(unique[i][0], unique[i][1]) for i in sorted(selected)],
            "duration_seconds": duration, "candidate_count": count, "failures": failures}


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
