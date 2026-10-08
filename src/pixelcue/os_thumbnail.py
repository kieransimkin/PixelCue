from __future__ import annotations

from pathlib import Path

from thumbmoves import backend_info, get_cached_thumbnail

from .logging_utils import log_event

DEFAULT_MAX_SIZE = (320, 240)


def os_cached_thumbnail_jpeg(
    path: str | Path,
    max_size: tuple[int, int] = DEFAULT_MAX_SIZE,
) -> bytes | None:
    """PixelCue logging adapter around the reusable ThumbMoves package."""
    p = Path(path)
    result = get_cached_thumbnail(p, max_size=max_size, output_format="JPEG")
    if result is not None:
        log_event(
            "thumbmoves", "hit",
            "restored thumbnail from operating-system cache",
            path=str(p), backend=result.backend, bytes=len(result.data),
        )
        return result.data

    log_event(
        "thumbmoves", "miss",
        "no cached operating-system thumbnail available",
        path=str(p), backend=backend_info().name,
    )
    return None
