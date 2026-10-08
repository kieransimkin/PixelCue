from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .archives import is_archive, sample_archive
from .media import first_video_frame, is_image, is_video, open_image, sample_video_frames
from .metadata import MagicDetector
from .model import DEFAULT_TAGGER_PROFILE, TAGGER_PROFILES, create_image_tagger

ProgressCallback = Callable[[dict[str, Any]], None]


class UnsupportedMediaError(ValueError):
    """Raised when content detection does not identify supported visual media."""


def _deduplicate(tags: list[str]) -> list[str]:
    unique: dict[str, str] = {}
    for raw in tags:
        tag = str(raw).strip()
        if tag:
            unique.setdefault(tag.casefold(), tag)
    return list(unique.values())


class KeywordExtractionService:
    """Serial, model-reusing keyword extraction for one local media file."""

    def __init__(
        self,
        *,
        tagger_factory: Callable[..., Any] = create_image_tagger,
        detector_factory: Callable[[], MagicDetector] = MagicDetector,
    ) -> None:
        self._tagger_factory = tagger_factory
        self._detector = detector_factory()
        self._taggers: dict[str, Any] = {}
        self._inference_lock = threading.Lock()
        self._active_progress: ProgressCallback | None = None

    def _emit(self, stage: str, message: str, **values: Any) -> None:
        callback = self._active_progress
        if callback is not None:
            callback({"stage": stage, "message": message, **values})

    def _model_status(self, message: str) -> None:
        self._emit("model", message)

    def _model_progress(self, completed: int, total: int) -> None:
        self._emit(
            "model-download",
            "Preparing the selected PixelCue model.",
            completed=max(0, int(completed)),
            total=max(0, int(total)),
        )

    def _tagger(self, model: str) -> Any:
        if model not in TAGGER_PROFILES:
            available = ", ".join(sorted(TAGGER_PROFILES))
            raise ValueError(f"Unknown PixelCue model {model!r}. Available: {available}.")
        if model not in self._taggers:
            self._taggers[model] = self._tagger_factory(
                model,
                status_callback=self._model_status,
                progress_callback=self._model_progress,
            )
        return self._taggers[model]

    def _tag_image(self, tagger: Any, image: Any) -> list[str]:
        try:
            return list(tagger.tags_for_image(image))
        finally:
            close = getattr(image, "close", None)
            if callable(close):
                close()

    def _image_keywords(self, path: Path, tagger: Any) -> list[str]:
        self._emit("analysis", "Tagging image.", completed=0, total=1)
        tags = self._tag_image(tagger, open_image(path))
        self._emit("analysis", "Tagged image.", completed=1, total=1)
        return tags

    def _video_keywords(self, path: Path, tagger: Any) -> list[str]:
        frames = list(sample_video_frames(path, max_frames=5))
        if not frames:
            raise UnsupportedMediaError("PixelCue could not decode a frame from the video.")
        tags: list[str] = []
        total = len(frames)
        for index, (_timestamp, frame) in enumerate(frames, start=1):
            self._emit(
                "analysis",
                "Tagging sampled video frame.",
                completed=index - 1,
                total=total,
            )
            tags.extend(self._tag_image(tagger, frame))
        self._emit(
            "analysis",
            "Tagged sampled video frames.",
            completed=total,
            total=total,
        )
        return tags

    def _archive_keywords(self, path: Path, mime: str, tagger: Any) -> list[str]:
        sampled = sample_archive(path, sample_size=5, mime=mime, detector=self._detector)
        tags: list[str] = []
        total = len(sampled.paths)
        try:
            for index, member in enumerate(sampled.paths, start=1):
                self._emit(
                    "analysis",
                    "Tagging sampled archive member.",
                    completed=index - 1,
                    total=total,
                )
                member_mime, _description = self._detector.identify(member)
                if is_image(member, member_mime):
                    image = open_image(member)
                elif is_video(member, member_mime):
                    image = first_video_frame(member)
                    if image is None:
                        continue
                else:
                    continue
                tags.extend(self._tag_image(tagger, image))
        finally:
            sampled.cleanup()
        self._emit(
            "analysis",
            "Tagged sampled archive members.",
            completed=total,
            total=total,
        )
        return tags

    def extract_keywords(
        self,
        path: str | Path,
        *,
        model: str = DEFAULT_TAGGER_PROFILE,
        progress: ProgressCallback | None = None,
    ) -> list[str]:
        candidate = Path(path).expanduser()
        if not candidate.exists():
            raise FileNotFoundError(f"File does not exist: {candidate}")
        if not candidate.is_file():
            raise IsADirectoryError(f"Expected one file, not a directory: {candidate}")
        candidate = candidate.resolve(strict=True)

        with self._inference_lock:
            self._active_progress = progress
            try:
                self._emit("inspection", "Inspecting media type.")
                mime, _description = self._detector.identify(candidate)
                if not (
                    is_image(candidate, mime)
                    or is_video(candidate, mime)
                    or is_archive(candidate, mime)
                ):
                    raise UnsupportedMediaError(
                        f"Unsupported or unverified media type {mime!r}."
                    )
                tagger = self._tagger(model)
                if is_image(candidate, mime):
                    tags = self._image_keywords(candidate, tagger)
                elif is_video(candidate, mime):
                    tags = self._video_keywords(candidate, tagger)
                else:
                    tags = self._archive_keywords(candidate, mime, tagger)
                keywords = _deduplicate(tags)
                self._emit(
                    "result",
                    "Keywords ready.",
                    completed=len(keywords),
                    total=len(keywords),
                )
                return keywords
            finally:
                self._active_progress = None
