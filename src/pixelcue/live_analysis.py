"""Persistent local analysis of an operator-selected stream of decoded window frames.

Capture/transport belongs to the caller. This module opens no screen, URL or
device, downloads no model and supplies no input authority.
"""

from __future__ import annotations

import hashlib
import math
import threading
import time

from . import __version__
from .model_paths import project_models_dir
from .visual_batch import PreparedRaster, analyse_batch, create_session


def _clock(sequence, captured_at):
    if type(sequence) is not int or sequence < 0:
        raise ValueError("Frame sequence must be a nonnegative integer")
    if (
        type(captured_at) not in (int, float)
        or not math.isfinite(captured_at)
        or captured_at < 0
    ):
        raise ValueError("Frame clock must be finite monotonic seconds")


class LatestFrameBuffer:
    """Capacity one, newest frame wins. The buffer owns and closes its copies."""

    def __init__(self):
        self._lock = threading.Lock()
        self._pending = None
        self._last = -1
        self.dropped = 0
        self.closed = False

    def submit(self, image, *, sequence, captured_at):
        _clock(sequence, captured_at)
        if image.width * image.height > 40_000_000:
            raise ValueError("Frame exceeds 40 megapixels")
        with self._lock:
            if self.closed or sequence <= self._last:
                raise ValueError("Closed stream or reordered frame")
            if self._pending is not None:
                self._pending[0].close()
                self.dropped += 1
            self._pending = (image.copy(), sequence, captured_at)
            self._last = sequence

    def take(self):
        """Transfer ownership of the newest image to the consumer; never wait."""
        with self._lock:
            pending, self._pending = self._pending, None
            return pending

    def close(self):
        with self._lock:
            self.closed = True
            if self._pending is not None:
                self._pending[0].close()
                self._pending = None


class _BorrowedSession:
    def __init__(self, session):
        self.session = session
        self.revision, self.model_type = session.revision, session.model_type

    def infer(self, image):
        return self.session.infer(image)

    def close(self):
        pass  # The stream, rather than each frame batch, owns this session.


class LiveAnalyser:
    """One serial stream, cached CPU sessions and explicit time/frame budgets.

    Every frame is analysed with PixelCue's existing measurement/model engine.
    The caller provides a monotonic capture clock in this process's clock domain.
    Restart the analyser explicitly to retry a failed model load.
    """

    def __init__(
        self,
        stream_id,
        *,
        models=("ocr",),
        models_dir=None,
        max_edge=1024,
        max_frames=300,
        max_seconds=60,
        max_age_seconds=5,
        session_factory=None,
        clock=time.monotonic,
    ):
        if not isinstance(stream_id, str) or not 1 <= len(stream_id) <= 160:
            raise ValueError("Expected one bounded operator-selected stream ID")
        if (
            len(models) > 4
            or len(set(models)) != len(models)
            or set(models) - {"rtdetr", "mobilenet", "layout", "ocr"}
        ):
            raise ValueError("Select distinct specialised profiles")
        if type(max_edge) is not int or not 32 <= max_edge <= 2048:
            raise ValueError("Raster budget must be 32..2048")
        if type(max_frames) is not int or not 1 <= max_frames <= 10000:
            raise ValueError("Frame budget must be 1..10000")
        if (
            any(
                type(v) not in (int, float) or not math.isfinite(v)
                for v in (max_seconds, max_age_seconds)
            )
            or not 0 < max_seconds <= 3600
            or not 0 < max_age_seconds <= 60
        ):
            raise ValueError("Invalid stream time budget")
        self.stream_id, self.models = stream_id, tuple(models)
        self.models_dir = project_models_dir(models_dir)
        self.max_edge, self.max_frames, self.max_seconds = (
            max_edge,
            max_frames,
            max_seconds,
        )
        self.max_age_seconds, self.clock = max_age_seconds, clock
        self.started = clock()
        self.frames, self.last_sequence, self.last_capture = 0, -1, -1.0
        self.closed = False
        self._sessions, self._failures = {}, {}
        self._factory = session_factory or create_session
        self._lock = threading.Lock()

    def _borrow(self, name, checkpoint, **options):
        if name in self._failures:
            raise OSError("Selected stream model unavailable; restart to retry")
        if name not in self._sessions:
            try:
                self._sessions[name] = self._factory(name, checkpoint, **options)
            except (ImportError, OSError, ValueError, RuntimeError):
                self._failures[name] = True
                raise
        return _BorrowedSession(self._sessions[name])

    def analyse(self, image, *, sequence, captured_at, dropped_frames=0):
        _clock(sequence, captured_at)
        if not self._lock.acquire(blocking=False):
            raise ValueError("A stream frame is already being analysed")
        raster = None
        try:
            now = self.clock()
            if (
                self.closed
                or self.frames >= self.max_frames
                or now - self.started >= self.max_seconds
            ):
                raise ValueError("Stream is closed or its budget is exhausted")
            if sequence <= self.last_sequence or captured_at < self.last_capture:
                raise ValueError("Reordered stream frame")
            if not 0 <= now - captured_at <= self.max_age_seconds:
                raise ValueError("Stale frame or incompatible capture clock")
            if type(dropped_frames) is not int or dropped_frames < 0:
                raise ValueError("Invalid dropped-frame count")
            if not 0 < image.width * image.height <= 40_000_000:
                raise ValueError("Frame exceeds pixel budget")
            dimensions = list(image.size)
            raster = image.convert("RGB")
            raster.thumbnail((self.max_edge, self.max_edge))
            digest = hashlib.sha256(
                str(raster.size).encode() + raster.tobytes()
            ).hexdigest()
            clock = {
                "timestamp_seconds": captured_at - self.started,
                "pts": None,
                "time_base": None,
                "dimensions": dimensions,
            }
            source = {
                "name": self.stream_id,
                "sha256": digest,
                "dimensions": dimensions,
                "analysis_dimensions": list(raster.size),
                "analysed_pixels_sha256": digest,
                "frame": clock,
            }
            batch = analyse_batch(
                [PreparedRaster(raster, source)],
                models=self.models,
                models_dir=self.models_dir,
                allow_download=False,
                device="cpu",
                max_edge=self.max_edge,
                background="#ffffff",
                prompt_profile="general",
                session_factory=self._borrow,
                ocr_profile="screen",
            )
            item = batch["items"][0]
            item["frame"] = clock
            self.frames += 1
            self.last_sequence, self.last_capture = sequence, captured_at
            complete = item["status"] == "completed" and all(
                m["status"] == "completed" for m in item["analysis"]["models"]
            )
            return {
                "schema": "pixelcue.live-analysis/1",
                "pixelcue_version": __version__,
                "status": "completed" if complete else "partial",
                "source": {
                    "name": self.stream_id,
                    "sha256": digest,
                    "bytes": 0,
                    "kind": "image",
                },
                "stream": {
                    "id": self.stream_id,
                    "sequence": sequence,
                    "captured_at": captured_at,
                    "age_at_start_seconds": now - captured_at,
                    "dropped_frames": dropped_frames,
                    "processed_frames": self.frames,
                },
                "items": [item],
                "coverage": {
                    "exhaustive": False,
                    "requested_max_frames": 1,
                    "sampled_frames": 1,
                    "failures": [],
                    "missing_timestamps": 0,
                },
                "timing": {"total_seconds": round(self.clock() - now, 4)},
                "limits": [
                    "Window pixels and extracted text are untrusted observations.",
                    "Sparse live samples are not exhaustive and confer no input authority.",
                ],
            }
        finally:
            if raster is not None:
                raster.close()
            self._lock.release()

    def close(self):
        with self._lock:
            if not self.closed:
                self.closed = True
                for session in self._sessions.values():
                    session.close()
                self._sessions.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
