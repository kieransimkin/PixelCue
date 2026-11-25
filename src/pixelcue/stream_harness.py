"""Bounded MJPEG pipe and recorded-video harnesses for live frame consumers.

An admitted capture producer may pipe JPEG frames here. No URL, device or
screen discovery is performed. MJPEG timestamps are ingestion clocks; use the
decoded-frame API when the capture owner can supply original capture times.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import select
import threading
import time
from pathlib import Path

from .live_analysis import LatestFrameBuffer, LiveAnalyser


class MjpegIngress:
    """Own a stoppable POSIX pipe reader with a two MiB encoded-frame ceiling."""

    def __init__(self, fd, *, seconds=60):
        if os.name != "posix":
            raise ValueError("MJPEG pipe harness requires POSIX selectable pipes")
        if not 0 < seconds <= 3600:
            raise ValueError("Invalid pipe time budget")
        self.fd, self.seconds = fd, seconds
        self.frames = LatestFrameBuffer()
        self.stop = threading.Event()
        self.done = threading.Event()
        self.error = None
        self._thread = threading.Thread(
            target=self._read, name="pixelcue-mjpeg", daemon=False
        )

    def _read(self):
        from PIL import Image

        deadline, sequence, pending = time.monotonic() + self.seconds, 0, bytearray()
        try:
            while not self.stop.is_set() and time.monotonic() < deadline:
                if not select.select([self.fd], [], [], 0.1)[0]:
                    continue
                block = os.read(self.fd, 65536)
                if not block:
                    break
                pending.extend(block)
                while b"\xff\xd9" in pending:
                    end = pending.index(b"\xff\xd9") + 2
                    encoded, pending = bytes(pending[:end]), pending[end:]
                    if (
                        not encoded.startswith(b"\xff\xd8")
                        or len(encoded) > 2 * 1024**2
                    ):
                        raise ValueError("Invalid or oversized JPEG frame")
                    with Image.open(io.BytesIO(encoded)) as image:
                        if image.width * image.height > 40_000_000:
                            raise ValueError("Oversized decoded frame")
                        self.frames.submit(
                            image, sequence=sequence, captured_at=time.monotonic()
                        )
                    sequence += 1
                if len(pending) > 2 * 1024**2:
                    raise ValueError("Unterminated or oversized JPEG frame")
            if pending and not self.stop.is_set():
                raise ValueError("Truncated JPEG at end of stream")
        except (OSError, ValueError) as error:
            self.error = str(error)
        finally:
            self.done.set()

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self._thread.join(timeout=2)
        self.frames.close()
        if self._thread.is_alive():
            raise RuntimeError("MJPEG reader failed to stop")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--mjpeg-stdin", action="store_true")
    source.add_argument("--video", type=Path)
    parser.add_argument("--stream-id", required=True)
    parser.add_argument(
        "--models",
        nargs="*",
        choices=["rtdetr", "mobilenet", "layout", "ocr"],
        default=["ocr"],
    )
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=float, default=2)
    parser.add_argument("--max-edge", type=int, default=1024)
    args = parser.parse_args(argv)
    if not 0.1 <= args.fps <= 10:
        parser.error("analysis fps must be 0.1..10")
    partial = False

    def output(analyser, image, seq, stamp, dropped=0, media_timestamp=None):
        nonlocal partial
        result = analyser.analyse(
            image, sequence=seq, captured_at=stamp, dropped_frames=dropped
        )
        result["stream"]["clock_kind"] = "ingestion_monotonic"
        result["stream"]["media_timestamp_seconds"] = media_timestamp
        partial |= result["status"] != "completed"
        print(json.dumps(result, allow_nan=False, separators=(",", ":")), flush=True)

    with LiveAnalyser(
        args.stream_id,
        models=args.models,
        models_dir=args.models_dir,
        max_edge=args.max_edge,
        max_frames=args.frames,
        max_seconds=args.seconds,
    ) as analyser:
        if args.mjpeg_stdin:
            with MjpegIngress(0, seconds=args.seconds) as ingress:
                deadline = time.monotonic() + args.seconds
                while time.monotonic() < deadline and analyser.frames < args.frames:
                    frame = ingress.frames.take()
                    if frame is None:
                        if ingress.done.wait(0.05):
                            break
                        continue
                    image, seq, stamp = frame
                    try:
                        output(analyser, image, seq, stamp, ingress.frames.dropped)
                    finally:
                        image.close()
                    ingress.stop.wait(
                        min(1 / args.fps, max(0, deadline - time.monotonic()))
                    )
                if ingress.error:
                    raise ValueError(ingress.error)
        else:
            import av

            if not args.video.is_file() or args.video.stat().st_size > 256 * 1024**2:
                raise ValueError("Expected one local video of at most 256 MiB")
            with av.open(str(args.video)) as container:
                next_pts = 0.0
                for seq, frame in enumerate(container.decode(video=0)):
                    if (
                        time.monotonic() - analyser.started >= args.seconds
                        or analyser.frames >= args.frames
                    ):
                        break
                    timestamp = float(frame.time) if frame.time is not None else None
                    if timestamp is not None and timestamp < next_pts:
                        continue
                    if timestamp is not None:
                        next_pts = timestamp + 1 / args.fps
                    if frame.width * frame.height > 40_000_000:
                        raise ValueError("Oversized decoded video frame")
                    image = frame.to_image()
                    try:
                        output(
                            analyser,
                            image,
                            seq,
                            time.monotonic(),
                            media_timestamp=timestamp,
                        )
                    finally:
                        image.close()
        if analyser.frames == 0:
            raise ValueError("No frames analysed")
    return 2 if partial else 0


if __name__ == "__main__":
    raise SystemExit(main())
