# Continuous window analysis

`pixelcue.live_analysis.LiveAnalyser` accepts decoded RGB window frames from an
already admitted capture owner. It keeps selected CPU model sessions loaded
between samples. It reuses the same PixelCue measurement and specialised-model
engine as the image/video API; it does not discover screens or open input devices.

[Actual captured window changing during the Potbot chroot benchmark](images/live-window.mp4)

Actual original Chromium fixture in a read-only rootfs under Bubblewrap and
Xpra/Xorg dummy. The retained capture clocks are controller ingestion clocks.
The movie was encoded from those captured JPEGs; all encoded frames were
independently decoded. It is not an installed production operator/Syn claim.

```python
import time
from pixelcue.live_analysis import LiveAnalyser

with LiveAnalyser("admitted-session:window-0", models=["ocr"],
                  models_dir="models", max_frames=120, max_seconds=60) as stream:
    for sequence, image in enumerate(admitted_capture_frames()):
        result = stream.analyse(image, sequence=sequence,
                                captured_at=time.monotonic())
        consume(result)
```

Provide original capture timestamps in the same monotonic clock domain when
available. Label ingestion times when that is the actual evidence. Every result
retains stream ID, strictly increasing sequence, frame hash, source/analysis
dimensions, timestamp, age at analysis start, model revisions/status and timing.
Image hashes bind the analysed raster, rather than an encoded source file.
Each frame is an observation; sampling does not establish exhaustive coverage.

The default is OCR. Select any of `rtdetr`, `mobilenet`, `layout` and `ocr`.
Provision weights separately beneath ignored project-local `models/`; live
inference never downloads them. The stream owns and closes its model sessions.
A failed model load remains unavailable until an explicit stream restart.
Loading multiple selected neural models keeps those models resident together;
budget memory for the chosen set. No general FPS or accuracy guarantee is made.

Window OCR uses two bounded scales when the raster is at least 256 pixels on
its shorter edge. The extra pass recovered a canvas-button label missed in the
real benchmark. Word boxes map back to the original analysis raster and
duplicates are merged. These are two views of the same Tesseract evidence,
not independent model consensus. Ordinary document OCR retains one pass.
Adjacent confident word regions are returned as `result.lines`; they do not
establish semantic roles or control actionability.

## Capture streams and replay harnesses

`LatestFrameBuffer` has capacity one. A producer submits copied images with
capture sequence/time, replacing and closing a superseded pending image.
The consumer takes ownership of the latest image and closes it after analysis.
Record the dropped count. Potbot's `WindowAnalysisStream` composes this same
policy with an admitted window owner and a persistent app observer.

```sh
# POSIX selectable pipe: concatenate complete JPEG frames from an owned capture.
owned-window-capture | python -m pixelcue.stream_harness --mjpeg-stdin \
  --stream-id admitted-window-0 --models ocr --seconds 30 --frames 60 --fps 2
# Local recorded-video replay, preserving media timestamps separately.
python -m pixelcue.stream_harness --video captured-window.mp4 \
  --stream-id replay-window-0 --models ocr --seconds 30 --frames 60 --fps 2
```

The MJPEG reader is a stoppable POSIX pipe reader, with a two MiB JPEG limit,
40-megapixel decoded limit, capacity-one queue and an explicit deadline. It
opens no URL or screen. Missing/invalid/truncated frames are explicit failures.
The decoded-frame API and file-replay harness also work on Windows. NDJSON
results identify ingestion clocks; recorded media PTS is a separate field.

An analysis/capture operation already in progress may finish beyond a soft
duration boundary. The capture/input owner must enforce its own hard deadlines
and process shutdown. Close the stream and its source explicitly on cancellation.
The test harness verifies idle-pipe cancellation and reader-thread shutdown.

## App understanding

Use Potbot's bundled `app-understanding` skill and `AppObserver` to combine
PixelCue text/regions with available app semantics. It maps resized pixels to
the logical viewport, rechecks targets before admitted input and verifies a new
observation afterward. An OCR region, object box or document layout region
grants no input authority. See [Potbot's app workflow](https://github.com/Synsphere/Potbot/blob/main/docs/app-understanding.md).
