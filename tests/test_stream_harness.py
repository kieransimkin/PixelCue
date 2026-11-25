import io
import os

import pytest
from PIL import Image

from pixelcue.stream_harness import MjpegIngress

pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="Selectable MJPEG pipe harness is POSIX; decoded-frame API is portable",
)


def test_mjpeg_pipe_keeps_latest_frame_and_stops_reader():
    read_fd, write_fd = os.pipe()
    try:
        with MjpegIngress(read_fd, seconds=2) as ingress:
            for colour in ["red", "green", "blue"]:
                image = Image.new("RGB", (64, 32), colour)
                encoded = io.BytesIO()
                image.save(encoded, format="JPEG")
                image.close()
                os.write(write_fd, encoded.getvalue())
            os.close(write_fd)
            write_fd = None
            assert ingress.done.wait(2)
            assert ingress.error is None
            image, sequence, _ = ingress.frames.take()
            assert sequence == 2 and ingress.frames.dropped == 2
            assert image.getpixel((20, 20))[2] > 200
            image.close()
        assert not ingress._thread.is_alive()
    finally:
        os.close(read_fd)
        if write_fd is not None:
            os.close(write_fd)


def test_idle_pipe_reader_can_be_cancelled_without_eof():
    read_fd, write_fd = os.pipe()
    try:
        with MjpegIngress(read_fd, seconds=10) as ingress:
            pass
        assert not ingress._thread.is_alive()
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_truncated_frame_is_an_explicit_failure():
    read_fd, write_fd = os.pipe()
    try:
        with MjpegIngress(read_fd, seconds=2) as ingress:
            os.write(write_fd, b"\xff\xd8broken")
            os.close(write_fd)
            write_fd = None
            assert ingress.done.wait(2)
            assert "Truncated" in ingress.error
    finally:
        os.close(read_fd)
        if write_fd is not None:
            os.close(write_fd)
