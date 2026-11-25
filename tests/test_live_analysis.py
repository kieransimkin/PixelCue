import pytest
from PIL import Image

from pixelcue.live_analysis import LatestFrameBuffer, LiveAnalyser


def test_persistent_session_and_explicit_cleanup():
    calls = []

    class Session:
        revision, model_type = "fixture", "fixture"

        def infer(self, image):
            return {"categories": [], "size": image.size}

        def close(self):
            calls.append("closed")

    def factory(*args, **kwargs):
        assert kwargs["allow_download"] is False
        assert kwargs["device"] == "cpu"
        calls.append("loaded")
        return Session()

    image = Image.new("RGB", (90, 60))
    with LiveAnalyser(
        "window-a", models=["mobilenet"], session_factory=factory, clock=lambda: 10
    ) as live:
        first = live.analyse(image, sequence=0, captured_at=10)
        second = live.analyse(image, sequence=1, captured_at=10)
        assert first["status"] == second["status"] == "completed"
        assert second["stream"]["processed_frames"] == 2
        assert calls == ["loaded"]
    assert calls == ["loaded", "closed"]
    image.close()


def test_latest_frame_replaces_stale_queue_and_closes():
    image = Image.new("RGB", (64, 32))
    buffer = LatestFrameBuffer()
    buffer.submit(image, sequence=1, captured_at=10)
    buffer.submit(image, sequence=2, captured_at=11)
    selected, seq, stamp = buffer.take()
    assert (seq, stamp, buffer.dropped) == (2, 11, 1)
    selected.close()
    buffer.close()
    with pytest.raises(ValueError):
        buffer.submit(image, sequence=3, captured_at=12)
    image.close()


def test_freshness_order_and_budget_are_enforced():
    image = Image.new("RGB", (64, 32))
    with LiveAnalyser("a", models=[], max_frames=1, clock=lambda: 10) as live:
        with pytest.raises(ValueError, match="Stale"):
            live.analyse(image, sequence=0, captured_at=1)
        live.analyse(image, sequence=0, captured_at=10)
        with pytest.raises(ValueError, match="budget"):
            live.analyse(image, sequence=1, captured_at=10)
    with LiveAnalyser("a", models=[], clock=lambda: 10) as live:
        live.analyse(image, sequence=2, captured_at=10)
        with pytest.raises(ValueError, match="Reordered"):
            live.analyse(image, sequence=1, captured_at=10)
    image.close()


def test_unavailable_model_is_not_reloaded_for_each_frame():
    calls = []

    def missing(*args, **kwargs):
        calls.append(1)
        raise OSError("fixture missing")

    image = Image.new("RGB", (64, 32))
    with LiveAnalyser(
        "a", models=["mobilenet"], session_factory=missing, clock=lambda: 10
    ) as live:
        for seq in range(3):
            assert (
                live.analyse(image, sequence=seq, captured_at=10)["status"] == "partial"
            )
    assert len(calls) == 1
    image.close()
