import hashlib
import sys
from fractions import Fraction
from types import SimpleNamespace

import pytest
from PIL import Image
from fastapi.testclient import TestClient

from pixelcue.media import _frames_at_timestamps, sample_video_evidence
from pixelcue.media_analysis import analyse_media
from pixelcue.server import create_server
from pixelcue.visual_models import run_models
from pixelcue.vision_tasks import detection_result


def make_video(path):
    av = pytest.importorskip("av")
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=5)
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "yuv420p"
        for index in range(12):
            image = Image.new("RGB", (64, 48), (index * 20, 30, 200))
            frame = av.VideoFrame.from_image(image)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_real_video_sampling_hashes_and_decoded_clock(tmp_path):
    path = tmp_path / "synthetic.mp4"
    make_video(path)
    report = analyse_media(
        path, models=[], max_frames=3, models_dir=tmp_path / "models"
    )
    assert report["source"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert report["coverage"]["exhaustive"] is False
    assert report["status"] == "completed"
    assert 1 <= report["coverage"]["sampled_frames"] <= 3
    assert report["coverage"]["candidate_count"] <= 7
    for item in report["items"]:
        clock = item["frame"]
        assert clock["timestamp_seconds"] == pytest.approx(
            clock["pts"] * float(Fraction(clock["time_base"]))
            - clock["stream_origin_seconds"]
        )
        assert item["analysis"]["measurements"]
    assert any(
        item["frame"]["timestamp_seconds"] != item["frame"]["requested_seconds"]
        for item in report["items"][1:]
    )


def test_video_models_load_once_and_return_partial_on_missing_backend(tmp_path):
    path = tmp_path / "fixture.mp4"
    make_video(path)
    calls = []

    class Session:
        revision = "synthetic-interface-fixture"
        model_type = "fixture"

        def __init__(self, name, checkpoint, **options):
            calls.append(name)
            assert options["allow_download"] is False
            assert options["cache_dir"].is_relative_to(tmp_path / "models")
            if name == "layout":
                raise OSError("uncached")

        def infer(self, image):
            return {"categories": [{"label": "fixture", "label_id": 0, "score": 0.7}]}

        def close(self):
            pass

    report = analyse_media(
        path,
        models=["mobilenet", "layout"],
        max_frames=3,
        models_dir=tmp_path / "models",
        session_factory=Session,
    )
    assert calls == ["mobilenet", "layout"]
    assert report["status"] == "partial"
    assert all(
        [m["status"] for m in item["analysis"]["models"]]
        == ["completed", "unavailable"]
        for item in report["items"]
    )


def test_image_models_share_the_single_image_adapter(tmp_path, monkeypatch):
    path = tmp_path / "fixture.png"
    Image.new("RGB", (64, 48), "red").save(path)
    seen = []

    class Session:
        model = SimpleNamespace(
            config=SimpleNamespace(_commit_hash="fixture", model_type="fixture")
        )

        def infer(self, image):
            return {"categories": []}

        def close(self):
            seen.append("closed")

    def create(name, checkpoint, **options):
        seen.append(name)
        assert options["allow_download"] is False
        return Session()

    monkeypatch.setattr("pixelcue.visual_batch.create_session", create)
    result = run_models(
        Image.new("RGB", (32, 32)), ["mobilenet"], {}, models_dir=tmp_path / "models"
    )
    assert result[0]["status"] == "completed"
    assert seen == ["mobilenet", "closed"]


def test_origin_and_unknown_timestamp_are_not_fabricated(tmp_path, monkeypatch):
    import av

    stream = SimpleNamespace(start_time=100, time_base=Fraction(1, 10))
    frame = SimpleNamespace(
        time=10.4,
        pts=104,
        time_base=Fraction(1, 10),
        width=32,
        height=32,
        to_image=lambda: Image.new("RGB", (32, 32)),
    )

    class Container:
        streams = SimpleNamespace(video=[stream])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def seek(self, offset, **options):
            assert offset == 104

        def decode(self, selected):
            return iter([frame])

    monkeypatch.setattr(av, "open", lambda path: Container())
    result = _frames_at_timestamps(tmp_path / "fixture", [0.4], evidence=True)
    assert result[0][0]["timestamp_seconds"] == pytest.approx(0.4)
    result[0][1].close()
    frame.time = None
    result = _frames_at_timestamps(tmp_path / "fixture", [0.4], evidence=True)
    assert result[0][0]["timestamp_seconds"] is None
    result[0][1].close()


def test_ocr_receipt_keeps_text_confidence_and_coordinates(monkeypatch, tmp_path):
    fake = SimpleNamespace(
        Output=SimpleNamespace(DICT="dict"),
        get_tesseract_version=lambda: "fixture-engine",
        image_to_data=lambda image, **options: {
            "text": ["", "PIXELCUE"],
            "conf": [-1, 97],
            "left": [0, 3],
            "top": [0, 4],
            "width": [0, 20],
            "height": [0, 8],
        },
    )
    monkeypatch.setitem(sys.modules, "pytesseract", fake)
    receipt = run_models(
        Image.new("RGB", (32, 32)), ["ocr"], {}, models_dir=tmp_path / "models"
    )[0]
    assert receipt["status"] == "completed"
    assert receipt["result"]["words"] == [
        {"text": "PIXELCUE", "confidence": 97.0, "box_xywh": [3, 4, 20, 8]}
    ]


def test_detector_coordinates_are_clamped_sorted_and_bounded():
    class Values:
        def __init__(self, value):
            self.value = value

        def tolist(self):
            return self.value

    output = {
        "scores": Values([0.4, 0.9, 0.1]),
        "labels": Values([0, 1, 0]),
        "boxes": Values([[-1, -2, 99, 99], [1, 2, 4, 5], [0, 0, 4, 4]]),
    }
    result = detection_result(output, {0: "object", 1: "title"}, (32, 16), maximum=1)
    assert result["detections"][0]["label"] == "title"
    assert result["total_detections"] == 2
    assert result["truncated"] is True
    output["scores"] = Values([float("nan")])
    output["labels"] = Values([0])
    output["boxes"] = Values([[0, 0, 1, 1]])
    with pytest.raises(ValueError):
        detection_result(output, {}, (32, 16))


def test_real_http_media_route_and_compatibility(tmp_path):
    path = tmp_path / "fixture.png"
    Image.new("RGB", (32, 32), "blue").save(path)

    class Keywords:
        def extract_keywords(self, *args, **kwargs):
            return ["fixture"]

    client = TestClient(create_server(Keywords()).http)
    health = client.get("/health").json()
    assert {"rtdetr", "mobilenet", "layout", "ocr"} <= set(
        health["media_analysis"]["models"]
    )
    result = client.post("/v1/media-analysis", json={"path": str(path)})
    assert result.status_code == 200, result.text
    assert result.json()["status"] == "completed"
    assert result.json()["items"][0]["analysis"]["models"] == []
    assert client.post("/v1/keywords", json={"path": str(path)}).json() == ["fixture"]
    for options in [
        {"max_frames": 6},
        {"models": ["invented"]},
        {"keyword_models": ["joycaption"]},
    ]:
        assert (
            client.post(
                "/v1/media-analysis", json={"path": str(path), **options}
            ).status_code
            == 422
        )


@pytest.mark.parametrize(
    "options", [{"max_frames": 0}, {"max_frames": True}, {"max_edge": 5000}]
)
def test_bounds_before_decode(tmp_path, options):
    with pytest.raises(ValueError):
        analyse_media(tmp_path / "missing", **options)


def test_video_sampler_rejects_missing_decoder(tmp_path, monkeypatch):
    monkeypatch.setattr("pixelcue.media.video_duration_seconds", lambda path: None)
    monkeypatch.setattr(
        "pixelcue.media._frames_at_timestamps", lambda *args, **kwargs: []
    )
    result = sample_video_evidence(tmp_path / "missing")
    assert result["frames"] == []
