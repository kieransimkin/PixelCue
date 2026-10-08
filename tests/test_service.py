from pathlib import Path

import pytest
from PIL import Image

from pixelcue.service import KeywordExtractionService, UnsupportedMediaError


class FakeDetector:
    def identify(self, path: Path) -> tuple[str, str]:
        if path.suffix == ".png":
            return "image/png", "PNG image"
        return "text/plain", "plain text"


class FakeTagger:
    def tags_for_image(self, image: Image.Image) -> list[str]:
        assert image.size == (16, 12)
        return ["red", " abstract art ", "RED"]


def fake_tagger_factory(model: str, **callbacks):
    assert model == "smolvlm-256m"
    assert callable(callbacks["status_callback"])
    assert callable(callbacks["progress_callback"])
    return FakeTagger()


def test_extracts_deduplicated_keywords_from_one_image(tmp_path):
    source = tmp_path / "fixture.png"
    Image.new("RGB", (16, 12), "red").save(source)
    events = []
    service = KeywordExtractionService(
        tagger_factory=fake_tagger_factory,
        detector_factory=FakeDetector,
    )

    result = service.extract_keywords(
        source,
        model="smolvlm-256m",
        progress=events.append,
    )

    assert result == ["red", "abstract art"]
    assert events[0]["stage"] == "inspection"
    assert events[-1] == {
        "stage": "result",
        "message": "Keywords ready.",
        "completed": 2,
        "total": 2,
    }


def test_rejects_directory(tmp_path):
    service = KeywordExtractionService(
        tagger_factory=fake_tagger_factory,
        detector_factory=FakeDetector,
    )
    with pytest.raises(IsADirectoryError, match="Expected one file"):
        service.extract_keywords(tmp_path, model="smolvlm-256m")


def test_rejects_unverified_media(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text("not visual media", encoding="utf-8")
    service = KeywordExtractionService(
        tagger_factory=fake_tagger_factory,
        detector_factory=FakeDetector,
    )
    with pytest.raises(UnsupportedMediaError, match="Unsupported or unverified"):
        service.extract_keywords(source, model="smolvlm-256m")
