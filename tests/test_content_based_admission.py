from pathlib import Path

from pixelcue.archives import is_archive
from pixelcue.media import is_image, is_video


def test_extension_cannot_override_confident_non_media_mime():
    assert not is_image(Path("$I123.JPG"), "application/octet-stream")
    assert not is_video(Path("fake.mp4"), "application/octet-stream")
    assert not is_archive(Path("fake.zip"), "application/octet-stream")


def test_content_mime_can_identify_media_with_no_extension():
    assert is_image(Path("nameless"), "image/jpeg")
    assert is_video(Path("nameless"), "video/mp4")
    assert is_archive(Path("nameless"), "application/zip")


def test_suffix_is_only_a_no_mime_fallback():
    assert is_image(Path("x.jpg"), "")
    assert is_video(Path("x.mp4"), "")
    assert is_archive(Path("x.zip"), "")


def test_scanner_classification_uses_recorded_mime_before_queueing():
    source = (
        Path(__file__).parents[1]
        / "src"
        / "pixelcue"
        / "scanner.py"
    ).read_text(encoding="utf-8")
    start = source.index("def _classify_or_queue_file")
    end = source.index("def _walk_filesystem", start)
    segment = source[start:end]

    assert 'mime = str(rec.get("mime_type"' in segment
    assert "is_image(path, mime)" in segment
    assert "is_video(path, mime)" in segment
    assert "is_archive(path, mime)" in segment
