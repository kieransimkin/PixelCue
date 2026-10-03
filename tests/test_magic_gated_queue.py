from pathlib import Path

from pixelcue.archives import is_archive
from pixelcue.media import is_image, is_video


def test_mime_overrides_misleading_image_extension():
    assert not is_image(Path("$IABC.JPG"), "application/octet-stream")
    assert not is_image(Path("looks-like.jpg"), "text/plain")
    assert is_image(Path("no-extension"), "image/jpeg")


def test_mime_overrides_misleading_video_extension():
    assert not is_video(Path("fake.mp4"), "application/octet-stream")
    assert is_video(Path("no-extension"), "video/mp4")


def test_mime_overrides_misleading_archive_extension():
    assert not is_archive(Path("fake.zip"), "text/plain")
    assert is_archive(Path("no-extension"), "application/zip")


def test_extension_fallback_only_when_no_mime_exists():
    # Utility fallback remains useful in contexts where content identification
    # is genuinely unavailable, but top-level scanner records always carry a
    # magic result (or application/octet-stream), so this cannot override magic.
    assert is_image(Path("photo.jpg"), "")
    assert is_video(Path("clip.mp4"), "")
    assert is_archive(Path("set.zip"), "")
