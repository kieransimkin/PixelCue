from io import BytesIO
from pathlib import Path
import zipfile

from PIL import Image

from pixelcue.archives import is_archive, sample_archive


def jpeg_bytes() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(buf, format="JPEG")
    return buf.getvalue()


def test_zip_samples_media_by_content_not_suffix(tmp_path: Path):
    zpath = tmp_path / "x.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        # Fake JPG must be rejected despite its name.
        z.writestr("fake.jpg", b"this is ordinary text, not a jpeg")
        # Real JPEG must be accepted despite having no image extension.
        z.writestr("mystery.bin", jpeg_bytes())
        z.writestr("note.txt", b"hello")

    sample = sample_archive(zpath, 5, mime="application/zip")
    try:
        assert sample.total_members == 3
        assert sample.total_media_members == 1
        assert sample.member_names == ["mystery.bin"]
        assert len(sample.paths) == 1
    finally:
        sample.cleanup()


def test_archive_mime_wins_over_suffix(tmp_path: Path):
    disguised_zip = tmp_path / "archive.notzip"
    with zipfile.ZipFile(disguised_zip, "w") as z:
        z.writestr("photo", jpeg_bytes())

    assert is_archive(disguised_zip, "application/zip")
    assert not is_archive(tmp_path / "looks.zip", "application/octet-stream")

    sample = sample_archive(
        disguised_zip,
        5,
        mime="application/zip",
    )
    try:
        assert sample.total_media_members == 1
    finally:
        sample.cleanup()
