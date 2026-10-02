from pathlib import Path
import zipfile

from pixelcue.archives import sample_archive


def test_zip_samples_only_media(tmp_path: Path):
    zpath = tmp_path / "x.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("a.jpg", b"not-a-real-image")
        z.writestr("b.txt", b"hello")
    sample = sample_archive(zpath, 5)
    try:
        assert sample.total_members == 2
        assert sample.total_media_members == 1
        assert len(sample.paths) == 1
        assert sample.paths[0].suffix == ".jpg"
    finally:
        sample.cleanup()
