from pathlib import Path

from pixelcue.metadata import stat_record


class FakeDetector:
    def identify_with_source(self, path: Path):
        return "application/octet-stream", "Windows Recycle Bin metadata", "libmagic"


def test_stat_record_records_magic_source_and_does_not_claim_image(tmp_path: Path):
    p = tmp_path / "$IFAKE.JPG"
    p.write_bytes(b"not a jpeg")
    rec = stat_record(p, FakeDetector())
    assert rec["mime_type"] == "application/octet-stream"
    assert rec["extra_metadata"]["type_detection"] == "libmagic"
    assert rec["exif"] == ""
