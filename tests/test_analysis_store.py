from pathlib import Path
import time

from pixelcue.analysis_store import AnalysisStore


def test_store_roundtrip_and_file_change_invalidation(tmp_path: Path):
    media = tmp_path / "image.jpg"
    media.write_bytes(b"abc")
    store = AnalysisStore(tmp_path / "cache.sqlite3")
    try:
        assert store.load(media, kind="media_tags", engine="demo") is None
        assert store.save(
            media,
            kind="media_tags",
            engine="demo",
            payload={"tags": ["rock", "clear water"]},
        )
        assert store.load(media, kind="media_tags", engine="demo") == {
            "tags": ["rock", "clear water"]
        }

        # A changed file is a different immutable signature and must not reuse
        # analysis created for the older bytes.
        time.sleep(0.002)
        media.write_bytes(b"abcd")
        assert store.load(media, kind="media_tags", engine="demo") is None
    finally:
        store.close()


def test_store_is_stage_and_engine_specific(tmp_path: Path):
    media = tmp_path / "x.bin"
    media.write_bytes(b"payload")
    store = AnalysisStore(tmp_path / "cache.sqlite3")
    try:
        store.save(media, kind="media_tags", engine="joycaption", payload={"tags": ["a"]})
        assert store.load(media, kind="media_tags", engine="joycaption") == {"tags": ["a"]}
        assert store.load(media, kind="media_tags", engine="smolvlm") is None
        assert store.load(media, kind="face_embedding", engine="joycaption") is None
    finally:
        store.close()
