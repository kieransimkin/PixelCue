import hashlib
from pathlib import Path
from PIL import Image, PngImagePlugin

from thumbmoves.api import get_cached_thumbnail


def test_freedesktop_cache_lookup(tmp_path, monkeypatch):
    source = tmp_path / "photo.jpg"
    source.write_bytes(b"not-opened-by-library")
    uri = source.resolve().as_uri()
    digest = hashlib.md5(uri.encode("utf-8")).hexdigest() + ".png"

    cache = tmp_path / "cache" / "thumbnails" / "large"
    cache.mkdir(parents=True)
    info = PngImagePlugin.PngInfo()
    info.add_text("Thumb::URI", uri)
    info.add_text("Thumb::MTime", str(int(source.stat().st_mtime)))
    Image.new("RGB", (640, 480), "white").save(cache / digest, pnginfo=info)

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr("sys.platform", "linux")
    result = get_cached_thumbnail(source, (160, 120))
    assert result is not None
    assert result.backend == "freedesktop-xdg"
    assert result.size == (160, 120)
    assert result.data.startswith(b"\xff\xd8")
