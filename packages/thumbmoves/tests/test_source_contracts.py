from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_windows_backend_uses_strict_cache_flags():
    source = (ROOT / "src/thumbmoves/backends/windows.py").read_text()
    assert "SIIGBF_INCACHEONLY = 0x00000010" in source
    assert "SIIGBF_THUMBNAILONLY = 0x00000008" in source
    assert "SIIGBF_THUMBNAILONLY | SIIGBF_INCACHEONLY" in source
    assert "DeleteObject" in source


def test_freedesktop_backend_uses_uri_md5():
    source = (ROOT / "src/thumbmoves/backends/freedesktop.py").read_text()
    assert 'hashlib.md5(uri.encode("utf-8")).hexdigest() + ".png"' in source
    assert '("xx-large", "x-large", "large", "normal")' in source


def test_library_has_no_pixelcue_dependency():
    for path in (ROOT / "src/thumbmoves").rglob("*.py"):
        assert "pixelcue" not in path.read_text().casefold()
