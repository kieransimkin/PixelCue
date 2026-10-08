from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_platform_code_no_longer_lives_in_pixelcue_package():
    adapter = (ROOT / "src/pixelcue/os_thumbnail.py").read_text(encoding="utf-8")
    assert "ctypes.windll" not in adapter
    assert "XDG_CACHE_HOME" not in adapter
    assert "comtypes" not in adapter


def test_library_project_can_stand_alone():
    root = ROOT / "packages/os-thumbnail-cache"
    for path in (root / "src/os_thumbnail_cache").rglob("*.py"):
        assert "from pixelcue" not in path.read_text(encoding="utf-8")
