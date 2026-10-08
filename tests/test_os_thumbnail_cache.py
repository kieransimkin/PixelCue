from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_pixelcue_os_thumbnail_is_only_adapter():
    source = (ROOT / "src/pixelcue/os_thumbnail.py").read_text(encoding="utf-8")
    assert "from thumbmoves import backend_info, get_cached_thumbnail" in source
    assert "IShellItemImageFactory" not in source
    assert "hashlib.md5" not in source
    assert "QuickLook" not in source


def test_reusable_package_is_vendored_as_independent_project():
    package = ROOT / "packages/thumbmoves"
    assert (package / "pyproject.toml").is_file()
    assert (package / "README.md").is_file()
    assert (package / "src/thumbmoves/api.py").is_file()
    pyproject = (package / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "thumbmoves"' in pyproject


def test_pixelcue_depends_on_reusable_package_via_uv_local_source():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"thumbmoves>=0.1.0"' in pyproject
    assert '"thumbmoves" = { path = "packages/thumbmoves" }' in pyproject
    assert 'comtypes>=1.4' not in pyproject


def test_scanner_tries_os_thumbnail_before_original_on_cache_restore():
    source = (ROOT / "src/pixelcue/scanner.py").read_text(encoding="utf-8")
    start = source.index("def _restore_media_analysis")
    end = source.index("# ------------------------------------------------------------------\n    # Model preparation", start)
    segment = source[start:end]
    assert "thumb = os_cached_thumbnail_jpeg(path) or b""" in segment
    assert segment.index("os_cached_thumbnail_jpeg(path)") < segment.index("open_image(path)")


def test_image_and_video_processing_reuse_os_cache_when_available():
    source = (ROOT / "src/pixelcue/scanner.py").read_text(encoding="utf-8")
    assert "os_cached_thumbnail_jpeg(path) or thumbnail_jpeg(image)" in source
    assert 'thumb = os_cached_thumbnail_jpeg(path) or b""' in source
