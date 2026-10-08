from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_pixelcue_uses_thumbmoves_distribution_and_import():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    adapter = (ROOT / "src/pixelcue/os_thumbnail.py").read_text(encoding="utf-8")

    assert '"thumbmoves>=0.1.0"' in pyproject
    assert '"thumbmoves" = { path = "packages/thumbmoves" }' in pyproject
    assert "from thumbmoves import backend_info, get_cached_thumbnail" in adapter


def test_vendored_thumbmoves_is_independent_project():
    package = ROOT / "packages/thumbmoves"
    assert (package / "pyproject.toml").is_file()
    assert (package / "src/thumbmoves/api.py").is_file()
    assert 'name = "thumbmoves"' in (package / "pyproject.toml").read_text(encoding="utf-8")
