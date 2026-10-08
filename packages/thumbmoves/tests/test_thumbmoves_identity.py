from pathlib import Path
import thumbmoves

ROOT = Path(__file__).parents[1]


def test_public_package_name_and_version():
    assert thumbmoves.__version__ == "0.1.1"


def test_project_identity_is_thumbmoves():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "thumbmoves"' in pyproject
    assert 'thumbmoves = "thumbmoves.cli:main"' in pyproject


def test_no_old_import_namespace_remains():
    assert (ROOT / "src/thumbmoves").is_dir()
    for path in (ROOT / "src/thumbmoves").rglob("*.py"):
        assert "os_thumbnail_cache" not in path.read_text(encoding="utf-8")
