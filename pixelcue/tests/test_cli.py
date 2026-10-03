from pathlib import Path

from pixelcue.cli import startup_path_from_argv


def test_startup_path_accepts_existing_path(tmp_path: Path):
    p = tmp_path / "media"
    p.mkdir()
    result = startup_path_from_argv(["pixelcue", str(p)])
    assert result is not None
    assert Path(result) == p.absolute()


def test_startup_path_rejects_missing_path(tmp_path: Path):
    p = tmp_path / "missing"
    assert startup_path_from_argv(["pixelcue", str(p)]) is None


def test_startup_path_accepts_file(tmp_path: Path):
    p = tmp_path / "image.jpg"
    p.write_bytes(b"x")
    result = startup_path_from_argv(["pixelcue", str(p)])
    assert result is not None
    assert Path(result) == p.absolute()
