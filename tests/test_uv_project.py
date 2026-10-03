from pathlib import Path
ROOT=Path(__file__).parents[1]

def test_python_pin_and_project_constraint():
    assert (ROOT/".python-version").read_text().strip()=="3.13"
    assert 'requires-python = ">=3.13,<3.14"' in (ROOT/"pyproject.toml").read_text()

def test_uv_commands_are_project_standard():
    source=(ROOT/"src"/"pixelcue"/"face.py").read_text()
    assert 'return "uv sync --extra faces"' in source
    assert "uv python install 3.13" in source
    readme=(ROOT/"README.md").read_text()
    assert "uv run pixelcue" in readme

def test_venv_is_ignored_but_lockfile_is_not_ignored():
    lines={x.strip() for x in (ROOT/".gitignore").read_text().splitlines()}
    assert ".venv/" in lines
    assert "uv.lock" not in lines
