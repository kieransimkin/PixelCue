from pathlib import Path

ROOT=Path(__file__).parents[1]

def test_readme_primary_install_is_uv_first():
    readme=(ROOT/"README.md").read_text(encoding="utf-8")
    install=readme[readme.index("## Installation (uv)"):readme.index("## Model download and diagnostics")]
    assert "uv python install 3.13" in install
    assert "uv sync --extra faces" in install
    assert "uv run pixelcue" in install
    assert "python -m venv" not in install
    assert "pip install -e" not in install

def test_readme_documents_lockfile_and_cuda_torch_source():
    readme=(ROOT/"README.md").read_text(encoding="utf-8")
    assert "Commit `uv.lock`" in readme
    assert "PyTorch CUDA 13.2" in readme
    assert "approximately **17 GB**" in readme

def test_old_release_notes_are_marked_historical():
    readme=(ROOT/"README.md").read_text(encoding="utf-8")
    assert "## Version notes (historical)" in readme
