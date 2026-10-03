from pathlib import Path

def test_model_uses_exact_hf_file_metadata():
    source = (Path(__file__).parents[1] / "src" / "pixelcue" / "model.py").read_text(encoding="utf-8")
    assert "files_metadata=True" in source
    assert "total = sum(size for _, size in files)" in source

def test_model_counts_partial_download_bytes():
    source = (Path(__file__).parents[1] / "src" / "pixelcue" / "model.py").read_text(encoding="utf-8")
    assert 'rglob("*.incomplete")' in source
    assert "self._progress(done, total)" in source

def test_gui_displays_exact_bytes_and_percent():
    source = (Path(__file__).parents[1] / "src" / "pixelcue" / "gui.py").read_text(encoding="utf-8")
    assert "{done:,} / {total:,} bytes" in source
    assert "{percentage:.1f}%" in source
