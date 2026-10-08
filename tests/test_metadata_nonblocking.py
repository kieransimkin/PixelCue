from pathlib import Path

ROOT = Path(__file__).parents[1]

def test_model_info_has_explicit_timeout():
    source = (ROOT / "src" / "pixelcue" / "model.py").read_text(encoding="utf-8")
    assert "timeout=MODEL_METADATA_TIMEOUT_SECONDS" in source

def test_download_thread_starts_without_waiting_for_metadata_thread():
    source = (ROOT / "src" / "pixelcue" / "model.py").read_text(encoding="utf-8")
    meta_start = source.index("metadata_thread.start()")
    download_start = source.index("download_thread.start()", meta_start)
    wait_loop = source.index("while not download_finished.wait", download_start)
    assert meta_start < download_start < wait_loop
    # There must be no blocking metadata join between metadata start and download start.
    between = source[meta_start:download_start]
    assert ".join(" not in between
    assert "metadata_finished.wait(" not in between

def test_default_model_has_immediate_fallback_total():
    source = (ROOT / "src" / "pixelcue" / "model.py").read_text(encoding="utf-8")
    assert "MODEL_FALLBACK_TOTAL_BYTES = 17_000_000_000" in source
    assert "total = int(self.fallback_total_bytes)" in source

def test_gui_marks_fallback_denominator_approximate():
    source = (ROOT / "src" / "pixelcue" / "gui.py").read_text(encoding="utf-8")
    assert 'approx = "" if self.model_total_is_exact else "≈"' in source
