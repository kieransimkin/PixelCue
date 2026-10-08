from pathlib import Path


def test_scanner_defers_model_prefetch_until_uncached_media_needs_it():
    source = (Path(__file__).parents[1] / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    run_pos = source.index("def run(self)")
    run_end = source.index("TSV_COLUMNS", run_pos)
    run_source = source[run_pos:run_end]
    assert "VLM model deferred until uncached media is found" in run_source
    assert "self._start_model_prefetch()" not in run_source


def test_media_wait_path_starts_prefetch_lazily():
    source = (Path(__file__).parents[1] / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    start = source.index("def _wait_for_model_prefetch")
    end = source.index("def _captioning_available", start)
    section = source[start:end]
    assert "self._start_model_prefetch()" in section
