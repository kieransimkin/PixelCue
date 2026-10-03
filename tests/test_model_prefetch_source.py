from pathlib import Path


def test_scanner_starts_model_prefetch_before_filesystem_walk():
    source = (Path(__file__).parents[1] / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    run_pos = source.index("def run(self)")
    prefetch_pos = source.index("self._start_model_prefetch()", run_pos)
    media_pos = source.index("self._start_media_worker()", run_pos)
    walk_pos = source.index("self._walk_filesystem()", run_pos)
    assert prefetch_pos < walk_pos
    assert media_pos < walk_pos


def test_filesystem_walk_does_not_wait_for_model():
    source = (Path(__file__).parents[1] / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    start = source.index("def _walk_filesystem(self)")
    end = source.index("def run(self)", start)
    walk_source = source[start:end]
    assert "_wait_for_model_prefetch(" not in walk_source
    assert "_process_image(" not in walk_source
    assert "_process_video(" not in walk_source
    assert "_process_archive(" not in walk_source
