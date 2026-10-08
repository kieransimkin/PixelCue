from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_scanner_does_not_disable_future_jobs_after_per_file_failure():
    source = (ROOT / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    # The old global kill-switch string should no longer exist.
    assert "Tagging skipped because JoyCaption failed earlier in this scan" not in source
    assert "VLM tagging failed for this file; PixelCue will continue with the next queued media item." in source


def test_per_file_processors_do_not_set_global_captioning_failure_state():
    source = (ROOT / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    # Only model setup failures may still be treated globally.
    occurrences = source.count("captioning_disabled_reason")
    assert occurrences == 0


def test_model_setup_failure_path_still_exists():
    source = (ROOT / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    assert "self.model_download_error = str(e)" in source
    assert "Tagging skipped because VLM model setup failed:" in source
