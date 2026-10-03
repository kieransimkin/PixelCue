from pathlib import Path

def test_media_methods_are_inside_scanworker():
    source = (Path(__file__).parents[1] / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    assert "\n    def _start_model_prefetch(" in source
    assert "\n    def _process_image(" in source
    assert "\n    def _process_video(" in source
    assert "\n    def _process_archive(" in source
    assert "\ndef _process_image(" not in source

def test_scanner_exposes_large_byte_progress_signal():
    source = (Path(__file__).parents[1] / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    assert "model_progress = Signal(object, object)" in source
    assert "progress_callback=self.model_progress.emit" in source
