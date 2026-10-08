from pathlib import Path

ROOT=Path(__file__).parents[1]

def test_gallery_cleanup_never_calls_qt_methods_on_destroyed_wrappers():
    source=(ROOT/"src"/"pixelcue"/"gui.py").read_text(encoding="utf-8")
    assert "def _remove_window_by_id" in source
    assert "id(candidate) != target_id" in source
    assert ".isHidden()" not in source
    assert "_prune_windows" not in source

def test_both_gallery_types_use_safe_tracker():
    source=(ROOT/"src"/"pixelcue"/"gui.py").read_text(encoding="utf-8")
    assert "self._track_top_level_window(self.tag_windows, win)" in source
    assert "self._track_top_level_window(self.face_cluster_windows, win)" in source
    assert "win.destroyed.connect(" in source
