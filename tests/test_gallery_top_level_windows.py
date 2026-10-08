from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_gallery_windows_are_true_top_level_windows():
    source = (ROOT / "src" / "pixelcue" / "gui.py").read_text(encoding="utf-8")
    assert "super().__init__(None, Qt.Window)" in source
    assert "TagItemsWindow(" in source
    assert "FaceClusterWindow(cluster, self.thumbs, self.path_tags, None)" in source


def test_each_gallery_is_retained_independently_and_safely():
    source = (ROOT / "src" / "pixelcue" / "gui.py").read_text(encoding="utf-8")
    assert "self._track_top_level_window(self.tag_windows, win)" in source
    assert "self._track_top_level_window(self.face_cluster_windows, win)" in source
    assert "collection.append(win)" in source
    assert "win.destroyed.connect(" in source
