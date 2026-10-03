from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_gallery_windows_are_true_top_level_windows():
    source = (ROOT / "src" / "pixelcue" / "gui.py").read_text(encoding="utf-8")
    assert "super().__init__(None, Qt.Window)" in source
    assert "TagItemsWindow(tag, list(self.tag_to_paths.get(tag, [])), self.thumbs, None)" in source
    assert "FaceClusterWindow(cluster, self.thumbs, None)" in source


def test_each_gallery_is_retained_independently():
    source = (ROOT / "src" / "pixelcue" / "gui.py").read_text(encoding="utf-8")
    assert "self.tag_windows.append(win)" in source
    assert "self.face_cluster_windows.append(win)" in source
