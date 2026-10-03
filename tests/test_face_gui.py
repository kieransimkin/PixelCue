from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_gui_has_face_cluster_pane_and_click_window():
    source = (ROOT / "src" / "pixelcue" / "gui.py").read_text(encoding="utf-8")
    assert "Face identity clusters" in source
    assert "class FaceClusterButton" in source
    assert "class FaceClusterWindow" in source
    assert "face_clusters_updated.connect" in source
    assert "representative_path" in source
