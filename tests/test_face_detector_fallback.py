from pathlib import Path

from pixelcue import face


ROOT = Path(__file__).parents[1]


def test_retinaface_is_the_default_detector():
    source = (ROOT / "src" / "pixelcue" / "face.py").read_text(encoding="utf-8")
    assert 'PIXELCUE_FACE_DETECTOR", "retinaface"' in source


def test_opencv_backend_checks_real_cascade_file():
    source = (ROOT / "src" / "pixelcue" / "face.py").read_text(encoding="utf-8")
    assert "def _opencv_cascade_path()" in source
    assert "haarcascade_frontalface_default.xml" in source
    assert "candidate.is_file()" in source


def test_opencv_can_fallback_to_retinaface_or_mtcnn():
    source = (ROOT / "src" / "pixelcue" / "face.py").read_text(encoding="utf-8")
    assert 'find_spec("retinaface")' in source
    assert 'find_spec("mtcnn")' in source
    assert "falling back to RetinaFace" in source
    assert "falling back to MTCNN" in source


def test_analyzer_uses_resolved_backend_not_global_literal():
    source = (ROOT / "src" / "pixelcue" / "face.py").read_text(encoding="utf-8")
    assert "self.detector_backend, fallback_message = resolve_detector_backend()" in source
    assert "detector_backend=backend" in source


def test_faces_extra_explicitly_includes_fallback_detectors():
    p = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"retina-face>=0.0.14"' in p
    assert '"mtcnn>=0.1.0"' in p
