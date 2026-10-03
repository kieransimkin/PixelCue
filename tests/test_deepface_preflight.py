from pathlib import Path
ROOT=Path(__file__).parents[1]

def test_face_module_has_uv_dependency_repair_commands():
    source=(ROOT/"src"/"pixelcue"/"face.py").read_text()
    assert "def validate_deepface_runtime()" in source
    assert "uv sync --extra faces" in source
    assert "uv python install 3.13" in source

def test_scanner_disables_broken_face_runtime_once():
    source=(ROOT/"src"/"pixelcue"/"scanner.py").read_text()
    assert "self.face_disabled_reason" in source
    assert "validate_deepface_runtime()" in source
    assert "self.face_disabled_reason = face_reason" in source

def test_faces_extra_contains_stable_stack():
    p=(ROOT/"pyproject.toml").read_text()
    assert "deepface>=0.0.95" in p
    assert "tensorflow==2.21.0" in p
    assert "tf-keras==2.21.0" in p
