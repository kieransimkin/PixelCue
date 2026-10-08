from pathlib import Path
ROOT=Path(__file__).parents[1]

def test_face_module_has_uv_dependency_repair_commands():
    source=(ROOT/"src"/"pixelcue"/"face.py").read_text()
    assert "def validate_deepface_runtime()" in source
    assert "uv sync --extra faces" in source
    assert "uv python install 3.13" in source

def test_deepface_runtime_is_checked_lazily_only_when_analysis_is_needed():
    face=(ROOT/"src"/"pixelcue"/"face.py").read_text()
    scanner=(ROOT/"src"/"pixelcue"/"scanner.py").read_text()
    assert "validate_deepface_runtime()" in face
    assert "result = analyzer.extract_embedding(path)" in scanner
    assert "cached = self.analysis_store.load(" in scanner

def test_faces_extra_contains_stable_stack():
    p=(ROOT/"pyproject.toml").read_text()
    assert "deepface>=0.0.95" in p
    assert "tensorflow==2.21.0" in p
    assert "tf-keras==2.21.0" in p
