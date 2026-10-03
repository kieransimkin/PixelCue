from pathlib import Path
ROOT=Path(__file__).parents[1]

def test_faces_extra_pins_stable_stack():
    p=(ROOT/"pyproject.toml").read_text()
    assert "tensorflow==2.21.0" in p
    assert "tf-keras==2.21.0" in p
    assert 'requires-python = ">=3.13,<3.14"' in p

def test_runtime_requires_python313_and_stable_tensorflow():
    source=(ROOT/"src"/"pixelcue"/"face.py").read_text()
    assert "sys.version_info[:2] != (3, 13)" in source
    assert 'tf_version != "2.21.0"' in source
    assert 'tf_keras_version != "2.21.0"' in source

def test_scanner_uses_face_runtime_preflight():
    source=(ROOT/"src"/"pixelcue"/"scanner.py").read_text()
    assert "validate_deepface_runtime()" in source
