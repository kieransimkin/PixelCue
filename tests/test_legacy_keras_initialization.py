from pathlib import Path
import ast

ROOT = Path(__file__).parents[1]


def test_legacy_keras_flag_is_set_before_model_constants():
    source = (ROOT / "src" / "pixelcue" / "face.py").read_text(encoding="utf-8")
    flag_pos = source.index('os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")')
    model_pos = source.index("DEEPFACE_MODEL =")
    assert flag_pos < model_pos


def test_preflight_does_not_import_tensorflow():
    source = (ROOT / "src" / "pixelcue" / "face.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "validate_deepface_runtime"
    )
    segment = ast.get_source_segment(source, fn)

    assert 'importlib.metadata.version("tensorflow")' in segment
    assert "import tensorflow as tf" not in segment


def test_deepface_import_occurs_after_legacy_flag_reassertion():
    source = (ROOT / "src" / "pixelcue" / "face.py").read_text(encoding="utf-8")
    start = source.index("def validate_deepface_runtime")
    end = source.index("@dataclass", start)
    segment = source[start:end]

    assert segment.index('os.environ["TF_USE_LEGACY_KERAS"] = "1"') < segment.index(
        "from deepface import DeepFace"
    )


def test_analyzer_reasserts_legacy_keras_before_deepface_import():
    source = (ROOT / "src" / "pixelcue" / "face.py").read_text(encoding="utf-8")
    start = source.index("def extract_embedding(self, path: str)")
    end = source.index("def analyze_attributes", start)
    segment = source[start:end]

    assert segment.index('os.environ["TF_USE_LEGACY_KERAS"] = "1"') < segment.index(
        "from deepface import DeepFace"
    )
