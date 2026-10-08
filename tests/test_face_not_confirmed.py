from pathlib import Path
import ast

from pixelcue.face import FaceNotConfirmed, detector_fallback_chain

ROOT = Path(__file__).parents[1]


def test_face_not_confirmed_is_distinct_nonfatal_condition():
    assert issubclass(FaceNotConfirmed, RuntimeError)


def test_detector_fallback_chain_is_unique_and_includes_modern_detectors():
    chain = detector_fallback_chain("retinaface")
    assert len(chain) == len(set(chain))
    assert "retinaface" in chain
    assert "mtcnn" in chain


def test_embedding_extractor_tries_multiple_detectors_before_rejecting_face():
    source = (ROOT / "src" / "pixelcue" / "face.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DeepFaceAnalyzer")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "extract_embedding")
    segment = ast.get_source_segment(source, fn)
    assert "for backend in detector_fallback_chain" in segment
    assert "raise FaceNotConfirmed(" in segment


def test_scanner_treats_embedding_face_rejection_as_metadata_not_processing_error():
    source = (ROOT / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ScanWorker")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_face_worker")
    segment = ast.get_source_segment(source, fn)
    assert '"status": "face_not_confirmed"' in segment
    start = segment.index("except FaceNotConfirmed as e:")
    end = segment.index("except DeepFaceUnavailable as e:", start)
    handler = segment[start:end]
    assert "_record_error(" not in handler
    assert "processing_error.emit" not in handler
