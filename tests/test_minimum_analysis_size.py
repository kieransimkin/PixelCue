from pathlib import Path
import ast

ROOT = Path(__file__).parents[1]


def test_threshold_is_exactly_500000_bytes():
    source = (ROOT / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    assert "MIN_MEDIA_ANALYSIS_BYTES = 500_000" in source


def test_size_filter_happens_before_media_queue_admission():
    source = (ROOT / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ScanWorker")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_classify_or_queue_file")
    segment = ast.get_source_segment(source, fn)
    assert segment.index("size_bytes < MIN_MEDIA_ANALYSIS_BYTES") < segment.index(
        "self._queue_media_job(path, rec, category)"
    )


def test_small_files_remain_catalogued_and_record_skip_reason():
    source = (ROOT / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    assert 'rec["category"] = category' in source
    assert 'rec["extra_metadata"]["joycaption_skipped"] = True' in source
    assert '"joycaption_skip_reason"' in source


def test_exactly_500000_bytes_is_eligible():
    source = (ROOT / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    assert "size_bytes < MIN_MEDIA_ANALYSIS_BYTES" in source
    assert "size_bytes <= MIN_MEDIA_ANALYSIS_BYTES" not in source
