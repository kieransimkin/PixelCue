from pathlib import Path
import ast

ROOT = Path(__file__).parents[1]


def test_media_jobs_use_heapq_priority_queue():
    source = (ROOT / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    assert "import heapq" in source
    assert "heapq.heappush(" in source
    assert "heapq.heappop(" in source


def test_media_job_priority_uses_negative_size_bytes():
    source = (ROOT / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ScanWorker")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_media_job_priority")
    seg = ast.get_source_segment(source, fn)
    assert '-int(rec.get("size_bytes") or 0)' in seg


def test_media_queue_preserves_discovery_order_on_size_ties():
    source = (ROOT / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    assert "self.media_job_sequence = 0" in source
    assert "seq = self.media_job_sequence" in source
    assert "self.media_job_sequence += 1" in source


def test_readme_mentions_largest_files_first():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "largest files first" in readme
