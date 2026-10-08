from pathlib import Path
import ast

ROOT = Path(__file__).parents[1]


def test_sorted_image_insertion_algorithm_is_literal_descending_walk():
    source = (ROOT / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ScanWorker")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_insert_image_job_sorted")
    segment = ast.get_source_segment(source, fn)

    # Exact requested semantics: start at the top, walk until a smaller item,
    # then insert at that position. Strict > keeps equal-sized items stable.
    assert "insert_at = 0" in segment
    assert "while insert_at < len(self.image_jobs):" in segment
    assert "queued_size" in segment
    assert "if size_bytes > queued_size:" in segment
    assert "insert_at += 1" in segment
    assert "self.image_jobs.insert(" in segment
