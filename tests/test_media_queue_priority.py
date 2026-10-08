from pathlib import Path
import ast

ROOT = Path(__file__).parents[1]


def test_images_use_explicit_descending_list_not_heap():
    source = (ROOT / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    assert "self.image_jobs: list[tuple[int, int, MediaJob]] = []" in source
    assert "self.image_jobs.insert(" in source
    assert "self.image_jobs.pop(0)" in source


def test_image_insertion_walks_from_top_until_smaller_item():
    source = (ROOT / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ScanWorker")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_insert_image_job_sorted")
    segment = ast.get_source_segment(source, fn)

    assert "insert_at = 0" in segment
    assert "while insert_at < len(self.image_jobs):" in segment
    assert "if size_bytes > queued_size:" in segment
    assert "insert_at += 1" in segment
    assert "self.image_jobs.insert(" in segment


def test_equal_sizes_preserve_discovery_order():
    source = (ROOT / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    assert "if size_bytes > queued_size:" in source
    assert "if size_bytes >= queued_size:" not in source


def test_archives_and_videos_are_separate_from_image_list():
    source = (ROOT / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    assert "self.archive_media_jobs:" in source
    assert "self.video_media_jobs:" in source
    assert 'elif category == "archive":' in source
    assert 'elif category == "video":' in source


def test_readme_mentions_largest_discovered_image_is_next():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "largest currently" in readme.casefold()
