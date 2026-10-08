from pathlib import Path
import ast

ROOT = Path(__file__).parents[1]


def test_sampled_videos_have_separate_low_priority_media_heap():
    source = (ROOT / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    assert "self.video_media_jobs:" in source
    assert 'elif category == "video":' in source
    assert "heapq.heappush(" in source


def test_media_worker_checks_images_before_archives_and_videos():
    source = (ROOT / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ScanWorker")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_media_worker")
    segment = ast.get_source_segment(source, fn)

    image = segment.index("if self.image_jobs:")
    archive = segment.index("if self.archive_media_jobs:")
    video = segment.index("if self.filesystem_discovery_done and self.video_media_jobs:")
    assert image < archive < video


def test_video_joycaption_waits_until_filesystem_discovery_complete():
    source = (ROOT / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    assert "if self.filesystem_discovery_done and self.video_media_jobs:" in source


def test_video_sampler_opens_container_once_for_all_candidate_seeks():
    source = (ROOT / "src" / "pixelcue" / "media.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_frames_at_timestamps")
    segment = ast.get_source_segment(source, fn)
    assert segment.count("av.open(") == 1
    assert "for seconds in timestamps:" in segment


def test_video_sampler_uses_at_most_seven_candidates_and_five_results():
    source = (ROOT / "src" / "pixelcue" / "media.py").read_text(encoding="utf-8")
    assert "candidate_count = min(7" in source
    assert "min(int(max_frames), 5)" in source
