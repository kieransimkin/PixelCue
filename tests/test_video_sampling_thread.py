from pathlib import Path
import ast

ROOT = Path(__file__).parents[1]


def test_video_sampling_has_dedicated_worker_thread():
    source = (ROOT / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    assert 'name="pixelcue-video-sampler"' in source
    assert "target=self._video_sample_worker" in source


def test_filesystem_classifier_queues_video_sampling_not_media_directly():
    source = (ROOT / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ScanWorker")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_classify_or_queue_file")
    segment = ast.get_source_segment(source, fn)
    assert 'if category == "video":' in segment
    assert "self._queue_video_sampling(path, rec)" in segment


def test_joycaption_video_processor_does_not_decode_video():
    source = (ROOT / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ScanWorker")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_process_video")
    segment = ast.get_source_segment(source, fn)
    assert 'rec.pop("_pixelcue_sampled_frames", [])' in segment
    assert "sample_video_frames(" not in segment
    assert "video_duration_seconds(" not in segment


def test_media_worker_waits_for_sampler_before_terminating():
    source = (ROOT / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    assert "and self.video_sampling_done" in source
