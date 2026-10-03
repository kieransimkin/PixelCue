from pathlib import Path
import ast


ROOT = Path(__file__).parents[1]


def test_walk_filesystem_only_queues_media_not_processes_it():
    source = (ROOT / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ScanWorker")
    walk = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_walk_filesystem")
    segment = ast.get_source_segment(source, walk)
    assert "_classify_or_queue_file" in segment
    assert "_process_image(" not in segment
    assert "_process_video(" not in segment
    assert "_process_archive(" not in segment
    assert "_wait_for_model_prefetch(" not in segment


def test_media_worker_is_separate_thread():
    source = (ROOT / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    assert 'name="pixelcue-media-tagger"' in source
    assert "target=self._media_worker" in source


def test_run_starts_model_and_media_workers_before_walking():
    source = (ROOT / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    run_pos = source.index("def run(self)")
    model_pos = source.index("self._start_model_prefetch()", run_pos)
    media_pos = source.index("self._start_media_worker()", run_pos)
    walk_pos = source.index("self._walk_filesystem()", run_pos)
    assert model_pos < walk_pos
    assert media_pos < walk_pos


def test_gui_exposes_discovery_and_media_queue_counts():
    source = (ROOT / "src" / "pixelcue" / "gui.py").read_text(encoding="utf-8")
    assert "files discovered" in source
    assert "Media: {waiting} queued / {completed} processed" in source


def test_media_worker_has_separate_magic_detector():
    source = (ROOT / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    assert "self.media_detector = MagicDetector()" in source
    assert "detector = self.media_detector or self.detector" in source
