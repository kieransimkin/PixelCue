from pathlib import Path

ROOT=Path(__file__).parents[1]


def test_verbose_logging_is_enabled_by_default_and_flushes_stdout():
    source=(ROOT/'src'/'pixelcue'/'logging_utils.py').read_text(encoding='utf-8')
    assert 'PIXELCUE_VERBOSE", "1"' in source
    assert 'file=sys.stdout, flush=True' in source
    assert 'pid=' in source
    assert 'thread=' in source
    assert 'worker=' in source
    assert 'event=' in source


def test_all_major_workers_use_logged_lifecycle_wrapper():
    source=(ROOT/'src'/'pixelcue'/'scanner.py').read_text(encoding='utf-8')
    for worker in ['vlm-model-prefetch','video-sampler','vlm-media-worker','deepface-worker']:
        assert worker in source
    assert 'def _run_logged_worker' in source
    assert 'log_event(worker_name, "start"' in source
    assert 'log_event(worker_name, "finish"' in source


def test_filesystem_and_cluster_process_log_start_and_finish():
    scanner=(ROOT/'src'/'pixelcue'/'scanner.py').read_text(encoding='utf-8')
    cluster=(ROOT/'src'/'pixelcue'/'cluster_process.py').read_text(encoding='utf-8')
    assert '"filesystem-scan", "start"' in scanner
    assert '"filesystem-scan", "finish"' in scanner
    assert 'log_event(worker, "start"' in cluster
    assert 'log_event(' in cluster and '"finish"' in cluster
