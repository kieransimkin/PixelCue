from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_scanner_looks_for_cached_media_before_queueing_or_size_filter():
    source = (ROOT / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    start = source.index("def _classify_or_queue_file")
    end = source.index("def _walk_filesystem", start)
    segment = source[start:end]
    cache_pos = segment.index("self._restore_media_analysis(path, rec, category)")
    size_pos = segment.index("size_bytes < MIN_MEDIA_ANALYSIS_BYTES")
    queue_pos = segment.index("self._queue_media_job(path, rec, category)")
    assert cache_pos < size_pos < queue_pos


def test_media_results_are_saved_and_restored_per_vlm_profile():
    source = (ROOT / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    assert 'kind="media_tags"' in source
    assert 'f"{MEDIA_CACHE_VERSION}:{self.model_profile_id}"' in source
    assert "self._persist_media_analysis(path, rec)" in source
    assert "self.tagged.emit(str(path), tags, thumb)" in source


def test_face_embeddings_are_persisted_and_restored():
    source = (ROOT / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    assert 'kind="face_embedding"' in source
    assert '"embedding": result.embedding' in source
    assert "self.face_results[str(path)] = result" in source
    assert '"status": "face_not_confirmed"' in source


def test_expensive_cluster_attributes_use_persistent_per_image_cache():
    source = (ROOT / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    assert 'kind="face_attributes"' in source
    assert "face-attributes-restore" in source
    load_pos = source.index('kind="face_attributes"')
    analyze_pos = source.index("attrs = analyzer.analyze_attributes", load_pos)
    assert load_pos < analyze_pos


def test_model_download_is_deferred_when_cache_can_satisfy_scan():
    source = (ROOT / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    run = source[source.index("def run(self)"):]
    assert 'self.model_status.emit("VLM model deferred until uncached media is found")' in run
    assert "self._start_model_prefetch()" not in run.split("def export_tsv", 1)[0]
    wait_start = source.index("def _wait_for_model_prefetch")
    wait_end = source.index("def _captioning_available", wait_start)
    assert "self._start_model_prefetch()" in source[wait_start:wait_end]
