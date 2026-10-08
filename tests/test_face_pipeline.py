from pathlib import Path
import ast

from pixelcue.face import FaceResult, cluster_embeddings, cosine_distance


ROOT = Path(__file__).parents[1]


def test_faceidentity_queues_deepface_only_for_images():
    source = (ROOT / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ScanWorker")
    image_fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_process_image")
    video_fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_process_video")
    image_segment = ast.get_source_segment(source, image_fn)
    video_segment = ast.get_source_segment(source, video_fn)
    assert "faceidentity" in image_segment.casefold()
    assert "_restore_or_queue_face_analysis(path, rec)" in image_segment
    assert "_restore_or_queue_face_analysis(path, rec)" not in video_segment


def test_face_worker_is_independent_thread():
    source = (ROOT / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    assert 'name="pixelcue-deepface"' in source
    assert "deepface-worker" in source
    assert "self._run_logged_worker" in source
    assert "CLUSTER_INTERVAL_SECONDS" in source


def test_every_embedding_gets_a_cluster_including_singletons():
    a = FaceResult("a.jpg", [1.0, 0.0], {})
    b = FaceResult("b.jpg", [0.999, 0.01], {})
    c = FaceResult("c.jpg", [0.0, 1.0], {})
    clusters = cluster_embeddings([a, b, c], threshold=0.1)
    flattened = [p for cluster in clusters for p in cluster["paths"]]
    assert sorted(flattened) == ["a.jpg", "b.jpg", "c.jpg"]
    assert any(cluster["paths"] == ["c.jpg"] for cluster in clusters)


def test_representative_is_nearest_centroid():
    a = FaceResult("a.jpg", [1.0, 0.0], {})
    b = FaceResult("b.jpg", [0.9, 0.1], {})
    c = FaceResult("c.jpg", [0.8, 0.2], {})
    cluster = cluster_embeddings([a, b, c], threshold=1.0)[0]
    centroid = cluster["centroid"]
    rep = cluster["representative_path"]
    distances = {x.path: cosine_distance(x.embedding, centroid) for x in [a, b, c]}
    assert distances[rep] == min(distances.values())


def test_face_metadata_extracts_embedding_during_scan():
    scanner = (ROOT / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    assert 'rec["extra_metadata"]["face_embedding"] = result.embedding' in scanner
    assert "result = analyzer.extract_embedding(path)" in scanner


def test_celebrity_lookup_is_optional_database_backed():
    source = (ROOT / "src" / "pixelcue" / "face.py").read_text(encoding="utf-8")
    assert "PIXELCUE_CELEBRITY_DB" in source
    assert "similarity_search=True" in source
    assert "no celebrity DB configured" in source


def test_deepface_is_kept_off_gpu_by_default():
    source = (ROOT / "src" / "pixelcue" / "face.py").read_text(encoding="utf-8")
    assert 'tf.config.set_visible_devices([], "GPU")' in source
