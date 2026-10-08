from pathlib import Path

from pixelcue.face import (
    CLUSTER_COSINE_DISTANCE,
    CLUSTER_MERGE_MARGIN,
    FaceResult,
    cluster_embeddings,
)

ROOT = Path(__file__).parents[1]


def test_default_cluster_distance_is_less_strict():
    assert CLUSTER_COSINE_DISTANCE == 0.50
    assert CLUSTER_MERGE_MARGIN == 0.06


def test_second_pass_merges_centroid_near_fragments():
    a = FaceResult("a.jpg", [1.0, 0.0], {})
    b = FaceResult("b.jpg", [0.92, 0.39], {})

    strict = cluster_embeddings(
        [a, b],
        threshold=0.01,
        merge_threshold=0.02,
    )
    tolerant = cluster_embeddings(
        [a, b],
        threshold=0.01,
        merge_threshold=0.10,
    )

    assert len(strict) == 2
    assert len(tolerant) == 1
    assert sorted(tolerant[0]["paths"]) == ["a.jpg", "b.jpg"]


def test_explicit_strict_threshold_only_gets_small_merge_margin():
    a = FaceResult("a.jpg", [1.0, 0.0], {})
    b = FaceResult("b.jpg", [0.0, 1.0], {})
    clusters = cluster_embeddings([a, b], threshold=0.10)
    assert len(clusters) == 2


def test_readme_documents_cached_embedding_reclustering():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "automatically reclusters" in readme
    assert "PIXELCUE_FACE_CLUSTER_MERGE_MARGIN" in readme
