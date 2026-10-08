from pathlib import Path

from pixelcue.face import (
    FaceResult,
    aggregate_cluster_attributes,
    central_cluster_members,
    cluster_attributes_confident,
)

ROOT = Path(__file__).parents[1]


def test_central_members_are_nearest_centroid_and_capped_at_three():
    members = [
        FaceResult("a", [1.0, 0.0], {}),
        FaceResult("b", [0.99, 0.01], {}),
        FaceResult("c", [0.9, 0.1], {}),
        FaceResult("d", [0.0, 1.0], {}),
    ]
    cluster = {
        "centroid": [1.0, 0.0],
        "members": members,
    }
    chosen = central_cluster_members(cluster, 3)
    assert len(chosen) == 3
    assert chosen[0].path == "a"
    assert "d" not in [x.path for x in chosen]


def _sample(age, gender="Woman", race="white", gscore=95, rscore=80):
    return {
        "age": age,
        "dominant_gender": gender,
        "gender": {gender: gscore},
        "dominant_ethnicity": race,
        "dominant_race": race,
        "ethnicity": {race: rscore},
        "race": {race: rscore},
        "dominant_emotion": "happy",
        "emotion": {"happy": 80},
        "celebrity_lookalike": "unavailable (no celebrity DB configured)",
    }


def test_two_confident_agreeing_samples_allow_early_stop():
    assert cluster_attributes_confident([
        _sample(31),
        _sample(34),
    ])


def test_age_or_demographic_disagreement_requires_third_sample():
    assert not cluster_attributes_confident([
        _sample(20),
        _sample(45),
    ])
    assert not cluster_attributes_confident([
        _sample(31, gender="Woman"),
        _sample(32, gender="Man"),
    ])


def test_cluster_summary_aggregates_samples():
    result = aggregate_cluster_attributes(
        [_sample(30), _sample(34)],
        ["a.jpg", "b.jpg"],
        True,
    )
    assert result["samples_analyzed"] == 2
    assert result["age"] == 32.0
    assert result["dominant_gender"] == "Woman"
    assert result["confidence_sufficient"] is True


def test_scanner_extracts_embeddings_during_scan_and_attributes_only_at_end():
    source = (ROOT / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    assert "result = analyzer.extract_embedding(path)" in source
    assert "def _finalize_face_clusters(" in source
    assert "analyzer.analyze_attributes(" in source
    assert "central_cluster_members(cluster, limit=3)" in source


def test_final_cluster_summary_is_written_to_all_member_records():
    source = (ROOT / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    assert 'rec["extra_metadata"]["face_cluster"] = member_summary' in source


def test_gui_exposes_cluster_summary():
    source = (ROOT / "src" / "pixelcue" / "gui.py").read_text(encoding="utf-8")
    assert 'summary = cluster.get("summary") or {}' in source
    assert "Samples analysed:" in source
