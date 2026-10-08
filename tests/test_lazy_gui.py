from pathlib import Path

ROOT=Path(__file__).parents[1]


def test_tag_cloud_is_paged_in_batches_of_100():
    source=(ROOT/'src'/'pixelcue'/'gui.py').read_text(encoding='utf-8')
    assert 'TAG_CLOUD_PAGE_SIZE = 100' in source
    assert 'ranked[: self.tag_render_limit]' in source
    assert 'self.tag_render_limit += TAG_CLOUD_PAGE_SIZE' in source
    assert 'verticalScrollBar().valueChanged.connect(self._on_tag_scroll)' in source


def test_tag_updates_are_coalesced_not_rendered_per_tag_event():
    source=(ROOT/'src'/'pixelcue'/'gui.py').read_text(encoding='utf-8')
    assert 'TAG_CLOUD_REFRESH_MS = 150' in source
    assert 'self.schedule_tag_cloud_sync()' in source
    assert 'QTimer.singleShot' in source
    # old per-tag immediate widget update function is gone
    assert 'def update_tag_button' not in source


def test_face_cluster_tiles_show_image_count():
    source=(ROOT/'src'/'pixelcue'/'gui.py').read_text(encoding='utf-8')
    assert 'Person cluster {cluster_id} — {count} images' in source
    assert 'Image count: {count}' in source


def test_face_clusters_are_also_lazily_materialised():
    source=(ROOT/'src'/'pixelcue'/'gui.py').read_text(encoding='utf-8')
    assert 'FACE_CLUSTER_PAGE_SIZE = 100' in source
    assert 'self.face_clusters[: self.face_cluster_render_limit]' in source
    assert 'self.face_cluster_render_limit += FACE_CLUSTER_PAGE_SIZE' in source
