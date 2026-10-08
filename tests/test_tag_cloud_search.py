from pathlib import Path
import ast


ROOT = Path(__file__).parents[1]


def test_gui_has_instant_tag_filter_field():
    source = (ROOT / "src" / "pixelcue" / "gui.py").read_text(encoding="utf-8")
    assert "self.tag_filter = QLineEdit()" in source
    assert 'self.tag_filter.setPlaceholderText("Filter tags…")' in source
    assert "self.tag_filter.textChanged.connect(self.on_tag_filter_changed)" in source


def test_filter_is_case_insensitive_and_frequency_rank_preserving():
    source = (ROOT / "src" / "pixelcue" / "gui.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MainWindow"
    )
    fn = next(
        n for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_filtered_ranked_tags"
    )
    segment = ast.get_source_segment(source, fn)
    assert "self.tag_counts.most_common()" in segment
    assert "query in tag.casefold()" in segment
    assert "if not query:" in segment


def test_filter_reads_live_search_widget_not_stale_shadow_state():
    source = (ROOT / "src" / "pixelcue" / "gui.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MainWindow"
    )
    current = next(
        n for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_current_tag_filter_query"
    )
    segment = ast.get_source_segment(source, current)
    assert "self.tag_filter.text().casefold().strip()" in segment


def test_search_bypasses_pending_background_refresh_queue():
    source = (ROOT / "src" / "pixelcue" / "gui.py").read_text(encoding="utf-8")
    start = source.index("def on_tag_filter_changed")
    end = source.index("def rebuild_cloud", start)
    segment = source[start:end]
    assert "self.tag_render_limit = TAG_CLOUD_PAGE_SIZE" in segment
    assert "setValue(0)" in segment
    assert "self._tag_refresh_pending = False" in segment
    assert "self.sync_tag_cloud()" in segment
    assert "schedule_tag_cloud_sync" not in segment


def test_sync_uses_one_live_query_for_filter_placeholder_and_count():
    source = (ROOT / "src" / "pixelcue" / "gui.py").read_text(encoding="utf-8")
    start = source.index("def sync_tag_cloud")
    end = source.index("def open_tag", start)
    segment = source[start:end]
    assert "query = self._current_tag_filter_query()" in segment
    assert "ranked = self._filtered_ranked_tags(query)" in segment
    assert "if query:" in segment


def test_incoming_tags_do_not_overwrite_filtered_count_immediately():
    source = (ROOT / "src" / "pixelcue" / "gui.py").read_text(encoding="utf-8")
    start = source.index("def on_tagged")
    end = source.index("def _current_tag_filter_query", start)
    segment = source[start:end]
    assert "self.schedule_tag_cloud_sync()" in segment
    assert "tag_count_label.setText" not in segment


def test_lazy_scroll_uses_filtered_match_count():
    source = (ROOT / "src" / "pixelcue" / "gui.py").read_text(encoding="utf-8")
    assert "matched_count = len(self._filtered_ranked_tags())" in source
    assert "self.tag_render_limit < matched_count" in source


def test_filter_count_shows_matches_vs_total():
    source = (ROOT / "src" / "pixelcue" / "gui.py").read_text(encoding="utf-8")
    assert 'f"{matched} of {total} tags — showing {shown}"' in source
