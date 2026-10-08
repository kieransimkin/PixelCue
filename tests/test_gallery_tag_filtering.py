from pathlib import Path
import ast

ROOT = Path(__file__).parents[1]


def test_tag_gallery_has_additive_filter_ui_and_autocomplete():
    source = (ROOT / "src" / "pixelcue" / "gui.py").read_text(encoding="utf-8")
    assert 'self.tag_filter.setPlaceholderText("e.g. clear water")' in source
    assert "QCompleter(self.available_tags" in source
    assert "self.tag_filter.returnPressed.connect(self.commit_filter)" in source
    assert 'QPushButton("Clear added filters")' in source


def test_tag_gallery_uses_and_semantics_for_committed_tags():
    source = (ROOT / "src" / "pixelcue" / "gui.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "TagItemsWindow"
    )
    fn = next(
        n for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "_path_matches"
    )
    segment = ast.get_source_segment(source, fn)
    assert "for required in self.required_tags:" in segment
    assert "return False" in segment
    assert "required.casefold() not in tags" in segment


def test_clicked_tag_is_locked_in_normal_tag_gallery():
    source = (ROOT / "src" / "pixelcue" / "gui.py").read_text(encoding="utf-8")
    start = source.index("def open_tag(self, tag: str)")
    end = source.index("def ask_symlink", start)
    segment = source[start:end]
    assert "required_tags=[tag]" in segment


def test_face_cluster_gallery_uses_same_filter_component_without_locked_tag():
    source = (ROOT / "src" / "pixelcue" / "gui.py").read_text(encoding="utf-8")
    assert "class FaceClusterWindow(TagItemsWindow)" in source
    start = source.index("class FaceClusterWindow")
    end = source.index("class ErrorLogDialog", start)
    segment = source[start:end]
    assert "path_tags" in segment
    assert "required_tags=[]" in segment


def test_main_window_tracks_tags_per_path_for_gallery_intersections():
    source = (ROOT / "src" / "pixelcue" / "gui.py").read_text(encoding="utf-8")
    assert "self.path_tags: dict[str, set[str]] = defaultdict(set)" in source
    assert "self.path_tags[path].update" in source
    assert "self.path_tags," in source


def test_gallery_lazy_loads_thumbnails_in_pages():
    source = (ROOT / "src" / "pixelcue" / "gui.py").read_text(encoding="utf-8")
    assert "GALLERY_PAGE_SIZE = 100" in source
    assert "self.render_limit += GALLERY_PAGE_SIZE" in source
    assert "self.filtered_paths[: self.render_limit]" in source


def test_live_preview_allows_partial_tag_search_without_committing():
    source = (ROOT / "src" / "pixelcue" / "gui.py").read_text(encoding="utf-8")
    assert "elif not any(query in tag for tag in tags):" in source
    assert "(preview)" in source
