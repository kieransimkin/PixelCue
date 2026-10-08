from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path

from PySide6.QtCore import QPoint, QRect, QSize, Qt, QUrl, Signal, QTimer
from PySide6.QtGui import QDesktopServices, QFont, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QCompleter,
    QFileDialog,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .scanner import ScanWorker, export_tsv
from .model import available_tagger_profiles, DEFAULT_TAGGER_PROFILE


TAG_CLOUD_PAGE_SIZE = 100
FACE_CLUSTER_PAGE_SIZE = 100
TAG_CLOUD_REFRESH_MS = 150
GALLERY_PAGE_SIZE = 100


class FlowLayout(QLayout):
    def __init__(self, parent=None, margin=8, h_spacing=6, v_spacing=6):
        super().__init__(parent)
        self._items = []
        self.setContentsMargins(margin, margin, margin, margin)
        self.h_spacing = h_spacing
        self.v_spacing = v_spacing

    def addItem(self, item):
        self._items.append(item)

    def setWidgetOrder(self, widgets):
        """Reorder existing layout items without destroying their widgets."""
        order = {id(widget): index for index, widget in enumerate(widgets)}
        fallback = len(order) + len(self._items)
        self._items.sort(
            key=lambda item: order.get(id(item.widget()), fallback)
        )
        self.invalidate()
        parent = self.parentWidget()
        if parent is not None:
            parent.updateGeometry()
            parent.update()

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientations(Qt.Orientation(0))

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._layout(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._layout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        l, t, r, b = self.getContentsMargins()
        size += QSize(l + r, t + b)
        return size

    def _layout(self, rect, test_only):
        x = rect.x()
        y = rect.y()
        line_height = 0
        for item in self._items:
            widget = item.widget()
            space_x = self.h_spacing
            space_y = self.v_spacing
            next_x = x + item.sizeHint().width() + space_x
            if next_x - space_x > rect.right() and line_height > 0:
                x = rect.x()
                y += line_height + space_y
                next_x = x + item.sizeHint().width() + space_x
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))
            x = next_x
            line_height = max(line_height, item.sizeHint().height())
        return y + line_height - rect.y()


class ThumbButton(QPushButton):
    def __init__(self, path: str, pixmap: QPixmap | None, parent=None):
        super().__init__(parent)
        self.path = path
        self.setToolTip(path)
        self.setFixedSize(180, 160)
        self.setIconSize(QSize(150, 112))
        if pixmap and not pixmap.isNull():
            from PySide6.QtGui import QIcon
            self.setIcon(QIcon(pixmap))
        else:
            self.setText(Path(path).name[:24])
        self.clicked.connect(self.open_item)

    def open_item(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.path))


class TagItemsWindow(QWidget):
    """Top-level thumbnail gallery with cumulative AND tag filters."""

    def __init__(
        self,
        tag: str,
        paths: list[str],
        thumbs: dict[str, QPixmap],
        path_tags: dict[str, set[str]] | None = None,
        parent=None,
        required_tags: list[str] | None = None,
    ):
        # True independent OS window.
        super().__init__(None, Qt.Window)
        self.base_title = str(tag)
        self.all_paths = list(paths)
        self.thumbs = thumbs
        self.path_tags = path_tags or {}
        self.locked_tags = list(required_tags if required_tags is not None else [tag])
        self.required_tags = list(self.locked_tags)
        self.preview_query = ""
        self.render_limit = GALLERY_PAGE_SIZE
        self.filtered_paths: list[str] = []
        self.thumb_widgets: list[ThumbButton] = []

        self.available_tags = self._available_tags()
        self._tag_by_casefold = {
            candidate.casefold(): candidate
            for candidate in self.available_tags
        }

        self.resize(920, 680)
        outer = QVBoxLayout(self)

        self.title_label = QLabel()
        outer.addWidget(self.title_label)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Add required tag:"))
        self.tag_filter = QLineEdit()
        self.tag_filter.setPlaceholderText("e.g. clear water")
        self.tag_filter.setClearButtonEnabled(True)

        completer = QCompleter(self.available_tags, self.tag_filter)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setFilterMode(Qt.MatchContains)
        self.tag_filter.setCompleter(completer)

        self.add_filter_btn = QPushButton("Add")
        self.clear_filter_btn = QPushButton("Clear added filters")
        filter_row.addWidget(self.tag_filter, 1)
        filter_row.addWidget(self.add_filter_btn)
        filter_row.addWidget(self.clear_filter_btn)
        outer.addLayout(filter_row)

        self.active_filter_label = QLabel()
        self.active_filter_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        outer.addWidget(self.active_filter_label)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.host = QWidget()
        self.grid = QGridLayout(self.host)
        self.grid.setAlignment(Qt.AlignTop)
        self.scroll.setWidget(self.host)
        outer.addWidget(self.scroll, 1)

        self.tag_filter.textChanged.connect(self._on_preview_changed)
        self.tag_filter.returnPressed.connect(self.commit_filter)
        self.add_filter_btn.clicked.connect(self.commit_filter)
        self.clear_filter_btn.clicked.connect(self.clear_added_filters)
        self.scroll.verticalScrollBar().valueChanged.connect(self._on_scroll)

        self.apply_filters()

    def _available_tags(self) -> list[str]:
        counts = Counter()
        for path in self.all_paths:
            for tag in self.path_tags.get(path, set()):
                counts[str(tag)] += 1
        return [tag for tag, _count in counts.most_common()]

    @staticmethod
    def _tags_casefold(tags: set[str] | list[str]) -> set[str]:
        return {str(tag).casefold() for tag in tags}

    def _path_matches(self, path: str) -> bool:
        tags = self._tags_casefold(self.path_tags.get(path, set()))

        # Every committed requirement must be present exactly.
        for required in self.required_tags:
            if required.casefold() not in tags:
                return False

        # While typing, preview an additional requirement. Exact tag matches
        # behave exactly; otherwise substring-match against the image's tags.
        query = self.preview_query.casefold().strip()
        if query:
            canonical = self._tag_by_casefold.get(query)
            if canonical is not None:
                if canonical.casefold() not in tags:
                    return False
            elif not any(query in tag for tag in tags):
                return False

        return True

    def _canonical_filter_from_input(self, text: str) -> str | None:
        query = text.casefold().strip()
        if not query:
            return None

        exact = self._tag_by_casefold.get(query)
        if exact is not None:
            return exact

        matches = [
            tag for tag in self.available_tags
            if query in tag.casefold()
        ]
        if len(matches) == 1:
            return matches[0]
        return None

    def _on_preview_changed(self, text: str) -> None:
        self.preview_query = str(text)
        self.render_limit = GALLERY_PAGE_SIZE
        self.scroll.verticalScrollBar().setValue(0)
        self.apply_filters()

    def commit_filter(self) -> None:
        canonical = self._canonical_filter_from_input(self.tag_filter.text())
        if canonical is None:
            # Keep the live preview in place if the query is ambiguous.
            return

        if canonical.casefold() not in {
            tag.casefold() for tag in self.required_tags
        }:
            self.required_tags.append(canonical)

        self.preview_query = ""
        self.tag_filter.clear()
        self.render_limit = GALLERY_PAGE_SIZE
        self.scroll.verticalScrollBar().setValue(0)
        self.apply_filters()

    def clear_added_filters(self) -> None:
        self.required_tags = list(self.locked_tags)
        self.preview_query = ""
        self.tag_filter.clear()
        self.render_limit = GALLERY_PAGE_SIZE
        self.scroll.verticalScrollBar().setValue(0)
        self.apply_filters()

    def _on_scroll(self, value: int) -> None:
        bar = self.scroll.verticalScrollBar()
        if bar.maximum() <= 0:
            return
        if value >= bar.maximum() - max(80, bar.pageStep() // 3):
            if self.render_limit < len(self.filtered_paths):
                self.render_limit += GALLERY_PAGE_SIZE
                self._render_page()

    def apply_filters(self) -> None:
        self.filtered_paths = [
            path for path in self.all_paths
            if self._path_matches(path)
        ]
        self._update_header()
        self._render_page()

    def _update_header(self) -> None:
        count = len(self.filtered_paths)
        self.setWindowTitle(f"{self.base_title} — {count} item(s)")
        self.title_label.setText(
            f"<b>{self.base_title}</b> — {count} of {len(self.all_paths)} item(s)"
        )

        committed = " AND ".join(self.required_tags) if self.required_tags else "(none)"
        preview = self.preview_query.strip()
        if preview:
            self.active_filter_label.setText(
                f"Required tags: {committed} AND <i>{preview}</i> (preview)"
            )
        else:
            self.active_filter_label.setText(
                f"Required tags: {committed}"
            )

    def _clear_rendered_thumbs(self) -> None:
        while self.grid.count():
            item = self.grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.thumb_widgets.clear()

    def _render_page(self) -> None:
        self._clear_rendered_thumbs()

        visible = self.filtered_paths[: self.render_limit]
        cols = 4
        for i, path in enumerate(visible):
            button = ThumbButton(path, self.thumbs.get(path))
            self.thumb_widgets.append(button)
            self.grid.addWidget(button, i // cols, i % cols)

        if not visible:
            message = QLabel("No images match all required tags.")
            self.grid.addWidget(message, 0, 0, 1, cols)

        self.host.updateGeometry()


class FaceClusterButton(QPushButton):
    def __init__(self, cluster: dict, pixmap: QPixmap | None, callback, parent=None):
        super().__init__(parent)
        self._callback = callback
        self.cluster: dict = {}
        self.setFixedSize(210, 180)
        self.setIconSize(QSize(150, 112))
        self.clicked.connect(lambda: self._callback(self.cluster))
        self.update_cluster(cluster, pixmap)

    def update_cluster(self, cluster: dict, pixmap: QPixmap | None = None) -> None:
        self.cluster = dict(cluster)
        count = int(cluster.get("size", len(cluster.get("paths", []))))
        cluster_id = cluster.get("cluster_id", "?")
        self.setText(f"Person cluster {cluster_id} — {count} images")

        summary = cluster.get("summary") or {}
        details = [
            f"Image count: {count}",
            f"Representative: {cluster.get('representative_path', '')}",
        ]
        if summary:
            if summary.get("age") is not None:
                details.append(f"Age: {summary.get('age')}")
            if summary.get("dominant_gender"):
                details.append(f"Gender: {summary.get('dominant_gender')}")
            if summary.get("dominant_ethnicity"):
                details.append(f"Ethnicity: {summary.get('dominant_ethnicity')}")
            if summary.get("dominant_emotion"):
                details.append(f"Emotion: {summary.get('dominant_emotion')}")
            if summary.get("celebrity_lookalike"):
                details.append(f"Look-alike: {summary.get('celebrity_lookalike')}")
            details.append(f"Attribute samples: {summary.get('samples_analyzed', 0)}")
        self.setToolTip("\n".join(details))

        if pixmap and not pixmap.isNull():
            from PySide6.QtGui import QIcon
            self.setIcon(QIcon(pixmap))



class FaceClusterWindow(TagItemsWindow):
    def __init__(
        self,
        cluster: dict,
        thumbs: dict[str, QPixmap],
        path_tags: dict[str, set[str]],
        parent=None,
    ):
        cluster_id = cluster.get("cluster_id", "?")
        paths = list(cluster.get("paths", []))
        super().__init__(
            f"Person cluster {cluster_id}",
            paths,
            thumbs,
            path_tags,
            None,
            required_tags=[],
        )

        summary = cluster.get("summary") or {}
        if summary:
            bits = []
            if summary.get("age") is not None:
                bits.append(f"Age ≈ {summary.get('age')}")
            if summary.get("dominant_gender"):
                bits.append(f"Gender: {summary.get('dominant_gender')}")
            if summary.get("dominant_ethnicity"):
                bits.append(f"Ethnicity: {summary.get('dominant_ethnicity')}")
            if summary.get("dominant_emotion"):
                bits.append(f"Emotion: {summary.get('dominant_emotion')}")
            if summary.get("celebrity_lookalike"):
                bits.append(f"Look-alike: {summary.get('celebrity_lookalike')}")
            bits.append(
                f"Samples analysed: {summary.get('samples_analyzed', 0)}"
            )
            bits.append(
                "Early-stop confidence: "
                + ("yes" if summary.get("confidence_sufficient") else "no")
            )
            self.layout().insertWidget(1, QLabel(" • ".join(bits)))


class ErrorLogDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PixelCue errors")
        self.resize(900, 560)
        layout = QVBoxLayout(self)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        layout.addWidget(self.text)

    def append_error(self, path: str, short: str, details: str):
        self.text.appendPlainText(
            f"{'=' * 80}\n{path}\n{short}\n\n{details.strip()}\n"
        )


class MainWindow(QMainWindow):
    def __init__(
        self,
        startup_path: str | None = None,
        startup_model_profile: str | None = None,
    ):
        super().__init__()
        self.setWindowTitle("PixelCue")
        self.resize(1100, 760)

        self.worker: ScanWorker | None = None
        self.tag_to_paths: dict[str, list[str]] = defaultdict(list)
        self.path_tags: dict[str, set[str]] = defaultdict(set)
        self.tag_counts = Counter()
        self.thumbs: dict[str, QPixmap] = {}
        self.records = {}
        self.tag_windows: list[TagItemsWindow] = []
        self.face_cluster_windows: list[FaceClusterWindow] = []
        self.face_clusters: list[dict] = []
        self.tag_buttons: dict[str, QPushButton] = {}
        self.face_cluster_buttons: dict[str, FaceClusterButton] = {}
        self.tag_render_limit = TAG_CLOUD_PAGE_SIZE
        self.tag_filter_text = ""
        self.face_cluster_render_limit = FACE_CLUSTER_PAGE_SIZE
        self._tag_refresh_pending = False
        self._tag_placeholder: QLabel | None = None
        self._face_placeholder: QLabel | None = None
        self.error_count = 0
        self.error_log = ErrorLogDialog(self)
        self._first_error_shown = False
        self.model_download_total = 0
        self.model_total_is_exact = False

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        top = QHBoxLayout()
        self.choose_btn = QPushButton("Select starting folder…")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.error_btn = QPushButton("Errors: 0")
        self.error_btn.setEnabled(False)
        self.model_combo = QComboBox()
        profiles = available_tagger_profiles()
        selected_profile = startup_model_profile or DEFAULT_TAGGER_PROFILE
        for profile in profiles:
            self.model_combo.addItem(profile.label, profile.id)
            index = self.model_combo.count() - 1
            self.model_combo.setItemData(
                index,
                f"{profile.description}  Parameters: {profile.approximate_parameters}",
                Qt.ToolTipRole,
            )
            if profile.id == selected_profile:
                self.model_combo.setCurrentIndex(index)
        self.active_model_label = self.model_combo.currentText()
        self.path_label = QLabel(startup_path or "No start path selected")
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        top.addWidget(self.choose_btn)
        top.addWidget(self.stop_btn)
        top.addWidget(self.error_btn)
        top.addWidget(QLabel("Tagger:"))
        top.addWidget(self.model_combo)
        top.addWidget(self.path_label, 1)
        layout.addLayout(top)

        model_row = QHBoxLayout()
        self.model_label = QLabel("Model: not loaded")
        self.model_progress = QProgressBar()
        self.model_progress.setTextVisible(False)
        self.model_progress.setFixedWidth(180)
        self.model_progress.hide()
        model_row.addWidget(self.model_label, 1)
        model_row.addWidget(self.model_progress)
        layout.addLayout(model_row)

        self.info = QLabel("Choose a folder to begin. Hidden folders are included." if not startup_path else f"Ready to scan: {startup_path}")
        layout.addWidget(self.info)

        tags_header = QHBoxLayout()
        tags_title = QLabel("<b>VLM tags</b>")
        self.tag_filter = QLineEdit()
        self.tag_filter.setPlaceholderText("Filter tags…")
        self.tag_filter.setClearButtonEnabled(True)
        self.tag_filter.setMaximumWidth(320)
        self.tag_count_label = QLabel("0 tags")
        tags_header.addWidget(tags_title)
        tags_header.addStretch(1)
        tags_header.addWidget(QLabel("Search:"))
        tags_header.addWidget(self.tag_filter)
        tags_header.addWidget(self.tag_count_label)
        layout.addLayout(tags_header)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.cloud_host = QWidget()
        self.cloud = FlowLayout(self.cloud_host)
        self.scroll.setWidget(self.cloud_host)
        self.scroll.verticalScrollBar().valueChanged.connect(self._on_tag_scroll)
        self.tag_filter.textChanged.connect(self.on_tag_filter_changed)
        layout.addWidget(self.scroll, 2)

        faces_header = QHBoxLayout()
        faces_title = QLabel("<b>Face identity clusters</b>")
        self.face_cluster_count_label = QLabel("0 clusters")
        faces_header.addWidget(faces_title)
        faces_header.addStretch(1)
        faces_header.addWidget(self.face_cluster_count_label)
        layout.addLayout(faces_header)

        self.face_scroll = QScrollArea()
        self.face_scroll.setWidgetResizable(True)
        self.face_scroll.setMinimumHeight(190)
        self.face_cluster_host = QWidget()
        self.face_cluster_layout = FlowLayout(self.face_cluster_host)
        self.face_scroll.setWidget(self.face_cluster_host)
        self.face_scroll.verticalScrollBar().valueChanged.connect(
            self._on_face_cluster_scroll
        )
        layout.addWidget(self.face_scroll, 1)
        self.rebuild_face_clusters()

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.count_label = QLabel("0 files discovered")
        self.queue_label = QLabel("Media: 0 queued / 0 tagged")
        self.face_queue_label = QLabel("Faces: 0 queued / 0 analysed")
        self.video_queue_label = QLabel("Video sampling: 0 queued / 0 done")
        self.status.addPermanentWidget(self.face_queue_label)
        self.status.addPermanentWidget(self.video_queue_label)
        self.status.addPermanentWidget(self.queue_label)
        self.status.addPermanentWidget(self.count_label)

        self.choose_btn.clicked.connect(self.choose_folder)
        self.stop_btn.clicked.connect(self.stop_scan)
        self.error_btn.clicked.connect(self.error_log.show)

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose starting folder")
        if not folder:
            return
        self.start_scan(folder)

    def start_scan(self, folder: str):
        if self.worker and self.worker.isRunning():
            return
        self.tag_to_paths.clear()
        self.path_tags.clear()
        self.tag_counts.clear()
        self.tag_filter.clear()
        self.tag_filter_text = ""
        self.thumbs.clear()
        self.records.clear()
        self.face_clusters.clear()
        self.tag_render_limit = TAG_CLOUD_PAGE_SIZE
        self.face_cluster_render_limit = FACE_CLUSTER_PAGE_SIZE
        self._tag_refresh_pending = False
        self.error_count = 0
        self._first_error_shown = False
        self.error_btn.setText("Errors: 0")
        self.error_btn.setEnabled(False)
        self.error_log.text.clear()
        self.queue_label.setText("Media: 0 queued / 0 processed")
        self.face_queue_label.setText("Faces: 0 queued / 0 analysed")
        self.video_queue_label.setText("Video sampling: 0 queued / 0 done")
        self.rebuild_face_clusters()
        self.model_download_total = 0
        self.model_total_is_exact = False
        self.model_label.setText("Model: preparing…")
        self.model_progress.setRange(0, 0)
        self.model_progress.show()
        self.rebuild_cloud()

        self.path_label.setText(folder)
        self.choose_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.model_combo.setEnabled(False)
        profile_id = str(self.model_combo.currentData() or "joycaption")
        self.active_model_label = self.model_combo.currentText()
        self.worker = ScanWorker(folder, model_profile_id=profile_id, parent=self)
        self.worker.status.connect(self.status.showMessage)
        self.worker.current_path.connect(lambda p: self.info.setText(f"Processing: {p}"))
        self.worker.media_queue_count.connect(self.on_media_queue_count)
        self.worker.face_queue_count.connect(self.on_face_queue_count)
        self.worker.video_sample_queue_count.connect(self.on_video_sample_queue_count)
        self.worker.face_clusters_updated.connect(self.on_face_clusters_updated)
        self.worker.discovered_count.connect(lambda n: self.count_label.setText(f"{n} files discovered"))
        self.worker.tagged.connect(self.on_tagged)
        self.worker.symlink_question.connect(self.ask_symlink)
        self.worker.model_status.connect(self.on_model_status)
        self.worker.model_progress.connect(self.on_model_progress)
        self.worker.processing_error.connect(self.on_processing_error)
        self.worker.fatal_error.connect(self.on_fatal)
        self.worker.completed.connect(self.on_completed)
        self.worker.start()

    def stop_scan(self):
        if self.worker:
            self.worker.request_stop()
            self.stop_btn.setEnabled(False)

    def on_media_queue_count(self, waiting: int, completed: int):
        self.queue_label.setText(
            f"Media: {waiting} queued / {completed} processed"
        )

    def on_video_sample_queue_count(self, waiting: int, completed: int):
        self.video_queue_label.setText(
            f"Video sampling: {waiting} queued / {completed} done"
        )

    def on_face_queue_count(self, waiting: int, completed: int):
        self.face_queue_label.setText(
            f"Faces: {waiting} queued / {completed} analysed"
        )

    def on_face_clusters_updated(self, clusters_obj: object):
        self.face_clusters = list(clusters_obj or [])
        self.face_cluster_count_label.setText(
            f"{len(self.face_clusters)} clusters"
        )
        self.sync_face_clusters()

    def rebuild_face_clusters(self):
        """Clear the face pane once; subsequent updates are incremental."""
        while self.face_cluster_layout.count():
            item = self.face_cluster_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.face_cluster_buttons.clear()
        self._face_placeholder = QLabel(
            "Face clusters will appear as DeepFace embeddings are analysed."
        )
        self.face_cluster_layout.addWidget(self._face_placeholder)

    def _on_face_cluster_scroll(self, value: int) -> None:
        bar = self.face_scroll.verticalScrollBar()
        if bar.maximum() <= 0:
            return
        if value >= bar.maximum() - max(80, bar.pageStep() // 3):
            if self.face_cluster_render_limit < len(self.face_clusters):
                self.face_cluster_render_limit += FACE_CLUSTER_PAGE_SIZE
                self.sync_face_clusters()

    def sync_face_clusters(self):
        visible_clusters = self.face_clusters[: self.face_cluster_render_limit]
        if visible_clusters and self._face_placeholder is not None:
            self._face_placeholder.setParent(None)
            self._face_placeholder.deleteLater()
            self._face_placeholder = None

        incoming_keys: set[str] = set()
        ordered_buttons = []
        for cluster in visible_clusters:
            representative = str(cluster.get("representative_path", ""))
            if not representative:
                continue
            incoming_keys.add(representative)
            pixmap = self.thumbs.get(representative)
            button = self.face_cluster_buttons.get(representative)
            if button is None:
                button = FaceClusterButton(
                    cluster, pixmap, self.open_face_cluster
                )
                self.face_cluster_buttons[representative] = button
                self.face_cluster_layout.addWidget(button)
            else:
                button.update_cluster(cluster, pixmap)
            ordered_buttons.append(button)

        stale = set(self.face_cluster_buttons) - incoming_keys
        for key in stale:
            button = self.face_cluster_buttons.pop(key)
            button.setParent(None)
            button.deleteLater()

        if ordered_buttons:
            self.face_cluster_layout.setWidgetOrder(ordered_buttons)

        if not visible_clusters and self._face_placeholder is None:
            self._face_placeholder = QLabel(
                "Face clusters will appear as DeepFace embeddings are analysed."
            )
            self.face_cluster_layout.addWidget(self._face_placeholder)
        self.face_cluster_host.updateGeometry()

    @staticmethod
    def _remove_window_by_id(collection: list, target_id: int) -> None:
        """Forget a destroyed top-level window without touching its Qt object."""
        collection[:] = [candidate for candidate in collection if id(candidate) != target_id]

    def _track_top_level_window(self, collection: list, win: QWidget) -> None:
        """Keep a gallery alive, then remove it safely when Qt destroys it."""
        collection.append(win)
        target_id = id(win)
        win.destroyed.connect(
            lambda _obj=None, items=collection, ident=target_id: self._remove_window_by_id(
                items, ident
            )
        )

    def open_face_cluster(self, cluster: dict):
        win = FaceClusterWindow(cluster, self.thumbs, self.path_tags, None)
        win.setAttribute(Qt.WA_DeleteOnClose, True)
        self._track_top_level_window(self.face_cluster_windows, win)
        win.show()

    def on_record(self, path: str, rec: object):
        self.records[path] = rec

    def on_tagged(self, path: str, tags_obj: object, thumb_bytes: bytes):
        tags = list(tags_obj or [])
        if thumb_bytes:
            pm = QPixmap()
            pm.loadFromData(thumb_bytes, "JPEG")
            if not pm.isNull():
                self.thumbs[path] = pm

        self.path_tags[path].update(str(tag) for tag in tags)

        for tag in tags:
            if path not in self.tag_to_paths[tag]:
                self.tag_to_paths[tag].append(path)
                self.tag_counts[tag] += 1
        # Let the coalesced cloud refresh update the count label as well.
        # Writing the unfiltered total here would momentarily overwrite an
        # active search result count while cached/tagging events stream in.
        self.schedule_tag_cloud_sync()

    def _current_tag_filter_query(self) -> str:
        """Read the live widget value so queued refreshes cannot use stale state."""
        return self.tag_filter.text().casefold().strip()

    def _filtered_ranked_tags(self, query: str | None = None) -> list[str]:
        if query is None:
            query = self._current_tag_filter_query()
        ranked = [tag for tag, _count in self.tag_counts.most_common()]
        if not query:
            return ranked
        return [tag for tag in ranked if query in tag.casefold()]

    def on_tag_filter_changed(self, text: str) -> None:
        """Apply search immediately; background tag updates stay throttled."""
        self.tag_filter_text = str(text)
        self.tag_render_limit = TAG_CLOUD_PAGE_SIZE
        self.scroll.verticalScrollBar().setValue(0)

        # Search is user-interactive and must never wait behind a pending
        # coalesced refresh generated by cached/new tagging results. Mark any
        # pending refresh as satisfied and synchronously render the current
        # widget text. A previously queued singleShot may still run later, but
        # sync_tag_cloud() always re-reads the live QLineEdit value.
        self._tag_refresh_pending = False
        self.sync_tag_cloud()

    def rebuild_cloud(self):
        """Reset the lazy tag cloud to its first page."""
        while self.cloud.count():
            item = self.cloud.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.tag_buttons.clear()
        self.tag_render_limit = TAG_CLOUD_PAGE_SIZE
        self._tag_refresh_pending = False
        self.tag_count_label.setText(f"{len(self.tag_counts)} tags")
        self._tag_placeholder = QLabel(
            "Tags will appear here as the selected VLM processes media."
        )
        self.cloud.addWidget(self._tag_placeholder)

    def schedule_tag_cloud_sync(self, *, immediate: bool = False) -> None:
        """Coalesce many tag updates into one bounded GUI refresh."""
        if self._tag_refresh_pending:
            return
        self._tag_refresh_pending = True
        QTimer.singleShot(
            0 if immediate else TAG_CLOUD_REFRESH_MS,
            self.sync_tag_cloud,
        )

    def _on_tag_scroll(self, value: int) -> None:
        bar = self.scroll.verticalScrollBar()
        if bar.maximum() <= 0:
            return
        if value >= bar.maximum() - max(80, bar.pageStep() // 3):
            matched_count = len(self._filtered_ranked_tags())
            if self.tag_render_limit < matched_count:
                self.tag_render_limit += TAG_CLOUD_PAGE_SIZE
                self.schedule_tag_cloud_sync(immediate=True)

    def sync_tag_cloud(self) -> None:
        """Render only the highest-ranked visible page of matching tags."""
        self._tag_refresh_pending = False
        query = self._current_tag_filter_query()
        query_text = self.tag_filter.text().strip()
        ranked = self._filtered_ranked_tags(query)
        visible_tags = ranked[: self.tag_render_limit]

        if visible_tags and self._tag_placeholder is not None:
            self._tag_placeholder.setParent(None)
            self._tag_placeholder.deleteLater()
            self._tag_placeholder = None

        visible_set = set(visible_tags)
        ordered_buttons = []
        for tag in visible_tags:
            count = int(self.tag_counts.get(tag, 0))
            button = self.tag_buttons.get(tag)
            if button is None:
                button = QPushButton()
                button.setFlat(True)
                button.setCursor(Qt.PointingHandCursor)
                button.clicked.connect(
                    lambda _checked=False, t=tag: self.open_tag(t)
                )
                self.tag_buttons[tag] = button
                self.cloud.addWidget(button)

            button.setText(f"{tag}  {count}")
            button.setToolTip(f"Show {count} item(s) tagged {tag}")
            font = button.font()
            font.setPointSizeF(
                9.0 + min(13.0, math.log2(count + 1) * 2.2)
            )
            button.setFont(font)
            ordered_buttons.append(button)

        # Crucially, tags outside the currently materialised page have no Qt
        # widgets at all. They stay only in Counter/dicts until scrolling asks
        # for the next page.
        stale = set(self.tag_buttons) - visible_set
        for tag in stale:
            button = self.tag_buttons.pop(tag)
            button.setParent(None)
            button.deleteLater()

        if ordered_buttons:
            self.cloud.setWidgetOrder(ordered_buttons)

        if not visible_tags and self._tag_placeholder is None:
            if query:
                placeholder_text = (
                    f"No tags match “{query_text}”."
                )
            else:
                placeholder_text = (
                    "Tags will appear here as the selected VLM processes media."
                )
            self._tag_placeholder = QLabel(placeholder_text)
            self.cloud.addWidget(self._tag_placeholder)
        elif not visible_tags and self._tag_placeholder is not None:
            if query:
                self._tag_placeholder.setText(
                    f"No tags match “{query_text}”."
                )
            else:
                self._tag_placeholder.setText(
                    "Tags will appear here as the selected VLM processes media."
                )

        shown = len(visible_tags)
        matched = len(ranked)
        total = len(self.tag_counts)
        if query:
            self.tag_count_label.setText(
                f"{matched} of {total} tags — showing {shown}"
            )
        else:
            self.tag_count_label.setText(
                f"{total} tags — showing {shown}" if total else "0 tags"
            )
        self.cloud_host.updateGeometry()

    def open_tag(self, tag: str):
        win = TagItemsWindow(
            tag,
            list(self.tag_to_paths.get(tag, [])),
            self.thumbs,
            self.path_tags,
            None,
            required_tags=[tag],
        )
        win.setAttribute(Qt.WA_DeleteOnClose, True)
        self._track_top_level_window(self.tag_windows, win)
        win.show()

    def ask_symlink(self, path: str, target: str):
        box = QMessageBox(self)
        box.setWindowTitle("Follow symbolic link?")
        box.setIcon(QMessageBox.Question)
        box.setText(f"Follow this symbolic link?\n\n{path}\n→ {target}")
        box.setInformativeText("Scanning continues while this question is open.")
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.No)
        box.setModal(False)

        def done(result: int):
            follow = result == QMessageBox.Yes
            if self.worker:
                self.worker.resolve_symlink(path, follow)
            box.deleteLater()

        box.finished.connect(done)
        box.open()

    @staticmethod
    def _human_bytes(value: int) -> str:
        value = max(0, int(value))
        units = ["bytes", "KiB", "MiB", "GiB", "TiB"]
        number = float(value)
        unit = units[0]
        for unit in units:
            if number < 1024.0 or unit == units[-1]:
                break
            number /= 1024.0
        if unit == "bytes":
            return f"{value:,} bytes"
        return f"{number:.2f} {unit}"

    def on_model_status(self, message: str):
        low = message.casefold()
        if "exact size confirmed" in low:
            self.model_total_is_exact = True
        elif "lookup timed out" in low or "lookup" in low and "failed" in low:
            self.model_total_is_exact = False

        # Do not overwrite an active quantitative download line with a transient
        # metadata warning unless no progress denominator exists yet.
        if "lookup timed out" not in low and "lookup" not in low:
            self.model_label.setText(f"Model: {message}")
        else:
            self.status.showMessage(message, 15000)

        if any(word in low for word in ("querying", "checking", "starting", "loading", "verifying")):
            self.model_progress.setRange(0, 0)
            self.model_progress.show()
        elif any(word in low for word in ("complete", "cached", "ready", "failed", "deferred")):
            self.model_progress.hide()

    def on_model_progress(self, done_obj: object, total_obj: object):
        done = max(0, int(done_obj))
        total = max(0, int(total_obj))
        self.model_download_total = total

        if total <= 0:
            self.model_progress.setRange(0, 0)
            self.model_progress.show()
            self.model_label.setText(
                f"Model: {self.active_model_label} downloading — {done:,} bytes received"
            )
            return

        done = min(done, total)
        percentage = (done / total) * 100.0

        # QProgressBar uses a C++ int, so represent 0.0–100.0% as 0–1000.
        self.model_progress.setRange(0, 1000)
        self.model_progress.setValue(round((done / total) * 1000))
        self.model_progress.show()

        approx = "" if self.model_total_is_exact else "≈"
        self.model_label.setText(
            f"Model: {self.active_model_label} downloading — "
            f"{done:,} / {approx}{total:,} bytes "
            f"({self._human_bytes(done)} / {approx}{self._human_bytes(total)}, "
            f"{percentage:.1f}%)"
        )

    def on_processing_error(self, path: str, short: str, details: str):
        self.error_count += 1
        self.error_btn.setText(f"Errors: {self.error_count}")
        self.error_btn.setEnabled(True)
        self.error_log.append_error(path, short, details)
        self.status.showMessage(short)

        if not self._first_error_shown:
            self._first_error_shown = True
            box = QMessageBox(self)
            box.setWindowTitle("PixelCue processing error")
            box.setIcon(QMessageBox.Critical)
            box.setText(short)
            box.setInformativeText(
                "PixelCue will continue scanning the filesystem. "
                "Open the Errors window for the complete diagnostic log."
            )
            box.setDetailedText(details)
            box.setStandardButtons(QMessageBox.Ok)
            box.setModal(False)
            box.open()
            self._error_box = box

    def on_fatal(self, details: str):
        QMessageBox.critical(self, "Scanner error", details)

    def on_completed(self, records_obj: object):
        records = list(records_obj or [])
        self.choose_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.model_combo.setEnabled(True)
        self.info.setText(f"Scan complete: {len(records)} filesystem file entries recorded.")
        self.status.showMessage("Scan complete")

        suggested = str(Path(self.path_label.text()).name or "scan") + "-media-scan.tsv"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save scan TSV",
            suggested,
            "TSV files (*.tsv);;All files (*)",
        )
        if path:
            try:
                export_tsv(records, path)
                self.status.showMessage(f"Saved {path}")
            except Exception as e:
                QMessageBox.critical(self, "Could not save TSV", str(e))

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.request_stop()
            self.worker.wait(3000)
        super().closeEvent(event)
