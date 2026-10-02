from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path

from PySide6.QtCore import QPoint, QRect, QSize, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .scanner import ScanWorker, export_tsv


class FlowLayout(QLayout):
    def __init__(self, parent=None, margin=8, h_spacing=6, v_spacing=6):
        super().__init__(parent)
        self._items = []
        self.setContentsMargins(margin, margin, margin, margin)
        self.h_spacing = h_spacing
        self.v_spacing = v_spacing

    def addItem(self, item):
        self._items.append(item)

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
    def __init__(self, tag: str, paths: list[str], thumbs: dict[str, QPixmap], parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{tag} — {len(paths)} item(s)")
        self.resize(900, 650)

        outer = QVBoxLayout(self)
        title = QLabel(f"<b>{tag}</b> — {len(paths)} item(s)")
        outer.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        host = QWidget()
        grid = QGridLayout(host)
        grid.setAlignment(Qt.AlignTop)

        cols = 4
        for i, path in enumerate(paths):
            grid.addWidget(ThumbButton(path, thumbs.get(path)), i // cols, i % cols)

        scroll.setWidget(host)
        outer.addWidget(scroll)


class MainWindow(QMainWindow):
    def __init__(self, startup_path: str | None = None):
        super().__init__()
        self.setWindowTitle("PixelCue")
        self.resize(1100, 760)

        self.worker: ScanWorker | None = None
        self.tag_to_paths: dict[str, list[str]] = defaultdict(list)
        self.tag_counts = Counter()
        self.thumbs: dict[str, QPixmap] = {}
        self.records = {}
        self.tag_windows: list[TagItemsWindow] = []

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        top = QHBoxLayout()
        self.choose_btn = QPushButton("Select starting folder…")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.path_label = QLabel(startup_path or "No start path selected")
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        top.addWidget(self.choose_btn)
        top.addWidget(self.stop_btn)
        top.addWidget(self.path_label, 1)
        layout.addLayout(top)

        self.info = QLabel("Choose a folder to begin. Hidden folders are included." if not startup_path else f"Ready to scan: {startup_path}")
        layout.addWidget(self.info)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.cloud_host = QWidget()
        self.cloud = FlowLayout(self.cloud_host)
        self.scroll.setWidget(self.cloud_host)
        layout.addWidget(self.scroll, 1)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.count_label = QLabel("0 files")
        self.status.addPermanentWidget(self.count_label)

        self.choose_btn.clicked.connect(self.choose_folder)
        self.stop_btn.clicked.connect(self.stop_scan)

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose starting folder")
        if not folder:
            return
        self.start_scan(folder)

    def start_scan(self, folder: str):
        if self.worker and self.worker.isRunning():
            return
        self.tag_to_paths.clear()
        self.tag_counts.clear()
        self.thumbs.clear()
        self.records.clear()
        self.rebuild_cloud()

        self.path_label.setText(folder)
        self.choose_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.worker = ScanWorker(folder, self)
        self.worker.status.connect(self.status.showMessage)
        self.worker.current_path.connect(lambda p: self.info.setText(f"Processing: {p}"))
        self.worker.discovered_count.connect(lambda n: self.count_label.setText(f"{n} files"))
        self.worker.tagged.connect(self.on_tagged)
        self.worker.record_updated.connect(self.on_record)
        self.worker.symlink_question.connect(self.ask_symlink)
        self.worker.fatal_error.connect(self.on_fatal)
        self.worker.completed.connect(self.on_completed)
        self.worker.start()

    def stop_scan(self):
        if self.worker:
            self.worker.request_stop()
            self.stop_btn.setEnabled(False)

    def on_record(self, path: str, rec: object):
        self.records[path] = rec

    def on_tagged(self, path: str, tags_obj: object, thumb_bytes: bytes):
        tags = list(tags_obj or [])
        if thumb_bytes:
            pm = QPixmap()
            pm.loadFromData(thumb_bytes, "JPEG")
            if not pm.isNull():
                self.thumbs[path] = pm

        for tag in tags:
            if path not in self.tag_to_paths[tag]:
                self.tag_to_paths[tag].append(path)
                self.tag_counts[tag] += 1
        self.rebuild_cloud()

    def rebuild_cloud(self):
        while self.cloud.count():
            item = self.cloud.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not self.tag_counts:
            label = QLabel("Tags will appear here as JoyCaption processes media.")
            self.cloud.addWidget(label)
            return

        max_count = max(self.tag_counts.values())
        for tag, count in self.tag_counts.most_common():
            btn = QPushButton(f"{tag}  {count}")
            # Log-ish scaling keeps a dominant tag from making everything else tiny.
            ratio = math.log1p(count) / math.log1p(max_count) if max_count else 1.0
            font = btn.font()
            font.setPointSizeF(9.0 + ratio * 13.0)
            btn.setFont(font)
            btn.setFlat(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setToolTip(f"Show {count} item(s) tagged {tag}")
            btn.clicked.connect(lambda _checked=False, t=tag: self.open_tag(t))
            self.cloud.addWidget(btn)
        self.cloud_host.updateGeometry()

    def open_tag(self, tag: str):
        win = TagItemsWindow(tag, list(self.tag_to_paths.get(tag, [])), self.thumbs, self)
        win.setAttribute(Qt.WA_DeleteOnClose, True)
        self.tag_windows.append(win)
        win.destroyed.connect(lambda: self._prune_windows())
        win.show()

    def _prune_windows(self):
        self.tag_windows = [w for w in self.tag_windows if w is not None and not w.isHidden()]

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

    def on_fatal(self, details: str):
        QMessageBox.critical(self, "Scanner error", details)

    def on_completed(self, records_obj: object):
        records = list(records_obj or [])
        self.choose_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
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
