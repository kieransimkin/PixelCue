from __future__ import annotations

import csv
import json
import os
import threading
import traceback
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThread, Signal

from .archives import is_archive, sample_archive
from .media import (
    first_video_frame,
    is_image,
    is_video,
    open_image,
    sample_video_frames,
    thumbnail_jpeg,
    video_duration_seconds,
)
from .metadata import MagicDetector, stat_record
from .model import JoyCaption4Bit


@dataclass
class QueueItem:
    path: Path
    follow_symlink: bool = False


class ScanWorker(QThread):
    status = Signal(str)
    current_path = Signal(str)
    discovered_count = Signal(int)
    tagged = Signal(str, object, bytes)      # path, list[str], representative thumbnail bytes
    record_updated = Signal(str, object)     # path, record dict
    symlink_question = Signal(str, str)      # path, target
    fatal_error = Signal(str)
    completed = Signal(object)               # list[dict]

    def __init__(self, start_dir: str, parent=None) -> None:
        super().__init__(parent)
        self.start_dir = Path(start_dir)
        self.todo: deque[QueueItem] = deque([QueueItem(self.start_dir)])
        self.detector = MagicDetector()
        self.captioner = JoyCaption4Bit()
        self.records: dict[str, dict[str, Any]] = {}
        self.visited_dirs: set[tuple[int, int]] = set()
        self.pending_symlinks: dict[str, str] = {}
        self.symlink_decisions: dict[str, bool] = {}
        self.cv = threading.Condition()
        self.stop_requested = False
        self.tag_counts = Counter()

    def request_stop(self) -> None:
        with self.cv:
            self.stop_requested = True
            self.cv.notify_all()

    def resolve_symlink(self, path: str, follow: bool) -> None:
        with self.cv:
            self.symlink_decisions[path] = follow
            self.cv.notify_all()

    def _only_unresolved_symlinks_remain(self) -> bool:
        if not self.todo:
            return False
        for item in self.todo:
            key = str(item.path)
            if item.follow_symlink or key not in self.pending_symlinks or key in self.symlink_decisions:
                return False
        return True

    def _ensure_record(self, path: Path, *, follow_symlink: bool = False) -> dict[str, Any]:
        key = str(path)
        existing = self.records.get(key)
        if existing is not None and not follow_symlink:
            return existing

        rec = stat_record(path, self.detector, follow_symlinks=follow_symlink)
        if existing and follow_symlink:
            # Preserve the fact that the filesystem entry itself is a symlink.
            rec["symlink_target"] = existing.get("symlink_target", "")
            rec["extra_metadata"]["followed_symlink"] = True
        self.records[key] = rec
        self.discovered_count.emit(len(self.records))
        self.record_updated.emit(key, rec)
        return rec

    @staticmethod
    def _dir_identity(path: Path) -> tuple[int, int]:
        st = path.stat()
        return int(st.st_dev), int(st.st_ino)

    def _process_image(self, path: Path, rec: dict[str, Any]) -> None:
        self.status.emit("Tagging image with JoyCaption (4-bit)…")
        image = open_image(path)
        tags = self.captioner.tags_for_image(image)
        rec["tags"] = tags
        rec["category"] = "image"
        rec["extra_metadata"].update({"width": image.width, "height": image.height})
        thumb = thumbnail_jpeg(image)
        for t in tags:
            self.tag_counts[t] += 1
        self.tagged.emit(str(path), tags, thumb)

    def _process_video(self, path: Path, rec: dict[str, Any]) -> None:
        self.status.emit("Sampling video frames and tagging with JoyCaption (4-bit)…")
        all_tags: dict[str, str] = {}
        sampled_times: list[float] = []
        thumb = b""
        frame_count = 0

        for ts, frame in sample_video_frames(path, interval_seconds=30):
            if self.stop_requested:
                break
            sampled_times.append(ts)
            frame_count += 1
            if not thumb:
                thumb = thumbnail_jpeg(frame)
            for tag in self.captioner.tags_for_image(frame):
                all_tags.setdefault(tag.casefold(), tag)

        tags = list(all_tags.values())
        rec["tags"] = tags
        rec["category"] = "video"
        rec["extra_metadata"].update({
            "duration_seconds": video_duration_seconds(path),
            "sampled_frame_times_seconds": sampled_times,
            "sampled_frame_count": frame_count,
        })
        for t in tags:
            self.tag_counts[t] += 1
        self.tagged.emit(str(path), tags, thumb)

    def _process_archive(self, path: Path, rec: dict[str, Any]) -> None:
        self.status.emit("Sampling media from archive…")
        sampled = sample_archive(path, sample_size=5)
        all_tags: dict[str, str] = {}
        thumb = b""
        sampled_names: list[str] = list(sampled.member_names)

        try:
            for member_path in sampled.paths:
                if self.stop_requested:
                    break
                try:
                    mime, _ = self.detector.identify(member_path)
                    if is_image(member_path, mime):
                        image = open_image(member_path)
                    elif is_video(member_path, mime):
                        image = first_video_frame(member_path)
                        if image is None:
                            continue
                    else:
                        continue

                    if not thumb:
                        thumb = thumbnail_jpeg(image)
                    for tag in self.captioner.tags_for_image(image):
                        all_tags.setdefault(tag.casefold(), tag)
                except Exception:
                    continue
        finally:
            sampled.cleanup()

        tags = list(all_tags.values())
        rec["tags"] = tags
        rec["category"] = "archive"
        rec["extra_metadata"].update({
            "archive_total_members": sampled.total_members,
            "archive_media_members": sampled.total_media_members,
            "archive_sample_size": len(sampled_names),
            "archive_sampled_members": sampled_names,
        })
        for t in tags:
            self.tag_counts[t] += 1
        if tags:
            self.tagged.emit(str(path), tags, thumb)

    def _process_file(self, path: Path, rec: dict[str, Any]) -> None:
        mime = rec.get("mime_type", "")
        try:
            if is_image(path, mime):
                self._process_image(path, rec)
            elif is_video(path, mime):
                self._process_video(path, rec)
            elif is_archive(path, mime):
                self._process_archive(path, rec)
            else:
                rec["category"] = "file"
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        finally:
            self.record_updated.emit(str(path), rec)

    def run(self) -> None:
        try:
            self.status.emit("Scanning…")
            while True:
                if self.stop_requested:
                    self.status.emit("Stopped.")
                    break

                if not self.todo:
                    break

                # Avoid a busy loop if all remaining FIFO entries are symlinks whose
                # non-modal GUI questions have not yet been answered.
                if self._only_unresolved_symlinks_remain():
                    with self.cv:
                        self.status.emit("Waiting for symlink decisions…")
                        self.cv.wait(timeout=0.25)
                    continue

                item = self.todo.popleft()
                path = item.path
                self.current_path.emit(str(path))

                # Symlinks literally go back to the bottom of the FIFO while the
                # non-modal GUI question is pending. Other queue items keep running.
                if path.is_symlink() and not item.follow_symlink:
                    rec = self._ensure_record(path, follow_symlink=False)
                    key = str(path)
                    if key not in self.pending_symlinks:
                        try:
                            target = os.readlink(path)
                        except Exception:
                            target = ""
                        self.pending_symlinks[key] = target
                        self.symlink_question.emit(key, target)

                    with self.cv:
                        decision = self.symlink_decisions.pop(key, None) if key in self.symlink_decisions else None

                    if decision is None:
                        self.todo.append(item)
                        continue

                    self.pending_symlinks.pop(key, None)
                    rec["extra_metadata"]["symlink_followed"] = bool(decision)
                    self.record_updated.emit(key, rec)
                    if not decision:
                        continue
                    item = QueueItem(path, follow_symlink=True)

                try:
                    if item.follow_symlink:
                        rec = self._ensure_record(path, follow_symlink=True)
                    else:
                        # Directories themselves are not rows in the final "every file" TSV.
                        rec = None if path.is_dir() else self._ensure_record(path)
                except Exception as e:
                    rec = self._ensure_record(path)
                    rec["error"] = f"inspect: {e}"
                    self.record_updated.emit(str(path), rec)
                    continue

                try:
                    is_dir = path.is_dir() if not item.follow_symlink else path.stat().st_mode is not None and path.is_dir()
                except Exception:
                    is_dir = False

                if is_dir:
                    try:
                        ident = self._dir_identity(path)
                        if ident in self.visited_dirs:
                            continue
                        self.visited_dirs.add(ident)

                        # Path.iterdir() includes dot-files/dot-directories.
                        # Appending in enumeration order gives FIFO breadth-first traversal.
                        for child in path.iterdir():
                            self.todo.append(QueueItem(child))
                    except Exception as e:
                        self.status.emit(f"Cannot scan directory: {path} ({e})")
                    continue

                if rec is not None:
                    self._process_file(path, rec)

            self.completed.emit(list(self.records.values()))
        except Exception:
            self.fatal_error.emit(traceback.format_exc())
            self.completed.emit(list(self.records.values()))


TSV_COLUMNS = [
    "filename",
    "tags",
    "exif",
    "path",
    "extension",
    "mime_type",
    "file_type",
    "size_bytes",
    "modified",
    "accessed",
    "created_or_changed",
    "birth_time",
    "permissions",
    "uid",
    "gid",
    "symlink_target",
    "category",
    "extra_metadata",
    "error",
]


def export_tsv(records: list[dict[str, Any]], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=TSV_COLUMNS,
            extrasaction="ignore",
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        for rec in records:
            row = dict(rec)
            row["tags"] = ", ".join(rec.get("tags") or [])
            row["extra_metadata"] = json.dumps(
                rec.get("extra_metadata") or {},
                ensure_ascii=False,
                sort_keys=True,
            )
            writer.writerow(row)
