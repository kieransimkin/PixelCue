from __future__ import annotations

import csv
import heapq
import json
import os
import threading
import time
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
from .model import JoyCaption4Bit, JoyCaptionError
from .face import (
    DeepFaceAnalyzer, DeepFaceUnavailable, FaceResult,
    cluster_embeddings, CLUSTER_INTERVAL_SECONDS,
    deepface_available, deepface_install_command, validate_deepface_runtime,
)


# Files smaller than 500,000 bytes remain in the inventory but are not sent to JoyCaption.
MIN_MEDIA_ANALYSIS_BYTES = 500_000


@dataclass
class QueueItem:
    path: Path
    follow_symlink: bool = False


@dataclass
class MediaJob:
    path: Path
    record: dict[str, Any]
    category: str


@dataclass
class VideoSampleJob:
    path: Path
    record: dict[str, Any]


class ScanWorker(QThread):
    status = Signal(str)
    current_path = Signal(str)
    discovered_count = Signal(int)
    media_queue_count = Signal(int, int)  # waiting, completed
    face_queue_count = Signal(int, int)   # waiting, completed
    video_sample_queue_count = Signal(int, int)
    face_analyzed = Signal(str, object)
    face_clusters_updated = Signal(object)
    tagged = Signal(str, object, bytes)
    record_updated = Signal(str, object)
    symlink_question = Signal(str, str)
    model_status = Signal(str)
    model_progress = Signal(object, object)
    processing_error = Signal(str, str, str)
    fatal_error = Signal(str)
    completed = Signal(object)

    def __init__(self, start_dir: str, parent=None) -> None:
        super().__init__(parent)
        self.start_dir = Path(start_dir)

        # Required filesystem traversal semantics: explicit FIFO todo list.
        self.todo: deque[QueueItem] = deque([QueueItem(self.start_dir)])

        self.detector = MagicDetector()
        self.captioner = JoyCaption4Bit(
            status_callback=self.model_status.emit,
            progress_callback=self.model_progress.emit,
        )

        self.records: dict[str, dict[str, Any]] = {}
        self.records_lock = threading.Lock()

        self.visited_dirs: set[tuple[int, int]] = set()

        self.pending_symlinks: dict[str, str] = {}
        self.symlink_decisions: dict[str, bool] = {}
        self.symlink_cv = threading.Condition()

        self.stop_requested = False
        self.tag_counts = Counter()

        # Model preparation is independent of both scanner and media worker.
        # Individual file inference failures must not disable later files.
        self.model_download_done = threading.Event()
        self.model_download_error: str | None = None
        self.model_download_thread: threading.Thread | None = None

        # Media analysis is queued by file size, largest first.
        # Ties preserve discovery order via a monotonically increasing sequence.
        self.media_jobs: list[tuple[int, int, MediaJob]] = []
        # Sampled videos are deliberately kept out of the normal JoyCaption heap.
        # They are lower priority than every image/archive, regardless of filesize.
        self.video_media_jobs: list[tuple[int, int, MediaJob]] = []
        self.media_job_sequence = 0
        self.media_cv = threading.Condition()
        self.filesystem_discovery_done = False
        self.media_jobs_completed = 0
        self.media_analysis_done = False

        # Video decoding/frame selection is independent of JoyCaption image work.
        self.video_sample_jobs: deque[VideoSampleJob] = deque()
        self.video_sample_cv = threading.Condition()
        self.video_sample_jobs_completed = 0
        self.video_sample_worker_thread: threading.Thread | None = None
        self.video_sampling_done = False
        self.media_worker_thread: threading.Thread | None = None
        self.media_detector: MagicDetector | None = None

        # Independent DeepFace pipeline, fed only by FaceIdentity images.
        self.face_jobs: deque[str] = deque()
        self.face_cv = threading.Condition()
        self.face_results: dict[str, FaceResult] = {}
        self.face_results_lock = threading.Lock()
        self.face_jobs_completed = 0
        self.face_worker_thread: threading.Thread | None = None
        self.face_stop = False
        self.face_disabled_reason: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle / synchronization
    # ------------------------------------------------------------------

    def request_stop(self) -> None:
        self.stop_requested = True
        with self.symlink_cv:
            self.symlink_cv.notify_all()
        with self.media_cv:
            self.media_cv.notify_all()
        with self.face_cv:
            self.face_stop = True
            self.face_cv.notify_all()
        with self.video_sample_cv:
            self.video_sample_cv.notify_all()

    def resolve_symlink(self, path: str, follow: bool) -> None:
        with self.symlink_cv:
            self.symlink_decisions[path] = follow
            self.symlink_cv.notify_all()

    def _emit_media_queue_count(self) -> None:
        with self.media_cv:
            waiting = len(self.media_jobs) + len(self.video_media_jobs)
            completed = self.media_jobs_completed
        self.media_queue_count.emit(waiting, completed)

    @staticmethod
    def _media_job_priority(rec: dict[str, Any]) -> int:
        """Return a max-priority key based on file size.

        Larger files should be processed first, so the heap stores a negative
        size value. Missing/invalid sizes are treated as zero.
        """
        try:
            return -int(rec.get("size_bytes") or 0)
        except (TypeError, ValueError):
            return 0

    def _queue_media_job(
        self,
        path: Path,
        rec: dict[str, Any],
        category: str,
    ) -> None:
        rec["category"] = category
        with self.media_cv:
            priority = self._media_job_priority(rec)
            seq = self.media_job_sequence
            self.media_job_sequence += 1
            target_heap = self.video_media_jobs if category == "video" else self.media_jobs
            heapq.heappush(
                target_heap,
                (priority, seq, MediaJob(path, rec, category)),
            )
            self.media_cv.notify()
        self.record_updated.emit(str(path), rec)
        self._emit_media_queue_count()

    # ------------------------------------------------------------------
    # Records / filesystem metadata
    # ------------------------------------------------------------------

    def _ensure_record(
        self,
        path: Path,
        *,
        follow_symlink: bool = False,
    ) -> dict[str, Any]:
        key = str(path)

        with self.records_lock:
            existing = self.records.get(key)

        if existing is not None and not follow_symlink:
            return existing

        rec = stat_record(path, self.detector, follow_symlinks=follow_symlink)

        if existing and follow_symlink:
            rec["symlink_target"] = existing.get("symlink_target", "")
            rec["extra_metadata"]["followed_symlink"] = True

        with self.records_lock:
            self.records[key] = rec
            count = len(self.records)

        self.discovered_count.emit(count)
        self.record_updated.emit(key, rec)
        return rec

    @staticmethod
    def _dir_identity(path: Path) -> tuple[int, int]:
        st = path.stat()
        return int(st.st_dev), int(st.st_ino)

    def _record_error(
        self,
        path: Path,
        rec: dict[str, Any],
        stage: str,
        exc: BaseException,
        details: str | None = None,
    ) -> None:
        short = f"{stage}: {type(exc).__name__}: {exc}"
        rec["error"] = short
        self.processing_error.emit(
            str(path),
            short,
            details or traceback.format_exc(),
        )
        self.record_updated.emit(str(path), rec)

    # ------------------------------------------------------------------
    # Model preparation
    # ------------------------------------------------------------------

    def _prefetch_model(self) -> None:
        try:
            self.captioner.ensure_downloaded()
        except Exception as e:
            self.model_download_error = str(e)
            self.processing_error.emit(
                "[JoyCaption model]",
                f"JoyCaption model setup failed: {type(e).__name__}: {e}",
                str(e),
            )
            self.model_status.emit(
                "JoyCaption 4-bit: model setup failed — see Errors"
            )
        finally:
            self.model_download_done.set()
            with self.media_cv:
                self.media_cv.notify_all()

    def _start_model_prefetch(self) -> None:
        if self.model_download_thread is not None:
            return

        self.model_status.emit("Starting JoyCaption 4-bit model preparation …")
        self.model_download_thread = threading.Thread(
            target=self._prefetch_model,
            name="pixelcue-joycaption-prefetch",
            daemon=True,
        )
        self.model_download_thread.start()

    def _wait_for_model_prefetch(
        self,
        path: Path,
        rec: dict[str, Any],
    ) -> bool:
        while not self.model_download_done.wait(timeout=0.10):
            if self.stop_requested:
                rec["error"] = (
                    "Tagging cancelled while waiting for JoyCaption model download."
                )
                self.record_updated.emit(str(path), rec)
                return False

        if self.model_download_error is not None:
            rec["error"] = (
                "Tagging skipped because JoyCaption model setup failed: "
                + self.model_download_error.splitlines()[0]
            )
            self.record_updated.emit(str(path), rec)
            return False

        return True

    def _captioning_available(
        self,
        path: Path,
        rec: dict[str, Any],
    ) -> bool:
        # Per-file JoyCaption inference failures do not disable later jobs.
        # Global model-setup failures are already handled by _wait_for_model_prefetch.
        return True

    # ------------------------------------------------------------------
    # DeepFace worker / face clustering
    # ------------------------------------------------------------------

    def _emit_face_queue_count(self) -> None:
        with self.face_cv:
            waiting = len(self.face_jobs)
            completed = self.face_jobs_completed
        self.face_queue_count.emit(waiting, completed)

    def _queue_face_analysis(self, path: Path) -> None:
        if self.face_disabled_reason is not None:
            return
        key = str(path)
        with self.face_results_lock:
            if key in self.face_results:
                return
        with self.face_cv:
            if key not in self.face_jobs:
                self.face_jobs.append(key)
                self.face_cv.notify()
        self._emit_face_queue_count()

    def _emit_face_clusters(self) -> None:
        with self.face_results_lock:
            results = list(self.face_results.values())
        clusters = cluster_embeddings(results)
        self.face_clusters_updated.emit([
            {
                "cluster_id": c["cluster_id"],
                "representative_path": c["representative_path"],
                "paths": list(c["paths"]),
                "size": len(c["paths"]),
            }
            for c in clusters
        ])

    def _face_worker(self) -> None:
        face_ok, face_reason = validate_deepface_runtime()
        if not face_ok:
            self.face_disabled_reason = face_reason
            self.processing_error.emit(
                "[DeepFace]",
                self.face_disabled_reason,
                self.face_disabled_reason,
            )
            self.status.emit(
                "DeepFace disabled for this scan; JoyCaption/filesystem processing continues."
            )
            return

        analyzer = DeepFaceAnalyzer(status_callback=self.status.emit)
        last_cluster = time.monotonic()

        while True:
            path = None
            with self.face_cv:
                if not self.face_jobs and not self.face_stop:
                    self.face_cv.wait(timeout=1.0)
                if self.face_stop:
                    return
                if self.face_jobs:
                    path = self.face_jobs.popleft()

            if path is not None:
                try:
                    result = analyzer.analyze(path)
                    with self.face_results_lock:
                        self.face_results[path] = result

                    rec = self.records.get(path)
                    if rec is not None:
                        rec["extra_metadata"]["deepface"] = result.attributes
                        rec["extra_metadata"]["face_embedding"] = result.embedding
                        self.record_updated.emit(path, rec)

                    self.face_analyzed.emit(path, result.attributes)
                except DeepFaceUnavailable as e:
                    self.face_disabled_reason = str(e)
                    self.processing_error.emit(
                        "[DeepFace]",
                        str(e).splitlines()[0],
                        str(e),
                    )
                    self.status.emit(
                        "DeepFace disabled for this scan; JoyCaption/filesystem processing continues."
                    )
                    with self.face_cv:
                        self.face_jobs.clear()
                    return
                except Exception as e:
                    rec = self.records.get(path)
                    if rec is not None:
                        self._record_error(Path(path), rec, "DeepFace", e)
                finally:
                    with self.face_cv:
                        self.face_jobs_completed += 1
                    self._emit_face_queue_count()

            now = time.monotonic()
            if now - last_cluster >= CLUSTER_INTERVAL_SECONDS:
                self._emit_face_clusters()
                last_cluster = now

            # Once filesystem/media work has ended and the face queue is empty,
            # publish one final clustering result and finish.
            with self.face_cv:
                done = self.media_analysis_done and not self.face_jobs
            if done:
                self._emit_face_clusters()
                return

    def _start_face_worker(self) -> None:
        if self.face_worker_thread is not None:
            return
        self.face_worker_thread = threading.Thread(
            target=self._face_worker,
            name="pixelcue-deepface",
            daemon=True,
        )
        self.face_worker_thread.start()

    # ------------------------------------------------------------------
    # Video frame sampling worker
    # ------------------------------------------------------------------

    def _emit_video_sample_queue_count(self) -> None:
        with self.video_sample_cv:
            waiting = len(self.video_sample_jobs)
            completed = self.video_sample_jobs_completed
        self.video_sample_queue_count.emit(waiting, completed)

    def _queue_video_sampling(self, path: Path, rec: dict[str, Any]) -> None:
        with self.video_sample_cv:
            self.video_sample_jobs.append(VideoSampleJob(path, rec))
            self.video_sample_cv.notify()
        self._emit_video_sample_queue_count()

    def _video_sample_worker(self) -> None:
        while True:
            job = None
            with self.video_sample_cv:
                while (
                    not self.video_sample_jobs
                    and not self.filesystem_discovery_done
                    and not self.stop_requested
                ):
                    self.video_sample_cv.wait(timeout=0.25)

                if self.stop_requested:
                    return

                if self.video_sample_jobs:
                    job = self.video_sample_jobs.popleft()
                elif self.filesystem_discovery_done:
                    self.video_sampling_done = True
                    with self.media_cv:
                        self.media_cv.notify_all()
                    return

            if job is None:
                continue

            try:
                frames = list(sample_video_frames(job.path, max_frames=5))
                job.record["extra_metadata"]["duration_seconds"] = video_duration_seconds(job.path)
                job.record["extra_metadata"]["sampled_frame_times_seconds"] = [
                    ts for ts, _frame in frames
                ]
                job.record["extra_metadata"]["sampled_frame_count"] = len(frames)

                # Attach sampled frames transiently to the MediaJob rather than
                # making the JoyCaption worker perform video decoding/seeking.
                job.record["_pixelcue_sampled_frames"] = frames
                self._queue_media_job(job.path, job.record, "video")
            except Exception as e:
                self._record_error(job.path, job.record, "Video sampling", e)
            finally:
                with self.video_sample_cv:
                    self.video_sample_jobs_completed += 1
                self._emit_video_sample_queue_count()

    def _start_video_sample_worker(self) -> None:
        if self.video_sample_worker_thread is not None:
            return
        self.video_sample_worker_thread = threading.Thread(
            target=self._video_sample_worker,
            name="pixelcue-video-sampler",
            daemon=True,
        )
        self.video_sample_worker_thread.start()

    # ------------------------------------------------------------------
    # Media worker — deliberately separate from filesystem traversal
    # ------------------------------------------------------------------

    def _process_image(self, path: Path, rec: dict[str, Any]) -> None:
        if not self._wait_for_model_prefetch(path, rec):
            return
        if not self._captioning_available(path, rec):
            return

        self.status.emit(f"Tagging image: {path}")
        image = open_image(path)

        try:
            tags = self.captioner.tags_for_image(image)
        except JoyCaptionError as e:
            raise

        rec["tags"] = tags
        rec["extra_metadata"].update({
            "width": image.width,
            "height": image.height,
        })

        thumb = thumbnail_jpeg(image)
        for tag in tags:
            self.tag_counts[tag] += 1
        self.tagged.emit(str(path), tags, thumb)

        if any(str(tag).casefold() == "faceidentity" for tag in tags):
            self._queue_face_analysis(path)

    def _process_video(self, path: Path, rec: dict[str, Any]) -> None:
        if not self._wait_for_model_prefetch(path, rec):
            return
        if not self._captioning_available(path, rec):
            return

        self.status.emit(f"Tagging pre-sampled video frames: {path}")
        all_tags: dict[str, str] = {}
        thumb = b""

        frames = rec.pop("_pixelcue_sampled_frames", [])
        for ts, frame in frames:
            if self.stop_requested:
                break
            if not thumb:
                thumb = thumbnail_jpeg(frame)
            frame_tags = self.captioner.tags_for_image(frame)
            for tag in frame_tags:
                all_tags.setdefault(tag.casefold(), tag)

        tags = list(all_tags.values())
        rec["tags"] = tags
        for tag in tags:
            self.tag_counts[tag] += 1
        self.tagged.emit(str(path), tags, thumb)

    def _process_archive(self, path: Path, rec: dict[str, Any]) -> None:
        if not self._wait_for_model_prefetch(path, rec):
            return
        if not self._captioning_available(path, rec):
            return

        self.status.emit(f"Sampling/tagging archive: {path}")
        sampled = sample_archive(path, sample_size=5, mime=str(rec.get("mime_type", "")))
        all_tags: dict[str, str] = {}
        thumb = b""
        member_errors: list[str] = []

        try:
            for member_path in sampled.paths:
                if self.stop_requested:
                    break

                try:
                    detector = self.media_detector or self.detector
                    mime, _ = detector.identify(member_path)

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

                    try:
                        member_tags = self.captioner.tags_for_image(image)
                    except JoyCaptionError as e:
                                    raise

                    for tag in member_tags:
                        all_tags.setdefault(tag.casefold(), tag)

                except JoyCaptionError:
                    raise
                except Exception as e:
                    member_errors.append(
                        f"{member_path.name}: {type(e).__name__}: {e}"
                    )
        finally:
            sampled.cleanup()

        tags = list(all_tags.values())
        rec["tags"] = tags
        rec["extra_metadata"].update({
            "archive_total_members": sampled.total_members,
            "archive_media_members": sampled.total_media_members,
            "archive_sample_size": len(sampled.member_names),
            "archive_sampled_members": list(sampled.member_names),
            "archive_member_errors": member_errors,
        })

        for tag in tags:
            self.tag_counts[tag] += 1

        if tags:
            self.tagged.emit(str(path), tags, thumb)

    def _process_media_job(self, job: MediaJob) -> None:
        try:
            if job.category == "image":
                self._process_image(job.path, job.record)
            elif job.category == "video":
                self._process_video(job.path, job.record)
            elif job.category == "archive":
                self._process_archive(job.path, job.record)
        except JoyCaptionError as e:
            self._record_error(
                job.path,
                job.record,
                "JoyCaption",
                e,
                details=str(e),
            )
            self.status.emit(
                "JoyCaption failed for this file; PixelCue will continue with the next queued media item. "
                "Filesystem discovery is unaffected."
            )
        except Exception as e:
            self._record_error(job.path, job.record, "Processing", e)
        finally:
            self.record_updated.emit(str(job.path), job.record)

    def _media_worker(self) -> None:
        # libmagic wrappers are kept thread-local: filesystem metadata and
        # archive-member inspection can now happen concurrently.
        self.media_detector = MagicDetector()

        while True:
            job = None
            with self.media_cv:
                while not self.stop_requested:
                    # Ordinary images/archives always win. This preserves the
                    # largest-first policy within that class without allowing a
                    # multi-GB video to jump ahead merely because it is large.
                    if self.media_jobs:
                        _priority, _seq, job = heapq.heappop(self.media_jobs)
                        break

                    # Do not spend JoyCaption time on video until filesystem
                    # discovery has completed, so newly discovered normal images
                    # can never be blocked behind video inference.
                    if self.filesystem_discovery_done and self.video_media_jobs:
                        _priority, _seq, job = heapq.heappop(self.video_media_jobs)
                        break

                    if (
                        self.filesystem_discovery_done
                        and self.video_sampling_done
                        and not self.media_jobs
                        and not self.video_media_jobs
                    ):
                        return

                    self.media_cv.wait(timeout=0.25)

                if self.stop_requested:
                    return

            if job is None:
                continue

            self._emit_media_queue_count()
            self._process_media_job(job)

            with self.media_cv:
                self.media_jobs_completed += 1
            self._emit_media_queue_count()

    def _start_media_worker(self) -> None:
        if self.media_worker_thread is not None:
            return

        self.media_worker_thread = threading.Thread(
            target=self._media_worker,
            name="pixelcue-media-tagger",
            daemon=True,
        )
        self.media_worker_thread.start()

    # ------------------------------------------------------------------
    # Filesystem walker
    # ------------------------------------------------------------------

    def _only_unresolved_symlinks_remain(self) -> bool:
        if not self.todo:
            return False

        for item in self.todo:
            key = str(item.path)
            if (
                item.follow_symlink
                or key not in self.pending_symlinks
                or key in self.symlink_decisions
            ):
                return False

        return True

    def _classify_or_queue_file(
        self,
        path: Path,
        rec: dict[str, Any],
    ) -> None:
        """Queue libmagic-verified media only when it is at least 500,000 bytes."""
        mime = str(rec.get("mime_type", "") or "").strip().lower()

        if is_image(path, mime):
            category = "image"
        elif is_video(path, mime):
            category = "video"
        elif is_archive(path, mime):
            category = "archive"
        else:
            rec["category"] = "file"
            self.record_updated.emit(str(path), rec)
            return

        try:
            size_bytes = int(rec.get("size_bytes") or 0)
        except (TypeError, ValueError):
            size_bytes = 0

        if size_bytes < MIN_MEDIA_ANALYSIS_BYTES:
            rec["category"] = category
            rec["extra_metadata"]["joycaption_skipped"] = True
            rec["extra_metadata"]["joycaption_skip_reason"] = (
                f"file smaller than {MIN_MEDIA_ANALYSIS_BYTES} bytes"
            )
            self.record_updated.emit(str(path), rec)
            return

        if category == "video":
            self._queue_video_sampling(path, rec)
        else:
            self._queue_media_job(path, rec, category)

    def _walk_filesystem(self) -> None:
        while True:
            if self.stop_requested:
                return

            if not self.todo:
                return

            if self._only_unresolved_symlinks_remain():
                with self.symlink_cv:
                    self.status.emit(
                        "Filesystem discovery waiting only for symlink decisions…"
                    )
                    self.symlink_cv.wait(timeout=0.25)
                continue

            item = self.todo.popleft()
            path = item.path

            # This signal now genuinely represents discovery, not media analysis.
            self.current_path.emit(str(path))

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

                with self.symlink_cv:
                    decision = self.symlink_decisions.pop(key, None)

                if decision is None:
                    # Exact requested semantics: move unresolved symlink to bottom.
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
                    rec = None if path.is_dir() else self._ensure_record(path)
            except Exception as e:
                try:
                    rec = self._ensure_record(path)
                    rec["error"] = f"inspect: {e}"
                    self.record_updated.emit(str(path), rec)
                except Exception:
                    pass
                continue

            try:
                is_dir = path.is_dir()
            except Exception:
                is_dir = False

            if is_dir:
                try:
                    ident = self._dir_identity(path)
                    if ident in self.visited_dirs:
                        continue
                    self.visited_dirs.add(ident)

                    # Includes hidden entries. FIFO breadth-first behaviour is
                    # preserved by appending every child to the right.
                    for child in path.iterdir():
                        self.todo.append(QueueItem(child))
                except Exception as e:
                    self.status.emit(
                        f"Cannot scan directory: {path} ({e})"
                    )
                continue

            if rec is not None:
                self._classify_or_queue_file(path, rec)

    # ------------------------------------------------------------------
    # QThread entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        try:
            self._start_model_prefetch()
            self._start_video_sample_worker()
            self._start_media_worker()
            self._start_face_worker()

            self.status.emit(
                "Filesystem discovery running independently of JoyCaption preparation…"
            )

            # This method never waits for model/tagging work.
            self._walk_filesystem()

            self.filesystem_discovery_done = True
            with self.video_sample_cv:
                self.video_sample_cv.notify_all()
            with self.media_cv:
                queued = len(self.media_jobs) + len(self.video_media_jobs)
                completed = self.media_jobs_completed
                self.media_cv.notify_all()

            if self.stop_requested:
                self.status.emit("Stopped.")
            elif queued:
                self.status.emit(
                    f"Filesystem discovery complete; "
                    f"waiting for {queued} queued media item(s) to be tagged…"
                )
            else:
                self.status.emit(
                    "Filesystem discovery complete; waiting for media worker…"
                )

            # Video sampling is independent and may still enqueue media jobs.
            if self.video_sample_worker_thread is not None:
                while self.video_sample_worker_thread.is_alive():
                    if self.stop_requested:
                        break
                    self.video_sample_worker_thread.join(timeout=0.25)

            # Final export must include completed tags, so completion waits for the
            # media queue — but discovery itself is already completely finished.
            if self.media_worker_thread is not None:
                while self.media_worker_thread.is_alive():
                    if self.stop_requested:
                        break
                    self.media_worker_thread.join(timeout=0.25)

            self.media_analysis_done = True

            # Media tagging may have queued DeepFace jobs. Let that independent
            # pipeline finish so the final TSV includes face attributes/vectors.
            with self.face_cv:
                self.face_cv.notify_all()
            if self.face_worker_thread is not None:
                while self.face_worker_thread.is_alive():
                    if self.stop_requested:
                        break
                    self.face_worker_thread.join(timeout=0.25)

            with self.records_lock:
                final_records = list(self.records.values())

            self.completed.emit(final_records)

        except Exception:
            self.fatal_error.emit(traceback.format_exc())
            with self.records_lock:
                final_records = list(self.records.values())
            self.completed.emit(final_records)


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
