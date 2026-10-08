from __future__ import annotations

import csv
import heapq
import json
import multiprocessing as mp
import os
import queue as queue_module
import threading
import time
import traceback
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PySide6.QtCore import QThread, Signal

from .archives import is_archive, sample_archive
from .os_thumbnail import os_cached_thumbnail_jpeg
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
from .model import JoyCaptionError, create_image_tagger
from .cluster_process import face_cluster_process_main
from .logging_utils import log_event, log_exception
from .analysis_store import AnalysisStore
from .face import (
    DeepFaceAnalyzer, DeepFaceUnavailable, FaceNotConfirmed, FaceResult,
    cluster_embeddings, central_cluster_members,
    cluster_attributes_confident, aggregate_cluster_attributes,
    CLUSTER_INTERVAL_SECONDS, DEEPFACE_MODEL, DEEPFACE_DETECTOR, CELEBRITY_DB,
    deepface_available, deepface_install_command, validate_deepface_runtime,
)


# Files smaller than 500,000 bytes remain in the inventory but are not sent to JoyCaption.
MIN_MEDIA_ANALYSIS_BYTES = 500_000
MEDIA_CACHE_VERSION = "vlm-tags-v1"
FACE_EMBEDDING_CACHE_VERSION = "face-embedding-v1"
FACE_ATTRIBUTE_CACHE_VERSION = "face-attributes-v1"


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

    def __init__(self, start_dir: str, model_profile_id: str = "joycaption", parent=None) -> None:
        super().__init__(parent)
        self.start_dir = Path(start_dir)
        self.model_profile_id = model_profile_id

        # Required filesystem traversal semantics: explicit FIFO todo list.
        self.todo: deque[QueueItem] = deque([QueueItem(self.start_dir)])

        self.detector = MagicDetector()
        self.captioner = create_image_tagger(
            self.model_profile_id,
            status_callback=self.model_status.emit,
            progress_callback=self.model_progress.emit,
        )
        self.analysis_store = AnalysisStore()
        self.media_cache_engine = (
            f"{MEDIA_CACHE_VERSION}:{self.model_profile_id}"
        )
        self.face_embedding_cache_engine = (
            f"{FACE_EMBEDDING_CACHE_VERSION}:{DEEPFACE_MODEL}:{DEEPFACE_DETECTOR}"
        )
        celebrity_key = os.path.normcase(os.path.abspath(CELEBRITY_DB)) if CELEBRITY_DB else "none"
        self.face_attribute_cache_engine = (
            f"{FACE_ATTRIBUTE_CACHE_VERSION}:{DEEPFACE_MODEL}:{DEEPFACE_DETECTOR}:{celebrity_key}"
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
        self.model_prefetch_lock = threading.Lock()

        # Images use an explicit descending-by-filesize list. Each newly
        # discovered image is inserted at the first position whose queued item
        # is smaller, exactly preserving "largest discovered image is next".
        # Equal-sized images retain discovery order.
        self.image_jobs: list[tuple[int, int, MediaJob]] = []

        # Archives stay separate from the image list, so archive ordering cannot
        # disturb image-to-image filesize ordering.
        self.archive_media_jobs: list[tuple[int, int, MediaJob]] = []

        # Sampled videos remain lower priority than images and archives.
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
        self.face_cluster_summaries: dict[str, dict[str, Any]] = {}
        self.face_results_lock = threading.Lock()
        self.face_jobs_completed = 0
        self.face_worker_thread: threading.Thread | None = None
        self.face_stop = False
        self.face_disabled_reason: str | None = None

        # CPU-heavy embedding clustering lives in a separate process so it can
        # never hold the Qt/main-process GIL. The subprocess keeps its own
        # embedding state; only new embeddings cross IPC on each ~10s update.
        self.cluster_context = mp.get_context("spawn")
        self.cluster_request_queue = self.cluster_context.Queue(maxsize=2)
        self.cluster_result_queue = self.cluster_context.Queue(maxsize=2)
        self.cluster_process = None
        self.cluster_job_id = 0
        self.cluster_job_inflight = False
        self.cluster_sent_paths: set[str] = set()
        self.latest_cluster_payload: list[dict[str, Any]] = []

        # Throttle UI-facing discovery progress signals; per-file state remains
        # in the worker and is delivered in full at completion.
        self._last_current_path_emit = 0.0
        self._last_discovered_count_emit = 0.0

    # ------------------------------------------------------------------
    # Lifecycle / synchronization
    # ------------------------------------------------------------------

    def _run_logged_worker(self, worker_name: str, target) -> None:
        log_event(worker_name, "start", "worker started")
        try:
            target()
        except BaseException as exc:
            log_exception(worker_name, "failure", exc)
            log_event(worker_name, "finish", f"{type(exc).__name__}: {exc}", level="ERROR", success=False)
            raise
        else:
            log_event(worker_name, "finish", "worker completed", success=True)

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
            waiting = (
                len(self.image_jobs)
                + len(self.archive_media_jobs)
                + len(self.video_media_jobs)
            )
            completed = self.media_jobs_completed
        self.media_queue_count.emit(waiting, completed)

    @staticmethod
    def _media_job_size(rec: dict[str, Any]) -> int:
        try:
            return int(rec.get("size_bytes") or 0)
        except (TypeError, ValueError):
            return 0

    def _insert_image_job_sorted(
        self,
        job: MediaJob,
        size_bytes: int,
        seq: int,
    ) -> int:
        """Insert image into descending filesize order.

        Compare against the current first item. If the new image is larger, it
        becomes index 0 (the next candidate). Otherwise walk down until the
        first queued item that is smaller, then insert immediately before it.
        Equal sizes retain discovery order by inserting after existing equals.
        """
        insert_at = 0
        while insert_at < len(self.image_jobs):
            queued_size, _queued_seq, _queued_job = self.image_jobs[insert_at]
            if size_bytes > queued_size:
                break
            insert_at += 1

        self.image_jobs.insert(
            insert_at,
            (size_bytes, seq, job),
        )
        return insert_at

    def _queue_media_job(
        self,
        path: Path,
        rec: dict[str, Any],
        category: str,
    ) -> None:
        rec["category"] = category

        with self.media_cv:
            size_bytes = self._media_job_size(rec)
            seq = self.media_job_sequence
            self.media_job_sequence += 1
            job = MediaJob(path, rec, category)

            if category == "image":
                insert_at = self._insert_image_job_sorted(
                    job,
                    size_bytes,
                    seq,
                )
                log_event(
                    "vlm-media-queue",
                    "image-insert",
                    "image inserted into descending filesize queue",
                    path=str(path),
                    size_bytes=size_bytes,
                    queue_position=insert_at,
                    queue_length=len(self.image_jobs),
                )

            elif category == "archive":
                # Archives remain size-prioritised, but in their own queue.
                heapq.heappush(
                    self.archive_media_jobs,
                    (-size_bytes, seq, job),
                )

            elif category == "video":
                heapq.heappush(
                    self.video_media_jobs,
                    (-size_bytes, seq, job),
                )

            else:
                raise ValueError(f"Unsupported media queue category: {category}")

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

        now = time.monotonic()
        if count <= 10 or now - self._last_discovered_count_emit >= 0.10:
            self.discovered_count.emit(count)
            self._last_discovered_count_emit = now
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
    # Persistent analysis cache
    # ------------------------------------------------------------------

    def _persist_media_analysis(
        self,
        path: Path,
        rec: dict[str, Any],
    ) -> None:
        payload = {
            "category": rec.get("category"),
            "tags": list(rec.get("tags") or []),
            "extra_metadata": dict(rec.get("extra_metadata") or {}),
        }
        # Never attempt to serialize transient in-memory video frames.
        payload["extra_metadata"].pop("_pixelcue_sampled_frames", None)
        self.analysis_store.save(
            path,
            kind="media_tags",
            engine=self.media_cache_engine,
            payload=payload,
        )

    def _restore_face_analysis_from_cache(
        self,
        path: Path,
        rec: dict[str, Any],
    ) -> bool:
        payload = self.analysis_store.load(
            path,
            kind="face_embedding",
            engine=self.face_embedding_cache_engine,
        )
        if payload is None:
            return False

        status = str(payload.get("status") or "ok")
        if status == "face_not_confirmed":
            rec["extra_metadata"]["deepface"] = {
                "status": "face_not_confirmed",
                "reason": payload.get("reason", "restored cached result"),
                "analysis_cache": "restored",
            }
            self.record_updated.emit(str(path), rec)
            return True

        embedding = payload.get("embedding") or []
        attributes = dict(payload.get("attributes") or {})
        if not embedding:
            return False

        attributes["analysis_cache"] = "restored"
        result = FaceResult(
            str(path),
            [float(value) for value in embedding],
            attributes,
            payload.get("celebrity_lookalike"),
        )
        with self.face_results_lock:
            self.face_results[str(path)] = result

        rec["extra_metadata"]["deepface"] = attributes
        rec["extra_metadata"]["face_embedding"] = result.embedding
        self.record_updated.emit(str(path), rec)
        self.face_analyzed.emit(str(path), attributes)
        log_event(
            "analysis-store",
            "face-restore",
            "restored cached face embedding",
            path=str(path),
        )
        return True

    def _restore_or_queue_face_analysis(
        self,
        path: Path,
        rec: dict[str, Any],
    ) -> None:
        if self._restore_face_analysis_from_cache(path, rec):
            return
        self._queue_face_analysis(path)

    def _restore_media_analysis(
        self,
        path: Path,
        rec: dict[str, Any],
        category: str,
    ) -> bool:
        payload = self.analysis_store.load(
            path,
            kind="media_tags",
            engine=self.media_cache_engine,
        )
        if payload is None:
            return False

        tags = [str(tag) for tag in payload.get("tags") or []]
        rec["category"] = str(payload.get("category") or category)
        rec["tags"] = tags
        cached_extra = dict(payload.get("extra_metadata") or {})
        cached_extra.pop("_pixelcue_sampled_frames", None)
        rec["extra_metadata"].update(cached_extra)
        rec["extra_metadata"]["analysis_cache"] = {
            "media": "restored",
            "engine": self.media_cache_engine,
        }

        thumb = b""
        if category in {"image", "video"}:
            thumb = os_cached_thumbnail_jpeg(path) or b""

        if category == "image" and not thumb:
            try:
                thumb = thumbnail_jpeg(open_image(path))
            except Exception:
                thumb = b""

        for tag in tags:
            self.tag_counts[tag] += 1
        self.tagged.emit(str(path), tags, thumb)
        self.record_updated.emit(str(path), rec)
        log_event(
            "analysis-store",
            "media-restore",
            "restored cached VLM/media analysis",
            path=str(path),
            category=category,
            tags=len(tags),
            engine=self.media_cache_engine,
        )

        if category == "image" and any(
            str(tag).casefold() == "faceidentity" for tag in tags
        ):
            self._restore_or_queue_face_analysis(path, rec)

        return True

    # ------------------------------------------------------------------
    # Model preparation
    # ------------------------------------------------------------------

    def _prefetch_model(self) -> None:
        try:
            self.captioner.ensure_downloaded()
            log_event("vlm-model-prefetch", "success", "model/cache preparation succeeded")
        except Exception as e:
            log_exception("vlm-model-prefetch", "setup-failure", e)
            self.model_download_error = str(e)
            self.processing_error.emit(
                "[JoyCaption model]",
                f"VLM model setup failed: {type(e).__name__}: {e}",
                str(e),
            )
            self.model_status.emit(
                "VLM model setup failed — see Errors"
            )
        finally:
            self.model_download_done.set()
            with self.media_cv:
                self.media_cv.notify_all()

    def _start_model_prefetch(self) -> None:
        with self.model_prefetch_lock:
            if self.model_download_thread is not None:
                return

            self.model_status.emit(
                "Starting VLM model preparation for first uncached media …"
            )
            self.model_download_thread = threading.Thread(
                target=lambda: self._run_logged_worker(
                    "vlm-model-prefetch", self._prefetch_model
                ),
                name="pixelcue-joycaption-prefetch",
                daemon=True,
            )
            self.model_download_thread.start()

    def _wait_for_model_prefetch(
        self,
        path: Path,
        rec: dict[str, Any],
    ) -> bool:
        self._start_model_prefetch()
        while not self.model_download_done.wait(timeout=0.10):
            if self.stop_requested:
                rec["error"] = (
                    "Tagging cancelled while waiting for JoyCaption model download."
                )
                self.record_updated.emit(str(path), rec)
                return False

        if self.model_download_error is not None:
            rec["error"] = (
                "Tagging skipped because VLM model setup failed: "
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

    def _start_cluster_process(self) -> None:
        if self.cluster_process is not None:
            return
        log_event("cluster-manager", "start", "launching clustering subprocess")
        self.cluster_process = self.cluster_context.Process(
            target=face_cluster_process_main,
            args=(self.cluster_request_queue, self.cluster_result_queue),
            name="pixelcue-face-cluster",
            daemon=True,
        )
        self.cluster_process.start()
        log_event(
            "cluster-manager",
            "started",
            "clustering subprocess running",
            child_pid=self.cluster_process.pid,
        )

    def _stop_cluster_process(self, reason: str = "scan complete") -> None:
        process = self.cluster_process
        if process is None:
            return
        try:
            self.cluster_request_queue.put_nowait({"command": "stop", "reason": reason})
        except Exception:
            pass
        process.join(timeout=5.0)
        if process.is_alive():
            log_event(
                "cluster-manager",
                "terminate",
                "clustering subprocess did not stop in time",
                level="WARNING",
                child_pid=process.pid,
            )
            process.terminate()
            process.join(timeout=2.0)
        log_event(
            "cluster-manager",
            "finish",
            reason,
            success=process.exitcode == 0,
            exitcode=process.exitcode,
        )
        self.cluster_process = None

    def _serialize_new_face_embeddings(self) -> list[dict[str, Any]]:
        with self.face_results_lock:
            new = [
                {"path": path, "embedding": list(result.embedding)}
                for path, result in self.face_results.items()
                if path not in self.cluster_sent_paths
            ]
        return new

    def _submit_face_cluster_job(self, *, force: bool = False) -> bool:
        if self.cluster_process is None or not self.cluster_process.is_alive():
            return False
        if self.cluster_job_inflight:
            return False
        items = self._serialize_new_face_embeddings()
        if not items and not force:
            return False
        if not items and force and self.latest_cluster_payload:
            return False

        self.cluster_job_id += 1
        message = {
            "command": "cluster",
            "job_id": self.cluster_job_id,
            "items": items,
        }
        try:
            self.cluster_request_queue.put_nowait(message)
        except queue_module.Full:
            return False
        for item in items:
            self.cluster_sent_paths.add(str(item["path"]))
        self.cluster_job_inflight = True
        log_event(
            "cluster-manager",
            "job-submit",
            "submitted face embeddings for clustering",
            job_id=self.cluster_job_id,
            new_embeddings=len(items),
        )
        return True

    def _publish_cluster_payload(self, payload: list[dict[str, Any]]) -> None:
        enriched = []
        for cluster in payload:
            item = dict(cluster)
            representative = str(item.get("representative_path", ""))
            item["summary"] = self.face_cluster_summaries.get(representative)
            item["size"] = int(item.get("size", len(item.get("paths", []))))
            enriched.append(item)
        self.face_clusters_updated.emit(enriched)

    def _poll_face_cluster_results(self) -> bool:
        changed = False
        while True:
            try:
                result = self.cluster_result_queue.get_nowait()
            except queue_module.Empty:
                break
            self.cluster_job_inflight = False
            if result.get("ok"):
                self.latest_cluster_payload = list(result.get("clusters") or [])
                self._publish_cluster_payload(self.latest_cluster_payload)
                changed = True
            else:
                log_event(
                    "cluster-manager",
                    "job-failure",
                    str(result.get("error") or "unknown clustering failure"),
                    level="ERROR",
                    job_id=result.get("job_id"),
                )
        return changed

    def _wait_for_cluster_result(self, timeout: float = 120.0) -> bool:
        deadline = time.monotonic() + timeout
        while self.cluster_job_inflight and time.monotonic() < deadline:
            try:
                result = self.cluster_result_queue.get(timeout=0.25)
            except queue_module.Empty:
                if self.stop_requested:
                    return False
                continue
            self.cluster_job_inflight = False
            if result.get("ok"):
                self.latest_cluster_payload = list(result.get("clusters") or [])
                self._publish_cluster_payload(self.latest_cluster_payload)
                return True
            log_event(
                "cluster-manager",
                "job-failure",
                str(result.get("error") or "unknown clustering failure"),
                level="ERROR",
                job_id=result.get("job_id"),
            )
            return False
        return not self.cluster_job_inflight

    def _final_face_clusters(self) -> list[dict[str, Any]]:
        # Finish any periodic job, then submit embeddings that arrived since it.
        if self.cluster_job_inflight and not self._wait_for_cluster_result():
            return []
        self._submit_face_cluster_job(force=True)
        if self.cluster_job_inflight and not self._wait_for_cluster_result():
            return []

        with self.face_results_lock:
            by_path = dict(self.face_results)
        hydrated = []
        for raw in self.latest_cluster_payload:
            members = [by_path[path] for path in raw.get("paths", []) if path in by_path]
            if not members:
                continue
            cluster = dict(raw)
            cluster["members"] = members
            cluster["centroid"] = list(raw.get("centroid") or [])
            hydrated.append(cluster)
        return hydrated

    def _emit_face_clusters(
        self,
        clusters: list[dict[str, Any]] | None = None,
    ) -> None:
        if clusters is None:
            self._publish_cluster_payload(self.latest_cluster_payload)
            return
        payload = []
        for c in clusters:
            payload.append({
                "cluster_id": int(c["cluster_id"]),
                "centroid": list(c.get("centroid") or []),
                "representative_path": str(c["representative_path"]),
                "paths": list(c["paths"]),
                "size": len(c["paths"]),
            })
        self.latest_cluster_payload = payload
        self._publish_cluster_payload(payload)

    def _finalize_face_clusters(
        self,
        analyzer: DeepFaceAnalyzer,
        clusters: list[dict[str, Any]],
    ) -> None:
        """Run expensive demographic/celebrity work only on final clusters."""
        if not clusters:
            self._emit_face_clusters(clusters)
            return

        self.status.emit(
            f"Final face-cluster analysis: {len(clusters)} cluster(s)"
        )

        for cluster in clusters:
            if self.stop_requested:
                break

            representative = cluster["representative_path"]
            candidates = central_cluster_members(cluster, limit=3)
            samples: list[dict[str, Any]] = []
            sampled_paths: list[str] = []
            sample_errors: list[str] = []
            confident = False

            for member in candidates:
                if self.stop_requested:
                    break

                try:
                    cached = self.analysis_store.load(
                        member.path,
                        kind="face_attributes",
                        engine=self.face_attribute_cache_engine,
                    )
                    if cached is not None:
                        if str(cached.get("status") or "ok") == "face_not_confirmed":
                            sample_errors.append(
                                f"{member.path}: cached face attributes not confirmed: "
                                f"{cached.get('reason', 'unknown reason')}"
                            )
                            continue
                        attrs = dict(cached.get("attributes") or {})
                        attrs["analysis_cache"] = "restored"
                        log_event(
                            "analysis-store",
                            "face-attributes-restore",
                            "restored cached cluster attribute analysis",
                            path=member.path,
                        )
                    else:
                        attrs = analyzer.analyze_attributes(
                            member.path,
                            preferred_detector=member.attributes.get("detector_backend"),
                        )
                        self.analysis_store.save(
                            member.path,
                            kind="face_attributes",
                            engine=self.face_attribute_cache_engine,
                            payload={"status": "ok", "attributes": attrs},
                        )

                    samples.append(attrs)
                    sampled_paths.append(member.path)

                    # We need two agreeing samples to establish cross-image
                    # confidence. If they agree, skip the third.
                    if len(samples) >= 2 and cluster_attributes_confident(samples):
                        confident = True
                        break
                except FaceNotConfirmed as e:
                    self.analysis_store.save(
                        member.path,
                        kind="face_attributes",
                        engine=self.face_attribute_cache_engine,
                        payload={"status": "face_not_confirmed", "reason": str(e)},
                    )
                    sample_errors.append(f"{member.path}: {e}")
                except Exception as e:
                    sample_errors.append(
                        f"{member.path}: {type(e).__name__}: {e}"
                    )

            summary = aggregate_cluster_attributes(
                samples,
                sampled_paths,
                confident,
            )
            summary.update({
                "cluster_id": cluster["cluster_id"],
                "cluster_size": len(cluster["paths"]),
                "representative_path": representative,
                "candidate_paths": [member.path for member in candidates],
                "sample_errors": sample_errors,
                "analysis_stage": "final_cluster",
            })
            self.face_cluster_summaries[representative] = summary

            for member in cluster["members"]:
                rec = self.records.get(member.path)
                if rec is None:
                    continue
                member_summary = dict(summary)
                member_summary["sampled_this_image"] = member.path in sampled_paths
                rec["extra_metadata"]["face_cluster"] = member_summary
                self.record_updated.emit(member.path, rec)

            # Let the face-cluster pane acquire summaries as final analysis progresses.
            self._emit_face_clusters(clusters)

        self._emit_face_clusters(clusters)

    def _face_worker(self) -> None:
        # Do not import/validate the DeepFace runtime up front. A restarted scan
        # may be able to restore every face embedding/attribute result from the
        # permanent cache without touching TensorFlow at all.
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
                log_event("deepface-worker", "job-start", "extracting face embedding", path=path)
                face_success = False
                face_reason = None
                try:
                    result = analyzer.extract_embedding(path)
                    with self.face_results_lock:
                        self.face_results[path] = result

                    rec = self.records.get(path)
                    if rec is not None:
                        rec["extra_metadata"]["deepface"] = result.attributes
                        rec["extra_metadata"]["face_embedding"] = result.embedding
                        self.record_updated.emit(path, rec)

                    self.analysis_store.save(
                        path,
                        kind="face_embedding",
                        engine=self.face_embedding_cache_engine,
                        payload={
                            "status": "ok",
                            "embedding": result.embedding,
                            "attributes": result.attributes,
                            "celebrity_lookalike": result.celebrity_lookalike,
                        },
                    )
                    self.face_analyzed.emit(path, result.attributes)
                    face_success = True
                except FaceNotConfirmed as e:
                    face_reason = f"face not confirmed: {e}"
                    # This is not a processing failure: the VLM nominated the
                    # image, but the dedicated face detector did not confirm it.
                    rec = self.records.get(path)
                    if rec is not None:
                        rec["extra_metadata"]["deepface"] = {
                            "status": "face_not_confirmed",
                            "reason": str(e),
                        }
                        self.record_updated.emit(path, rec)
                    self.analysis_store.save(
                        path,
                        kind="face_embedding",
                        engine=self.face_embedding_cache_engine,
                        payload={
                            "status": "face_not_confirmed",
                            "reason": str(e),
                        },
                    )
                    self.status.emit(
                        f"DeepFace did not confirm a face; continuing: {path}"
                    )
                except DeepFaceUnavailable as e:
                    face_reason = f"DeepFace unavailable: {e}"
                    log_exception("deepface-worker", "job-failure", e, path=path)
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
                    face_reason = f"{type(e).__name__}: {e}"
                    log_exception("deepface-worker", "job-failure", e, path=path)
                    rec = self.records.get(path)
                    if rec is not None:
                        self._record_error(Path(path), rec, "DeepFace", e)
                finally:
                    log_event(
                        "deepface-worker", "job-finish", face_reason or "success",
                        level="INFO" if face_success else "WARNING",
                        success=face_success, path=path,
                    )
                    with self.face_cv:
                        self.face_jobs_completed += 1
                    self._emit_face_queue_count()

            self._poll_face_cluster_results()
            now = time.monotonic()
            if now - last_cluster >= CLUSTER_INTERVAL_SECONDS:
                self._submit_face_cluster_job()
                last_cluster = now

            # Once all embeddings are collected, freeze final clusters and only
            # then run age/gender/ethnicity/emotion/celebrity analysis on up to
            # three centroid-nearest images per cluster.
            with self.face_cv:
                done = self.media_analysis_done and not self.face_jobs
            if done:
                clusters = self._final_face_clusters()
                self._emit_face_clusters(clusters)
                self._finalize_face_clusters(analyzer, clusters)
                return

    def _start_face_worker(self) -> None:
        if self.face_worker_thread is not None:
            return
        self.face_worker_thread = threading.Thread(
            target=lambda: self._run_logged_worker(
                "deepface-worker", self._face_worker
            ),
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

            log_event("video-sampler", "job-start", "sampling video", path=str(job.path))
            video_success = False
            video_reason = None
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
                video_success = True
            except Exception as e:
                video_reason = f"{type(e).__name__}: {e}"
                log_exception("video-sampler", "job-failure", e, path=str(job.path))
                self._record_error(job.path, job.record, "Video sampling", e)
            finally:
                log_event(
                    "video-sampler", "job-finish", video_reason or "success",
                    level="INFO" if video_success else "ERROR",
                    success=video_success, path=str(job.path),
                )
                with self.video_sample_cv:
                    self.video_sample_jobs_completed += 1
                self._emit_video_sample_queue_count()

    def _start_video_sample_worker(self) -> None:
        if self.video_sample_worker_thread is not None:
            return
        self.video_sample_worker_thread = threading.Thread(
            target=lambda: self._run_logged_worker(
                "video-sampler", self._video_sample_worker
            ),
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

        thumb = os_cached_thumbnail_jpeg(path) or thumbnail_jpeg(image)
        for tag in tags:
            self.tag_counts[tag] += 1
        self.tagged.emit(str(path), tags, thumb)
        self._persist_media_analysis(path, rec)

        if any(str(tag).casefold() == "faceidentity" for tag in tags):
            self._restore_or_queue_face_analysis(path, rec)

    def _process_video(self, path: Path, rec: dict[str, Any]) -> None:
        if not self._wait_for_model_prefetch(path, rec):
            return
        if not self._captioning_available(path, rec):
            return

        self.status.emit(f"Tagging pre-sampled video frames: {path}")
        all_tags: dict[str, str] = {}
        thumb = os_cached_thumbnail_jpeg(path) or b""

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
        self._persist_media_analysis(path, rec)

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
        self._persist_media_analysis(path, rec)

    def _process_media_job(self, job: MediaJob) -> None:
        worker = "vlm-media-worker"
        log_event(worker, "job-start", "processing media", path=str(job.path), category=job.category)
        success = False
        failure_reason = None
        try:
            if job.category == "image":
                self._process_image(job.path, job.record)
            elif job.category == "video":
                self._process_video(job.path, job.record)
            elif job.category == "archive":
                self._process_archive(job.path, job.record)
            success = not bool(job.record.get("error"))
            if not success:
                failure_reason = str(job.record.get("error"))
        except JoyCaptionError as e:
            failure_reason = f"{type(e).__name__}: {e}"
            log_exception(worker, "job-failure", e, path=str(job.path), category=job.category)
            self._record_error(
                job.path, job.record, "JoyCaption", e, details=str(e)
            )
            self.status.emit(
                "VLM tagging failed for this file; PixelCue will continue with the next queued media item. "
                "Filesystem discovery is unaffected."
            )
        except Exception as e:
            failure_reason = f"{type(e).__name__}: {e}"
            log_exception(worker, "job-failure", e, path=str(job.path), category=job.category)
            self._record_error(job.path, job.record, "Processing", e)
        finally:
            self.record_updated.emit(str(job.path), job.record)
            log_event(
                worker,
                "job-finish",
                failure_reason or "success",
                level="INFO" if success else "ERROR",
                success=success,
                path=str(job.path),
                category=job.category,
            )

    def _media_worker(self) -> None:
        # libmagic wrappers are kept thread-local: filesystem metadata and
        # archive-member inspection can now happen concurrently.
        self.media_detector = MagicDetector()

        while True:
            job = None
            with self.media_cv:
                while not self.stop_requested:
                    # Always take the largest discovered, unprocessed image.
                    # image_jobs is maintained in descending filesize order.
                    if self.image_jobs:
                        _size, _seq, job = self.image_jobs.pop(0)
                        break

                    # Archives are a separate high-priority class and therefore
                    # cannot disturb image-to-image ordering.
                    if self.archive_media_jobs:
                        _priority, _seq, job = heapq.heappop(
                            self.archive_media_jobs
                        )
                        break

                    # Do not spend VLM time on video until filesystem discovery
                    # has completed, so newly discovered still images can never
                    # be blocked behind video inference.
                    if self.filesystem_discovery_done and self.video_media_jobs:
                        _priority, _seq, job = heapq.heappop(
                            self.video_media_jobs
                        )
                        break

                    if (
                        self.filesystem_discovery_done
                        and self.video_sampling_done
                        and not self.image_jobs
                        and not self.archive_media_jobs
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
            target=lambda: self._run_logged_worker(
                "vlm-media-worker", self._media_worker
            ),
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

        # Restore permanent analysis before applying current queue policy. A
        # cached result costs no model time, even if the file is now below the
        # minimum size threshold.
        if self._restore_media_analysis(path, rec, category):
            return

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

            # UI progress is throttled so a fast filesystem cannot flood Qt.
            now = time.monotonic()
            if now - self._last_current_path_emit >= 0.10:
                self.current_path.emit(str(path))
                self._last_current_path_emit = now

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
        log_event(
            "filesystem-scan", "start", "scan worker started",
            start_path=str(self.start_dir), model_profile=self.model_profile_id,
        )
        scan_success = False
        try:
            self._start_cluster_process()
            self.model_status.emit("VLM model deferred until uncached media is found")
            self._start_video_sample_worker()
            self._start_media_worker()
            self._start_face_worker()

            self.status.emit(
                "Filesystem discovery running independently of JoyCaption preparation…"
            )

            # This method never waits for model/tagging work.
            self._walk_filesystem()
            with self.records_lock:
                self.discovered_count.emit(len(self.records))

            self.filesystem_discovery_done = True
            with self.video_sample_cv:
                self.video_sample_cv.notify_all()
            with self.media_cv:
                queued = (
                    len(self.image_jobs)
                    + len(self.archive_media_jobs)
                    + len(self.video_media_jobs)
                )
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

            self._stop_cluster_process("scan complete")

            with self.records_lock:
                final_records = list(self.records.values())

            self.completed.emit(final_records)
            scan_success = True
            log_event(
                "filesystem-scan", "finish", "scan completed", success=True,
                files=len(final_records),
            )

        except Exception as exc:
            log_exception("filesystem-scan", "failure", exc)
            self._stop_cluster_process(f"scan failure: {type(exc).__name__}: {exc}")
            self.fatal_error.emit(traceback.format_exc())
            with self.records_lock:
                final_records = list(self.records.values())
            self.completed.emit(final_records)
            log_event(
                "filesystem-scan", "finish", f"{type(exc).__name__}: {exc}",
                level="ERROR", success=False, files=len(final_records),
            )


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
