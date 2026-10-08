from __future__ import annotations

from typing import Any

from .face import (
    FaceResult,
    cluster_embeddings,
    CLUSTER_COSINE_DISTANCE,
    CLUSTER_MERGE_MARGIN,
)
from .logging_utils import log_event, log_exception


def face_cluster_process_main(request_queue, result_queue) -> None:
    """Persistent clustering subprocess; state is incrementally updated over IPC."""
    worker = "face-cluster-process"
    embeddings: dict[str, list[float]] = {}
    log_event(worker, "start", "clustering subprocess started")
    finish_reason = "normal stop"
    success = True
    try:
        while True:
            message = request_queue.get()
            command = message.get("command")
            if command == "stop":
                finish_reason = str(message.get("reason") or "stop requested")
                break
            if command != "cluster":
                continue

            job_id = int(message.get("job_id", 0))
            items = list(message.get("items") or [])
            threshold = float(message.get("threshold", CLUSTER_COSINE_DISTANCE))
            log_event(
                worker,
                "job-start",
                "clustering embeddings",
                job_id=job_id,
                new_embeddings=len(items),
                known_embeddings=len(embeddings),
                assignment_threshold=threshold,
                merge_threshold=min(
                    1.0,
                    threshold + max(0.0, CLUSTER_MERGE_MARGIN),
                ),
            )
            try:
                for item in items:
                    path = str(item["path"])
                    embeddings[path] = [float(x) for x in item["embedding"]]

                results = [
                    FaceResult(path=path, embedding=embedding, attributes={})
                    for path, embedding in embeddings.items()
                ]
                clusters = cluster_embeddings(results, threshold=threshold)
                payload = [
                    {
                        "cluster_id": int(cluster["cluster_id"]),
                        "centroid": [float(x) for x in cluster["centroid"]],
                        "representative_path": str(cluster["representative_path"]),
                        "paths": list(cluster["paths"]),
                        "size": len(cluster["paths"]),
                    }
                    for cluster in clusters
                ]
                result_queue.put({"job_id": job_id, "ok": True, "clusters": payload})
                log_event(
                    worker,
                    "job-finish",
                    "clustering succeeded",
                    job_id=job_id,
                    clusters=len(payload),
                    embeddings=len(embeddings),
                )
            except BaseException as exc:
                log_exception(worker, "job-failure", exc, job_id=job_id)
                result_queue.put(
                    {
                        "job_id": job_id,
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    except BaseException as exc:
        success = False
        finish_reason = f"{type(exc).__name__}: {exc}"
        log_exception(worker, "failure", exc)
    finally:
        log_event(
            worker,
            "finish",
            finish_reason,
            level="INFO" if success else "ERROR",
            success=success,
        )
