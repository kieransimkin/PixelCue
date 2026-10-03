from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

DEEPFACE_MODEL = os.environ.get("PIXELCUE_FACE_MODEL", "ArcFace")
DEEPFACE_DETECTOR = os.environ.get("PIXELCUE_FACE_DETECTOR", "opencv")
CELEBRITY_DB = os.environ.get("PIXELCUE_CELEBRITY_DB", "").strip()
CLUSTER_INTERVAL_SECONDS = float(os.environ.get("PIXELCUE_FACE_CLUSTER_INTERVAL", "10"))
CLUSTER_COSINE_DISTANCE = float(os.environ.get("PIXELCUE_FACE_CLUSTER_DISTANCE", "0.35"))


class DeepFaceUnavailable(RuntimeError):
    pass


def deepface_install_command() -> str:
    return "uv sync --extra faces"


def deepface_python313_setup_command() -> str:
    return "uv python install 3.13 && uv sync --extra faces"



def deepface_available() -> bool:
    ok, _reason = validate_deepface_runtime()
    return ok


def validate_deepface_runtime() -> tuple[bool, str]:
    """Validate PixelCue's stable DeepFace runtime."""
    try:
        import importlib.util
        import importlib.metadata

        if sys.version_info[:2] != (3, 13):
            return False, (
                f"PixelCue face analysis requires Python 3.13; running "
                f"{sys.version_info.major}.{sys.version_info.minor}.\n\n"
                "Run:\n"
                + deepface_python313_setup_command()
                + "\nThen launch with: uv run pixelcue <path>"
            )

        if importlib.util.find_spec("deepface") is None:
            return False, "DeepFace is missing. Repair with:\n" + deepface_install_command()
        if importlib.util.find_spec("tensorflow") is None:
            return False, "TensorFlow is missing. Repair with:\n" + deepface_install_command()

        import tensorflow as tf
        tf_version = str(getattr(tf, "__version__", "unknown"))
        if tf_version != "2.21.0":
            return False, (
                f"PixelCue DeepFace requires TensorFlow 2.21.0; found {tf_version}.\n"
                "Repair with:\n" + deepface_install_command()
            )

        if importlib.util.find_spec("tf_keras") is None:
            return False, "tf-keras is missing. Repair with:\n" + deepface_install_command()

        try:
            tf_keras_version = importlib.metadata.version("tf-keras")
        except Exception:
            tf_keras_version = "unknown"
        if tf_keras_version != "2.21.0":
            return False, (
                f"PixelCue DeepFace requires tf-keras 2.21.0; found {tf_keras_version}.\n"
                "Repair with:\n" + deepface_install_command()
            )

        from deepface import DeepFace  # noqa: F401
        return True, ""
    except Exception as e:
        return False, (
            f"DeepFace runtime validation failed: {type(e).__name__}: {e}\n\n"
            "Repair with:\n" + deepface_install_command()
        )


@dataclass
class FaceResult:
    path: str
    embedding: list[float]
    attributes: dict[str, Any]
    celebrity_lookalike: str | None = None


def cosine_distance(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 0 or nb <= 0:
        return 1.0
    return 1.0 - dot / (na * nb)


def normalized_mean(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    mean = [sum(v[i] for v in vectors) / len(vectors) for i in range(len(vectors[0]))]
    norm = math.sqrt(sum(x * x for x in mean))
    return [x / norm for x in mean] if norm > 0 else mean


def cluster_embeddings(results: list[FaceResult], threshold: float = CLUSTER_COSINE_DISTANCE):
    clusters: list[dict[str, Any]] = []
    for result in results:
        best_idx, best_distance = None, float("inf")
        for idx, cluster in enumerate(clusters):
            distance = cosine_distance(result.embedding, cluster["centroid"])
            if distance < best_distance:
                best_idx, best_distance = idx, distance
        if best_idx is None or best_distance > threshold:
            clusters.append({"members": [result], "centroid": normalized_mean([result.embedding])})
        else:
            members = clusters[best_idx]["members"]
            members.append(result)
            clusters[best_idx]["centroid"] = normalized_mean([m.embedding for m in members])

    output = []
    for cluster in clusters:
        centroid = cluster["centroid"]
        members = cluster["members"]
        representative = min(members, key=lambda m: cosine_distance(m.embedding, centroid))
        output.append({
            "centroid": centroid,
            "representative_path": representative.path,
            "paths": [m.path for m in members],
            "members": members,
        })
    output.sort(key=lambda c: (-len(c["paths"]), c["representative_path"]))
    for idx, cluster in enumerate(output, 1):
        cluster["cluster_id"] = idx
    return output


class DeepFaceAnalyzer:
    def __init__(self, status_callback: Callable[[str], None] | None = None):
        self.status_callback = status_callback or (lambda _text: None)
        self._tensorflow_configured = False

    def _configure_tensorflow_cpu(self) -> None:
        """Keep DeepFace/TensorFlow off the GPU reserved for JoyCaption."""
        if self._tensorflow_configured:
            return
        try:
            import tensorflow as tf
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            # If TensorFlow has no GPU backend this is already the desired state.
            pass
        self._tensorflow_configured = True

    @staticmethod
    def _largest_face(items):
        def area(item):
            region = item.get("facial_area") or item.get("region") or {}
            return int(region.get("w", 0) or 0) * int(region.get("h", 0) or 0)
        return max(items, key=area)

    def analyze(self, path: str) -> FaceResult:
        ok, reason = validate_deepface_runtime()
        if not ok:
            raise DeepFaceUnavailable(reason + "\nThen restart PixelCue.")
        self._configure_tensorflow_cpu()
        from deepface import DeepFace
        self.status_callback(f"DeepFace analysing: {path}")

        reps = DeepFace.represent(
            img_path=path, model_name=DEEPFACE_MODEL,
            detector_backend=DEEPFACE_DETECTOR,
            enforce_detection=True, align=True,
        )
        if not reps:
            raise RuntimeError("DeepFace did not find a face")
        primary_rep = self._largest_face(list(reps))
        embedding = [float(x) for x in primary_rep["embedding"]]

        analyses = DeepFace.analyze(
            img_path=path, actions=["age", "gender", "race", "emotion"],
            detector_backend=DEEPFACE_DETECTOR,
            enforce_detection=True, align=True, silent=True,
        )
        if isinstance(analyses, dict):
            analyses = [analyses]
        a = self._largest_face(list(analyses)) if analyses else {}
        attrs = {
            "age": a.get("age"),
            "dominant_gender": a.get("dominant_gender"),
            "gender": a.get("gender"),
            "dominant_race": a.get("dominant_race"),
            "race": a.get("race"),
            # DeepFace names this classifier "race"; expose ethnicity aliases
            # too so exported metadata matches PixelCue's UI/domain language.
            "dominant_ethnicity": a.get("dominant_race"),
            "ethnicity": a.get("race"),
            "dominant_emotion": a.get("dominant_emotion"),
            "emotion": a.get("emotion"),
            "face_confidence": a.get("face_confidence"),
            "region": a.get("region"),
            "embedding_model": DEEPFACE_MODEL,
            "embedding_dimensions": len(embedding),
        }
        celebrity = self._celebrity_lookalike(path)
        attrs["celebrity_lookalike"] = celebrity or "unavailable (no celebrity DB configured)"
        return FaceResult(path, embedding, attrs, celebrity)

    def _celebrity_lookalike(self, path: str) -> str | None:
        if not CELEBRITY_DB or not Path(CELEBRITY_DB).expanduser().is_dir():
            return None
        from deepface import DeepFace
        kwargs = dict(
            img_path=path, db_path=str(Path(CELEBRITY_DB).expanduser()),
            model_name=DEEPFACE_MODEL, detector_backend=DEEPFACE_DETECTOR,
            distance_metric="cosine", enforce_detection=True, align=True, silent=True,
        )
        try:
            matches = DeepFace.find(**kwargs, similarity_search=True, k=1)
        except TypeError:
            matches = DeepFace.find(**kwargs)
        if not matches or len(matches[0].index) == 0:
            return None
        identity = str(matches[0].iloc[0].get("identity", "") or "")
        if not identity:
            return None
        p = Path(identity)
        return p.parent.name or p.stem
