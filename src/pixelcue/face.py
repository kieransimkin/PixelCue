from __future__ import annotations

import math
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# DeepFace and its TensorFlow models expect the legacy tf.keras API.
# This MUST be set before TensorFlow, RetinaFace, or DeepFace imports.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

DEEPFACE_MODEL = os.environ.get("PIXELCUE_FACE_MODEL", "ArcFace")
DEEPFACE_DETECTOR = os.environ.get("PIXELCUE_FACE_DETECTOR", "retinaface").strip().lower()
CELEBRITY_DB = os.environ.get("PIXELCUE_CELEBRITY_DB", "").strip()
CLUSTER_INTERVAL_SECONDS = float(os.environ.get("PIXELCUE_FACE_CLUSTER_INTERVAL", "10"))
CLUSTER_COSINE_DISTANCE = float(os.environ.get("PIXELCUE_FACE_CLUSTER_DISTANCE", "0.50"))
CLUSTER_MERGE_MARGIN = float(os.environ.get("PIXELCUE_FACE_CLUSTER_MERGE_MARGIN", "0.06"))
CLUSTER_ATTRIBUTE_MAX_SAMPLES = 3
CLUSTER_AGE_TOLERANCE_YEARS = float(os.environ.get("PIXELCUE_FACE_AGE_TOLERANCE", "8"))
CLUSTER_GENDER_CONFIDENCE = float(os.environ.get("PIXELCUE_FACE_GENDER_CONFIDENCE", "65"))
CLUSTER_ETHNICITY_CONFIDENCE = float(os.environ.get("PIXELCUE_FACE_ETHNICITY_CONFIDENCE", "50"))


def _opencv_cascade_path() -> Path | None:
    """Return OpenCV's frontal-face Haar cascade when it really exists."""
    try:
        import cv2

        candidates = []
        data = getattr(cv2, "data", None)
        if data is not None:
            haar_dir = getattr(data, "haarcascades", None)
            if haar_dir:
                candidates.append(
                    Path(haar_dir) / "haarcascade_frontalface_default.xml"
                )

        candidates.append(
            Path(cv2.__file__).resolve().parent
            / "data"
            / "haarcascade_frontalface_default.xml"
        )

        for candidate in candidates:
            if candidate.is_file():
                return candidate
    except Exception:
        pass
    return None


def detector_fallback_chain(primary: str | None = None) -> list[str]:
    """Return a small, unique detector fallback chain for one difficult image."""
    primary, _message = resolve_detector_backend(primary)
    candidates = [primary, "retinaface", "mtcnn"]

    # OpenCV is only useful when its cascade is actually present.
    if _opencv_cascade_path() is not None:
        candidates.append("opencv")

    out: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in out:
            out.append(candidate)
    return out


def resolve_detector_backend(requested: str | None = None) -> tuple[str, str | None]:
    """Resolve a usable DeepFace detector, preferring RetinaFace.

    Returns (backend, fallback_message). The message is non-None only when the
    requested backend could not be used and PixelCue selected a fallback.
    """
    import importlib.util

    requested = (requested or DEEPFACE_DETECTOR or "retinaface").strip().lower()

    if requested == "opencv":
        if _opencv_cascade_path() is not None:
            return "opencv", None

        if importlib.util.find_spec("retinaface") is not None:
            return (
                "retinaface",
                "OpenCV face detector data is incomplete; falling back to RetinaFace.",
            )
        if importlib.util.find_spec("mtcnn") is not None:
            return (
                "mtcnn",
                "OpenCV face detector data is incomplete; falling back to MTCNN.",
            )
        return "opencv", (
            "OpenCV face detector data is incomplete and no RetinaFace/MTCNN "
            "fallback is installed."
        )

    if requested == "retinaface":
        if importlib.util.find_spec("retinaface") is not None:
            return "retinaface", None
        if importlib.util.find_spec("mtcnn") is not None:
            return "mtcnn", "RetinaFace is unavailable; falling back to MTCNN."
        return "retinaface", "RetinaFace is unavailable and no MTCNN fallback is installed."

    return requested, None


class DeepFaceUnavailable(RuntimeError):
    pass


class FaceNotConfirmed(RuntimeError):
    """JoyCaption suggested a face but DeepFace could not confirm one."""
    pass


def deepface_install_command() -> str:
    return "uv sync --extra faces"


def deepface_python313_setup_command() -> str:
    return "uv python install 3.13 && uv sync --extra faces"



def deepface_available() -> bool:
    ok, _reason = validate_deepface_runtime()
    return ok


def validate_deepface_runtime() -> tuple[bool, str]:
    """Validate PixelCue's stable DeepFace runtime without importing TensorFlow first."""
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

        # Do NOT import tensorflow here. Importing it before DeepFace/legacy-Keras
        # configuration can permanently initialise Keras 3 for this process.
        try:
            tf_version = importlib.metadata.version("tensorflow")
        except Exception:
            tf_version = "unknown"

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

        # Reassert immediately before DeepFace import in case another library
        # mutated the process environment.
        os.environ["TF_USE_LEGACY_KERAS"] = "1"

        # DeepFace itself also sets TF_USE_LEGACY_KERAS=1 before TensorFlow import.
        from deepface import DeepFace  # noqa: F401

        detector, detector_message = resolve_detector_backend()
        if detector == "opencv" and _opencv_cascade_path() is None:
            return False, (
                detector_message
                or "OpenCV face detector data is incomplete."
            ) + "\nRepair face dependencies with:\n" + deepface_install_command()

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


def _merge_nearby_face_clusters(
    clusters: list[dict[str, Any]],
    merge_threshold: float,
) -> list[dict[str, Any]]:
    """Merge centroid-near clusters after the first assignment pass.

    The first pass can fragment one identity because its centroid depends on
    discovery order. Re-merging the closest centroids makes clustering much
    less brittle without requiring every individual face embedding pair to be
    within the original assignment cutoff.
    """
    clusters = [
        {
            "members": list(cluster["members"]),
            "centroid": list(cluster["centroid"]),
        }
        for cluster in clusters
    ]

    while len(clusters) > 1:
        best_pair: tuple[int, int] | None = None
        best_distance = float("inf")

        for left in range(len(clusters) - 1):
            for right in range(left + 1, len(clusters)):
                distance = cosine_distance(
                    clusters[left]["centroid"],
                    clusters[right]["centroid"],
                )
                if distance < best_distance:
                    best_distance = distance
                    best_pair = (left, right)

        if best_pair is None or best_distance > merge_threshold:
            break

        left, right = best_pair
        merged_members = (
            clusters[left]["members"] + clusters[right]["members"]
        )
        clusters[left] = {
            "members": merged_members,
            "centroid": normalized_mean(
                [member.embedding for member in merged_members]
            ),
        }
        del clusters[right]

    return clusters


def cluster_embeddings(
    results: list[FaceResult],
    threshold: float = CLUSTER_COSINE_DISTANCE,
    merge_threshold: float | None = None,
):
    """Cluster face embeddings with a deliberately tolerant two-stage policy."""
    clusters: list[dict[str, Any]] = []
    for result in results:
        best_idx, best_distance = None, float("inf")
        for idx, cluster in enumerate(clusters):
            distance = cosine_distance(result.embedding, cluster["centroid"])
            if distance < best_distance:
                best_idx, best_distance = idx, distance

        if best_idx is None or best_distance > threshold:
            clusters.append({
                "members": [result],
                "centroid": normalized_mean([result.embedding]),
            })
        else:
            members = clusters[best_idx]["members"]
            members.append(result)
            clusters[best_idx]["centroid"] = normalized_mean(
                [member.embedding for member in members]
            )

    if merge_threshold is None:
        merge_threshold = min(
            1.0,
            float(threshold) + max(0.0, CLUSTER_MERGE_MARGIN),
        )

    clusters = _merge_nearby_face_clusters(
        clusters,
        merge_threshold=float(merge_threshold),
    )

    output = []
    for cluster in clusters:
        centroid = cluster["centroid"]
        members = cluster["members"]
        representative = min(
            members,
            key=lambda member: cosine_distance(member.embedding, centroid),
        )
        output.append({
            "centroid": centroid,
            "representative_path": representative.path,
            "paths": [member.path for member in members],
            "members": members,
        })

    output.sort(key=lambda c: (-len(c["paths"]), c["representative_path"]))
    for idx, cluster in enumerate(output, 1):
        cluster["cluster_id"] = idx
    return output


def central_cluster_members(cluster: dict[str, Any], limit: int = 3) -> list[FaceResult]:
    """Return members nearest the cluster centroid, most central first."""
    centroid = cluster.get("centroid") or []
    members = list(cluster.get("members") or [])
    members.sort(key=lambda m: cosine_distance(m.embedding, centroid))
    return members[: max(0, min(int(limit), CLUSTER_ATTRIBUTE_MAX_SAMPLES))]


def _distribution_score(sample: dict[str, Any], field: str, label: str | None) -> float:
    if not label:
        return 0.0
    distribution = sample.get(field)
    if not isinstance(distribution, dict):
        return 0.0
    try:
        return float(distribution.get(label, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _two_samples_confident(a: dict[str, Any], b: dict[str, Any]) -> bool:
    ages = []
    genders = []
    ethnicities = []
    for sample in (a, b):
        try:
            ages.append(float(sample.get("age")))
        except (TypeError, ValueError):
            return False

        gender = sample.get("dominant_gender")
        ethnicity = sample.get("dominant_ethnicity") or sample.get("dominant_race")
        if not gender or not ethnicity:
            return False

        if _distribution_score(sample, "gender", str(gender)) < CLUSTER_GENDER_CONFIDENCE:
            return False
        ethnicity_score = _distribution_score(sample, "ethnicity", str(ethnicity))
        if ethnicity_score < CLUSTER_ETHNICITY_CONFIDENCE:
            ethnicity_score = _distribution_score(sample, "race", str(ethnicity))
        if ethnicity_score < CLUSTER_ETHNICITY_CONFIDENCE:
            return False

        genders.append(str(gender).casefold())
        ethnicities.append(str(ethnicity).casefold())

    if len(set(genders)) != 1 or len(set(ethnicities)) != 1:
        return False
    if max(ages) - min(ages) > CLUSTER_AGE_TOLERANCE_YEARS:
        return False

    celebrities = [
        str(s.get("celebrity_lookalike")).casefold()
        for s in (a, b)
        if s.get("celebrity_lookalike")
        and "unavailable" not in str(s.get("celebrity_lookalike")).casefold()
    ]
    if len(celebrities) >= 2 and len(set(celebrities)) != 1:
        return False
    return True


def cluster_attributes_confident(samples: list[dict[str, Any]]) -> bool:
    """Return true when any two analysed central samples strongly agree."""
    if len(samples) < 2:
        return False
    for i in range(len(samples) - 1):
        for j in range(i + 1, len(samples)):
            if _two_samples_confident(samples[i], samples[j]):
                return True
    return False

def _average_distribution(samples: list[dict[str, Any]], field: str) -> dict[str, float]:
    keys: set[str] = set()
    for sample in samples:
        value = sample.get(field)
        if isinstance(value, dict):
            keys.update(str(k) for k in value)

    averaged: dict[str, float] = {}
    for key in keys:
        vals = []
        for sample in samples:
            value = sample.get(field)
            if isinstance(value, dict):
                try:
                    vals.append(float(value.get(key, 0.0) or 0.0))
                except (TypeError, ValueError):
                    pass
        if vals:
            averaged[key] = sum(vals) / len(vals)
    return averaged


def aggregate_cluster_attributes(
    samples: list[dict[str, Any]],
    sampled_paths: list[str],
    confident: bool,
) -> dict[str, Any]:
    if not samples:
        return {
            "status": "unavailable",
            "samples_analyzed": 0,
            "sampled_paths": sampled_paths,
            "confidence_sufficient": False,
        }

    ages = []
    for sample in samples:
        try:
            ages.append(float(sample.get("age")))
        except (TypeError, ValueError):
            pass

    gender = _average_distribution(samples, "gender")
    ethnicity = _average_distribution(samples, "ethnicity")
    if not ethnicity:
        ethnicity = _average_distribution(samples, "race")
    emotion = _average_distribution(samples, "emotion")

    def dominant(distribution: dict[str, float]) -> str | None:
        return max(distribution, key=distribution.get) if distribution else None

    celebrities = [
        str(sample["celebrity_lookalike"])
        for sample in samples
        if sample.get("celebrity_lookalike")
        and "unavailable" not in str(sample["celebrity_lookalike"]).casefold()
    ]
    celebrity = None
    if celebrities:
        celebrity = max(set(celebrities), key=celebrities.count)

    return {
        "status": "complete",
        "samples_analyzed": len(samples),
        "sampled_paths": sampled_paths,
        "confidence_sufficient": bool(confident),
        "age": round(statistics.median(ages), 1) if ages else None,
        "age_range": [round(min(ages), 1), round(max(ages), 1)] if ages else None,
        "dominant_gender": dominant(gender),
        "gender": gender,
        "dominant_race": dominant(ethnicity),
        "dominant_ethnicity": dominant(ethnicity),
        "race": ethnicity,
        "ethnicity": ethnicity,
        "dominant_emotion": dominant(emotion),
        "emotion": emotion,
        "celebrity_lookalike": celebrity or "unavailable (no confident match)",
    }


class DeepFaceAnalyzer:
    def __init__(self, status_callback: Callable[[str], None] | None = None):
        self.status_callback = status_callback or (lambda _text: None)
        self._tensorflow_configured = False
        self.detector_backend, fallback_message = resolve_detector_backend()
        if fallback_message:
            self.status_callback(fallback_message)

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

    def extract_embedding(self, path: str) -> FaceResult:
        """Fast per-image face stage: confirm face and extract embedding only."""
        ok, reason = validate_deepface_runtime()
        if not ok:
            raise DeepFaceUnavailable(reason + "\nThen restart PixelCue.")

        os.environ["TF_USE_LEGACY_KERAS"] = "1"
        from deepface import DeepFace
        self._configure_tensorflow_cpu()

        self.status_callback(f"DeepFace embedding: {path}")

        reps = None
        selected_backend = None
        detector_errors: list[str] = []

        for backend in detector_fallback_chain(self.detector_backend):
            try:
                candidate = DeepFace.represent(
                    img_path=path,
                    model_name=DEEPFACE_MODEL,
                    detector_backend=backend,
                    enforce_detection=True,
                    align=True,
                )
                if candidate:
                    reps = candidate
                    selected_backend = backend
                    break
            except Exception as e:
                detector_errors.append(
                    f"{backend}: {type(e).__name__}: {e}"
                )

        if not reps or selected_backend is None:
            raise FaceNotConfirmed(
                "DeepFace could not confirm a face after trying: "
                + ", ".join(detector_fallback_chain(self.detector_backend))
                + (
                    f". Last errors: {' | '.join(detector_errors[-3:])}"
                    if detector_errors
                    else ""
                )
            )

        primary_rep = self._largest_face(list(reps))
        embedding = [float(x) for x in primary_rep["embedding"]]
        attrs = {
            "status": "embedding_extracted",
            "embedding_model": DEEPFACE_MODEL,
            "embedding_dimensions": len(embedding),
            "detector_backend": selected_backend,
            "facial_area": primary_rep.get("facial_area"),
            "face_confidence": primary_rep.get("face_confidence"),
        }
        return FaceResult(path, embedding, attrs, None)

    def analyze_attributes(
        self,
        path: str,
        preferred_detector: str | None = None,
    ) -> dict[str, Any]:
        """Expensive age/gender/ethnicity/emotion/celebrity stage used per cluster."""
        ok, reason = validate_deepface_runtime()
        if not ok:
            raise DeepFaceUnavailable(reason + "\nThen restart PixelCue.")

        os.environ["TF_USE_LEGACY_KERAS"] = "1"
        from deepface import DeepFace
        self._configure_tensorflow_cpu()

        self.status_callback(f"DeepFace cluster attributes: {path}")

        analyses = None
        selected_backend = None
        errors: list[str] = []
        for backend in detector_fallback_chain(preferred_detector or self.detector_backend):
            try:
                candidate = DeepFace.analyze(
                    img_path=path,
                    actions=["age", "gender", "race", "emotion"],
                    detector_backend=backend,
                    enforce_detection=True,
                    align=True,
                    silent=True,
                )
                if isinstance(candidate, dict):
                    candidate = [candidate]
                if candidate:
                    analyses = candidate
                    selected_backend = backend
                    break
            except Exception as e:
                errors.append(f"{backend}: {type(e).__name__}: {e}")

        if not analyses or selected_backend is None:
            raise FaceNotConfirmed(
                "DeepFace could not analyse cluster attributes for this face"
                + (f": {' | '.join(errors[-3:])}" if errors else "")
            )

        a = self._largest_face(list(analyses))
        race = a.get("race")
        attrs = {
            "age": a.get("age"),
            "dominant_gender": a.get("dominant_gender"),
            "gender": a.get("gender"),
            "dominant_race": a.get("dominant_race"),
            "race": race,
            "dominant_ethnicity": a.get("dominant_race"),
            "ethnicity": race,
            "dominant_emotion": a.get("dominant_emotion"),
            "emotion": a.get("emotion"),
            "face_confidence": a.get("face_confidence"),
            "region": a.get("region"),
            "detector_backend": selected_backend,
        }
        celebrity = self._celebrity_lookalike(path, selected_backend)
        attrs["celebrity_lookalike"] = (
            celebrity or "unavailable (no celebrity DB configured)"
        )
        return attrs

    def _celebrity_lookalike(self, path: str, detector_backend: str | None = None) -> str | None:
        if not CELEBRITY_DB or not Path(CELEBRITY_DB).expanduser().is_dir():
            return None
        from deepface import DeepFace
        kwargs = dict(
            img_path=path, db_path=str(Path(CELEBRITY_DB).expanduser()),
            model_name=DEEPFACE_MODEL, detector_backend=detector_backend or self.detector_backend,
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
