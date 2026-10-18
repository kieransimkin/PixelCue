"""Cheap, bounded temporal evidence. No neural models, captions or downloads.

PixelCue owns raster decoding and optical flow. PrismMoves selects representative
frames and turns this evidence into a description. Never join conditions, infer
motion across sparse gaps, or treat optical-flow confidence as a probability.
"""
from __future__ import annotations

from collections import defaultdict
import math
import statistics
import time

import numpy as np
from PIL import Image

from .visual_analysis import load_image, _linear

SEQUENCE_VERSION = "1.0.0"
MAX_FRAMES = 32768
HUES = ("red", "orange", "yellow", "lime", "green", "turquoise",
        "cyan", "blue", "indigo", "violet", "magenta", "pink")


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _median(values):
    values = [float(x) for x in values if _finite(x)]
    return statistics.median(values) if values else None


def frame_features(image):
    """Linear-light brightness and small spatial descriptors, not aesthetic scores."""
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    linear = _linear(rgb) @ np.array([.2126, .7152, .0722], dtype=np.float32)
    grey = rgb @ np.array([.2126, .7152, .0722], dtype=np.float32)
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0
    visible = (hsv[..., 2] > .08) & (hsv[..., 1] > .1)
    weights = hsv[..., 1] * hsv[..., 2] * visible
    hue, _ = np.histogram(hsv[..., 0], bins=12, range=(0, 1), weights=weights)
    hue = hue / max(float(hue.sum()), 1e-9)
    gy, gx = [np.zeros_like(grey), np.zeros_like(grey)]
    if grey.shape[0] > 1:
        gy = np.gradient(grey, axis=0)
    if grey.shape[1] > 1:
        gx = np.gradient(grey, axis=1)
    edge = np.hypot(gx, gy)
    small = image.resize((8, 8), Image.Resampling.BOX)
    sketch = (np.asarray(small, dtype=float) / 255.0).reshape(-1)
    small.close()
    uniform = float(grey.std()) < .012
    near_black = float((rgb.max(axis=2) < .02).mean())
    near_white = float((rgb.min(axis=2) > .98).mean())
    return {
        "mean_linear_luminance": float(linear.mean()),
        "luminance_std": float(linear.std()),
        "contrast": float(grey.std()),
        "mean_saturation": float(hsv[..., 1][hsv[..., 2] > .08].mean())
            if np.any(hsv[..., 2] > .08) else 0.0,
        "hue_histogram_12": hue.tolist(),
        "coloured_fraction": float(visible.mean()),
        "near_white_fraction": near_white,
        "near_black_fraction": near_black,
        "edge_density": float((edge > .06).mean()),
        "horizontal_mirror_mae": float(np.abs(grey - grey[:, ::-1]).mean()),
        "vertical_mirror_mae": float(np.abs(grey - grey[::-1]).mean()),
        "semantic_eligible": not uniform and near_black < .98 and near_white < .98,
        "semantic_exclusion": "near-uniform raster" if uniform else
            "near-white washout" if near_white >= .98 else
            "near-black raster" if near_black >= .98 else None,
        "descriptor": np.round(sketch, 4).tolist(),
    }


def _unit_grey(image):
    return np.asarray(image.convert("L"), dtype=np.uint8)


def pair_motion(previous, current, dt, *, max_gap=.25):
    """Forward/backward LK tracks, robust similarity fit and dense residual flow.

    Rotation sign is in image coordinates: positive means clockwise. Translation
    is evaluated at the image centre, avoiding apparent translation from rotation.
    Dense flow is a deformation proxy only; trails/intent cannot be deduced from it.
    """
    if not _finite(dt) or dt <= 0:
        raise ValueError("Frame timestamps must strictly increase")
    empty = {"status": "unavailable", "dt_seconds": dt, "reason": None}
    if dt > max_gap:
        return dict(empty, reason="sampling gap exceeds temporal limit")
    if previous.size != current.size:
        return dict(empty, reason="analysis raster dimensions changed")
    a, b = _unit_grey(previous), _unit_grey(current)
    delta = float(np.abs(a.astype(float) - b).mean() / 255.0)
    result = dict(empty, pixel_change=delta)
    if min(float(a.std()), float(b.std())) < 3:
        return dict(result, reason="insufficient texture; uniform frames do not establish motion")
    if delta < .0005:
        return dict(result, status="measured", reason=None, rotation_radians_per_second=0.0,
                    log_scale_per_second=0.0, translation_xy_per_second=[0.0, 0.0],
                    deformation_per_second=0.0, stationary=True, tracking_support=1.0,
                    method="near-identical textured rasters")
    try:
        import cv2
    except ImportError:
        return dict(result, reason="OpenCV unavailable; install pixelcue[sequence]")
    h, w = a.shape
    points = cv2.goodFeaturesToTrack(a, maxCorners=240, qualityLevel=.015,
                                    minDistance=max(3, min(w, h) // 40), blockSize=5)
    if points is None or len(points) < 8:
        return dict(result, reason="too few trackable features")
    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
    after, ok, _ = cv2.calcOpticalFlowPyrLK(a, b, points, None, **lk)
    if after is None or ok is None:
        return dict(result, reason="forward tracking failed")
    before, back_ok, _ = cv2.calcOpticalFlowPyrLK(b, a, after, None, **lk)
    if before is None or back_ok is None:
        return dict(result, reason="reverse tracking failed")
    p, q = points[:, 0, :], after[:, 0, :]
    error = np.linalg.norm(before[:, 0, :] - p, axis=1)
    good = ok.ravel().astype(bool) & back_ok.ravel().astype(bool) & (error < 1.0)
    good &= np.isfinite(q).all(axis=1) & (q[:, 0] >= 0) & (q[:, 0] < w)
    good &= (q[:, 1] >= 0) & (q[:, 1] < h)
    p, q = p[good], q[good]
    if len(p) < 8:
        return dict(result, reason="too few forward/backward-consistent tracks")
    matrix, mask = cv2.estimateAffinePartial2D(
        p, q, method=cv2.RANSAC, ransacReprojThreshold=1.5,
        maxIters=1000, confidence=.99, refineIters=10)
    if matrix is None or mask is None:
        return dict(result, reason="global motion fit failed")
    inliers = mask.ravel().astype(bool)
    supported = p[inliers]
    coverage = (float(np.ptp(supported[:, 0]) * np.ptp(supported[:, 1]) / (w * h))
                if len(supported) else 0)
    fit_support = float(inliers.mean())
    track_support = float(len(p) / len(points))
    if len(supported) < 8 or fit_support < .65 or coverage < .08 or track_support < .3:
        return dict(result, reason="unreliable or spatially restricted motion fit",
                    tracking_support=track_support, inlier_fraction=fit_support,
                    spatial_coverage=coverage)
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    if not .7 <= scale <= 1.4:
        return dict(result, reason="per-frame scale exceeds supported tracking range")
    centre = np.array([(w - 1) / 2, (h - 1) / 2, 1.0])
    displacement = matrix @ centre - centre[:2]
    # Dense flow residual on textured, non-border pixels. Forward/backward
    # disagreement rejects occlusions instead of calling them liquid motion.
    flow = cv2.calcOpticalFlowFarneback(a, b, None, .5, 3, 15, 3, 5, 1.2, 0)
    reverse = cv2.calcOpticalFlowFarneback(b, a, None, .5, 3, 15, 3, 5, 1.2, 0)
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    mx, my = xx + flow[..., 0], yy + flow[..., 1]
    backward = cv2.remap(reverse, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    texture = cv2.cornerMinEigenVal(a, blockSize=5) > .00005
    valid = texture & (mx >= 3) & (mx < w - 3) & (my >= 3) & (my < h - 3)
    valid &= np.linalg.norm(flow + backward, axis=2) < 1.5
    predicted = np.stack((matrix[0, 0]*xx + matrix[0, 1]*yy + matrix[0, 2] - xx,
                          matrix[1, 0]*xx + matrix[1, 1]*yy + matrix[1, 2] - yy), axis=2)
    residual = np.linalg.norm(flow - predicted, axis=2)
    deformation = (float(np.median(residual[valid])) / math.hypot(w, h) / dt
                   if valid.mean() > .03 else None)
    return dict(result, status="measured", reason=None,
                rotation_radians_per_second=float(math.atan2(matrix[1, 0], matrix[0, 0]) / dt),
                log_scale_per_second=float(math.log(scale) / dt),
                translation_xy_per_second=(displacement / [w, h] / dt).tolist(),
                deformation_per_second=deformation, stationary=False,
                tracking_support=track_support, inlier_fraction=fit_support,
                spatial_coverage=coverage, tracks=len(p),
                dense_valid_fraction=float(valid.mean()),
                method="LK forward/backward + RANSAC similarity + Farneback residual")


def brightness_period(frames):
    """Require uniform cadence, adequate duration and >=3 observed cycles."""
    if len(frames) < 16:
        return {"status": "unknown", "reason": "too few frames"}
    times = np.array([f["seconds"] for f in frames])
    delta = np.diff(times)
    duration = float(times[-1] - times[0])
    if duration < 2 or delta.min() <= 0 or delta.max() > .25:
        return {"status": "unknown", "reason": "insufficient dense temporal coverage"}
    if float(delta.std() / delta.mean()) > .03:
        return {"status": "unknown", "reason": "irregular cadence; not interpolated"}
    values = np.array([f["features"]["mean_linear_luminance"] for f in frames])
    if float(np.ptp(values)) < .06:
        return {"status": "not_detected", "reason": "brightness variation too small"}
    # Detrend: a single fade is not a pulse train.
    values -= np.polyval(np.polyfit(times - times[0], values, 1), times - times[0])
    power = abs(np.fft.rfft(values * np.hanning(len(values)))) ** 2
    frequencies = np.fft.rfftfreq(len(values), float(delta.mean()))
    eligible = (frequencies * duration >= 3) & (frequencies <= .25 / float(delta.mean()))
    if not eligible.any() or power[1:].sum() < 1e-8:
        return {"status": "not_detected"}
    peak = int(np.argmax(np.where(eligible, power, 0)))
    concentration = float(power[max(1, peak-1):peak+2].sum() / power[1:].sum())
    if concentration < .65:
        return {"status": "not_detected", "spectral_concentration": concentration}
    return {"status": "measured", "frequency_hz": float(frequencies[peak]),
            "cycles_observed": float(frequencies[peak] * duration),
            "frequency_resolution_hz": float(1 / (len(values) * delta.mean())),
            "spectral_concentration": concentration,
            "caveat": "Brightness periodicity only, not beat synchronisation or flash safety."}


def _condition_summary(frames, transitions):
    measured = [x for x in transitions if x["status"] == "measured"]
    result = {
        "frames": len(frames), "transitions": len(transitions),
        "measured_transitions": len(measured),
        "temporal_coverage": len(measured) / max(1, len(transitions)),
        "duration_seconds": frames[-1]["seconds"] - frames[0]["seconds"] if frames else 0,
        "brightness_period": brightness_period(frames),
        "motion": [],
        "input_static": bool(frames) and all(f.get("input_pixels_sha256") for f in frames)
            and len({f.get("input_pixels_sha256") for f in frames}) == 1,
    }
    # Stable evidence needs several successive measurements, not one fortunate fit.
    if len(measured) < 4 or result["temporal_coverage"] < .35:
        result["motion_status"] = "unknown"
        return result
    result["motion_status"] = "measured"
    fields = ("rotation_radians_per_second", "log_scale_per_second", "deformation_per_second")
    for field in fields:
        result[field] = _median(x.get(field) for x in measured)
    for field, cutoff, positive, negative in (
        (fields[0], .045, "clockwise rotation", "counterclockwise rotation"),
        (fields[1], .035, "outward expansion", "inward contraction"),
    ):
        positive_pairs = [x for x in measured if x.get(field, 0) > cutoff]
        negative_pairs = [x for x in measured if x.get(field, 0) < -cutoff]
        for label, evidence in ((positive, positive_pairs), (negative, negative_pairs)):
            if len(evidence) >= 4 and len(evidence) / len(measured) >= .3:
                result["motion"].append({"label": label,
                    "support_fraction": len(evidence) / len(measured),
                    "evidence": [x["id"] for x in evidence],
                    "median_rate": _median(x[field] for x in evidence)})
    for axis, labels in ((0, ("rightward drift", "leftward drift")),
                         (1, ("downward drift", "upward drift"))):
        for sign, label in ((1, labels[0]), (-1, labels[1])):
            evidence = [x for x in measured if sign*x["translation_xy_per_second"][axis] > .02]
            if len(evidence) >= 4 and len(evidence) / len(measured) >= .5:
                result["motion"].append({"label": label, "evidence": [x["id"] for x in evidence],
                                         "support_fraction": len(evidence)/len(measured)})
    residual = [x for x in measured if (x.get("deformation_per_second") or 0) > .02]
    if len(residual) >= 4 and len(residual) / len(measured) > .4:
        result["motion"].append({"label": "local warping", "evidence": [x["id"] for x in residual],
                                 "support_fraction": len(residual)/len(measured)})
    if not result["motion"] and all(x.get("stationary") for x in measured):
        result["motion"].append({"label": "visually stationary in the sampled window",
                                "evidence": [x["id"] for x in measured], "support_fraction": 1.0})
    return result


def analyse_sequences(records, *, max_edge=256, max_gap=.25, background="#000000"):
    """Analyse explicit frame records, independently for each condition.

    Records: id, condition_id, seconds, path, sha256 (required), and optional
    input_path/input_sha256. Source paths are deliberately absent from receipts.
    Memory is bounded to adjacent rasters; the compact evidence retains every
    frame, including blanks and unsupported intervals. No semantic prerequisite.
    """
    if not isinstance(records, list) or not records or len(records) > MAX_FRAMES:
        raise ValueError(f"Expected 1..{MAX_FRAMES} explicit frame records")
    if not _finite(max_gap) or not .01 <= max_gap <= .5:
        raise ValueError("max_gap must be between .01 and .5 seconds")
    if not 64 <= max_edge <= 512:
        raise ValueError("Temporal max_edge must be between 64 and 512")
    groups = defaultdict(list)
    seen = set()
    for row in records:
        identity = row.get("id")
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError("Frame ids must be unique nonempty strings")
        seen.add(identity)
        condition = row.get("condition_id")
        if not isinstance(condition, str) or not condition or not _finite(row.get("seconds")):
            raise ValueError("Each frame needs a condition and finite timestamp")
        if row["seconds"] < 0:
            raise ValueError("Negative frame timestamp")
        sha = row.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("Each frame needs a lowercase SHA-256 source digest")
        groups[condition].append(row)
    if len(groups) > 512:
        raise ValueError("At most 512 experimental conditions")
    started = time.monotonic()
    all_frames, conditions, errors = [], [], []
    for condition, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: r["seconds"])
        if any(b["seconds"] <= a["seconds"] for a, b in zip(rows, rows[1:])):
            raise ValueError("Timestamps must strictly increase within each condition")
        previous, prior_row = None, None
        frames, transitions = [], []
        try:
            for row in rows:
                try:
                    image, source = load_image(row["path"], max_edge=max_edge, background=background)
                    if source["sha256"] != row["sha256"]:
                        image.close()
                        raise ValueError("capture hash mismatch")
                except (OSError, ValueError) as exc:
                    errors.append({"id": row["id"], "condition_id": condition,
                                   "reason": type(exc).__name__ + ": capture unavailable or changed"})
                    if previous is not None:
                        previous.close()
                    previous, prior_row = None, None
                    continue
                features = frame_features(image)
                item = {"id": row["id"], "condition_id": condition, "seconds": row["seconds"],
                        "sha256": source["sha256"], "pixels_sha256": source["analysed_pixels_sha256"],
                        "analysis_dimensions": source["analysis_dimensions"], "features": features}
                for key in ("imagery", "audio_mode", "sample"):
                    if key in row:
                        item[key] = row[key]
                if row.get("input_path") and row.get("input_sha256"):
                    try:
                        original, original_source = load_image(row["input_path"], max_edge=max_edge,
                                                                background=background)
                        try:
                            if original_source["sha256"] != row["input_sha256"]:
                                raise ValueError("input hash mismatch")
                            item["input_pixels_sha256"] = original_source["analysed_pixels_sha256"]
                            original_features = frame_features(original)
                            item["input_comparison"] = {
                                "input_sha256": original_source["sha256"],
                                "edge_density_difference": features["edge_density"]-original_features["edge_density"],
                                "input_edge_density": original_features["edge_density"],
                                "source_mean_luminance": original_features["mean_linear_luminance"],
                            }
                        finally:
                            original.close()
                    except (OSError, ValueError):
                        item["input_comparison"] = {"status": "unavailable"}
                frames.append(item)
                if previous is not None:
                    motion = pair_motion(previous, image, row["seconds"]-prior_row["seconds"], max_gap=max_gap)
                    transitions.append(dict(motion, id=prior_row["id"]+"->"+row["id"],
                                            a=prior_row["id"], b=row["id"]))
                    previous.close()
                previous, prior_row = image, row
        finally:
            if previous is not None:
                previous.close()
        all_frames.extend(frames)
        conditions.append({"id": condition, "summary": _condition_summary(frames, transitions),
                           "transitions": transitions})
    try:
        import cv2
        flow_version = cv2.__version__
    except ImportError:
        flow_version = None
    return {"schema": "pixelcue.visual-sequence/1", "analysis_version": SEQUENCE_VERSION,
            "state": "completed" if not errors else "partial", "frames": all_frames,
            "conditions": conditions, "errors": errors,
            "settings": {"max_edge": max_edge, "max_gap_seconds": max_gap, "background": background},
            "dependencies": {"numpy": np.__version__, "opencv": flow_version},
            "seconds": round(time.monotonic()-started, 3),
            "limits": ["Optical-flow fits are bounded estimates, not calibrated probabilities.",
                       "Frame sampling cannot certify flash safety, comfortable motion or artistic intent.",
                       "No tracking across conditions, missing frames, large gaps or changed dimensions.",
                       "Trails, aliasing and banding require further dedicated evidence; no defect is invented."]}
