"""Independent local optical flow; never equate a failed global fit with stillness.

This is an engineering measurement with explicit rejection diagnostics, not an
object recogniser or a causal audio model. Calls decode no media and load no ML.
"""
from __future__ import annotations
import math
import numpy as np


def local_flow(previous, current, dt):
    import cv2
    empty = {"status": "unavailable", "reason": "insufficient consistent local flow"}
    if not math.isfinite(dt) or not 0 < dt <= .65 or previous.size != current.size:
        return dict(empty, reason="unsupported local-flow interval or dimensions")
    a = np.asarray(previous.convert("L"), dtype=np.uint8)
    b = np.asarray(current.convert("L"), dtype=np.uint8)
    if min(a.std(), b.std()) < 3:
        return dict(empty, reason="uniform raster")
    # Compensate a GLOBAL exposure offset/gain, but not changing local structure.
    # Original luminance is retained by visual_sequence for audio comparisons.
    def normalise(x):
        lo, hi = np.percentile(x, (5, 95))
        if hi - lo < 8:
            return x
        return np.clip((x.astype(np.float32)-lo)*220/(hi-lo)+16, 0, 255).astype(np.uint8)
    a, b = normalise(a), normalise(b)
    flow = cv2.calcOpticalFlowFarneback(a, b, None, .5, 3, 21, 4, 7, 1.5, 0)
    reverse = cv2.calcOpticalFlowFarneback(b, a, None, .5, 3, 21, 4, 7, 1.5, 0)
    h, w = a.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    mx, my = xx+flow[..., 0], yy+flow[..., 1]
    back = cv2.remap(reverse, mx, my, cv2.INTER_LINEAR)
    warped = cv2.remap(b, mx, my, cv2.INTER_LINEAR)
    speed = np.linalg.norm(flow, axis=2)
    consistency = np.linalg.norm(flow+back, axis=2)
    texture = cv2.cornerMinEigenVal(a, 5) > .00005
    border = (xx >= 5) & (yy >= 5) & (xx < w-5) & (yy < h-5)
    domain = border & texture
    valid = domain & (mx >= 5) & (my >= 5) & (mx < w-5) & (my < h-5)
    valid &= np.isfinite(flow).all(axis=2) & (consistency < .8+.05*speed)
    photo = np.abs(a.astype(np.float32)-warped.astype(np.float32))
    valid &= photo < 18
    supported = float(valid.mean())
    ratio = float(valid.sum()/max(1, domain.sum()))
    if supported < .04 or ratio < .45:
        return dict(empty, valid_fraction=supported, texture_support=ratio)
    diag = math.hypot(w, h)
    velocity = flow[valid]/diag/dt
    central = np.median(velocity, axis=0)
    dispersion = float(np.median(np.linalg.norm(velocity-central, axis=1)))
    median_speed = float(np.median(speed[valid])/diag/dt)
    moving = float((speed[valid] > .2).mean())
    return {"status": "measured", "reason": None,
            "speed_per_second": median_speed, "dispersion_per_second": dispersion,
            "moving_fraction": moving, "valid_fraction": supported,
            "texture_support": ratio, "photometric_error": float(np.median(photo[valid])),
            "method": "independent bidirectional Farneback with exposure/photometric consistency",
            "limits": "Apparent local motion only; not automatically deformation, trails or quality."}
