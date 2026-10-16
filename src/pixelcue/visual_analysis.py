"""Bounded, model-free visual measurements owned by PixelCue.

Nothing here identifies people, establishes provenance, or scores aesthetic quality.
"""
from __future__ import annotations

import hashlib
import io
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps

ANALYSIS_VERSION = "1.0.0"


def _entropy(values, bins=256):
    counts = np.bincount(np.asarray(values, dtype=np.uint8).ravel(), minlength=bins)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def _linear(rgb):
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def _lab(rgb):
    xyz = _linear(rgb) @ np.array([[.4124564, .3575761, .1804375],
                                  [.2126729, .7151522, .0721750],
                                  [.0193339, .1191920, .9503041]]).T
    xyz /= np.array([.95047, 1., 1.08883])
    f = np.where(xyz > (6/29)**3, np.cbrt(xyz), xyz/(3*(6/29)**2)+4/29)
    return np.stack([116*f[:, 1]-16, 500*(f[:, 0]-f[:, 1]),
                     200*(f[:, 1]-f[:, 2])], axis=1)


def load_image(path, *, max_edge=768, background="#ffffff", max_pixels=40_000_000):
    """Decode one bounded first frame, orient, convert ICC, composite, then resize."""
    if not 32 <= max_edge <= 2048:
        raise ValueError("max_edge must be between 32 and 2048")
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", background):
        raise ValueError("background must be #RRGGBB")
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file() or source.stat().st_size > 128 * 1024**2:
        raise ValueError("Expected a file of at most 128 MiB")
    with source.open("rb") as stream:
        raw = stream.read(128*1024**2+1)
    if len(raw) > 128*1024**2:
        raise ValueError("Input grew beyond 128 MiB while reading")
    digest = hashlib.sha256(raw).hexdigest()
    notes = []
    try:
        decoded = Image.open(io.BytesIO(raw))
    except Image.DecompressionBombError as exc:
        raise ValueError("Decoded image exceeds Pillow safety limit") from exc
    with decoded as original:
        if original.width * original.height > max_pixels:
            raise ValueError("Decoded image exceeds pixel limit")
        frames = getattr(original, "n_frames", 1)
        mode, format_name = original.mode, original.format
        image = ImageOps.exif_transpose(original).convert("RGBA")
        dimensions = list(image.size)
        alpha = image.getchannel("A")
        alpha.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        alpha_array = np.asarray(alpha)
        transparency = {"nonopaque_fraction": float((alpha_array < 255).mean()),
                        "fully_transparent_fraction": float((alpha_array == 0).mean())}
        icc = original.info.get("icc_profile")
        color_handling = "assumed sRGB; no embedded ICC profile"
        if icc:
            try:
                rgb = ImageCms.profileToProfile(image.convert("RGB"),
                    ImageCms.ImageCmsProfile(io.BytesIO(icc)), ImageCms.createProfile("sRGB"), outputMode="RGB")
                rgb.putalpha(image.getchannel("A"))
                image = rgb
                color_handling = "embedded ICC converted to sRGB"
            except (ValueError, OSError, ImageCms.PyCMSError):
                color_handling = "invalid/unsupported ICC; assumed sRGB"
                notes.append("ICC conversion failed; colour measurements are provisional.")
        matte = Image.new("RGBA", image.size, background)
        image = Image.alpha_composite(matte, image).convert("RGB")
        if frames > 1:
            notes.append(f"Only frame 0 of {frames} frames analysed; no motion claim.")
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        # Small images stay small; numerical operators below support singleton axes.
        pixels_sha256 = hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()
    return image, {"name": source.name, "sha256": digest, "bytes": len(raw),
        "dimensions": dimensions, "analysis_dimensions": list(image.size), "format": format_name,
        "source_mode": mode, "frames": frames, "background": background,
        "color_handling": color_handling, "alpha": transparency,
        "analysed_pixels_sha256": pixels_sha256, "notes": notes}


def _palette(rgb, count=6):
    flat = rgb.reshape(-1, 3)
    # Deterministic stratified sample: bounded time/memory regardless of input resolution.
    sample = flat[np.linspace(0, len(flat)-1, min(len(flat), 8192), dtype=int)]
    lab = _lab(sample)
    centers = [lab[np.argmin(lab[:, 0])]]
    for _ in range(count-1):
        distance = ((lab[:, None, :] - np.array(centers)[None, :, :])**2).sum(2).min(1)
        if distance.max() < 1e-10:
            break
        centers.append(lab[distance.argmax()])
    centers = np.array(centers)
    for _ in range(20):
        labels = ((lab[:, None, :] - centers[None, :, :])**2).sum(2).argmin(1)
        updated = np.array([lab[labels == k].mean(0) if np.any(labels == k) else centers[k]
                            for k in range(len(centers))])
        if np.allclose(updated, centers, atol=.05):
            centers = updated
            break
        centers = updated
    labels = ((lab[:, None, :] - centers[None, :, :])**2).sum(2).argmin(1)
    result = []
    for k in range(len(centers)):
        mask = labels == k
        if not mask.any():
            continue
        color = sample[mask].mean(0)
        ints = np.clip(np.rint(color * 255), 0, 255).astype(int)
        result.append({"hex": "#"+"".join(f"{x:02x}" for x in ints),
            "srgb": ints.tolist(), "sample_fraction": float(mask.mean()),
            "relative_luminance": float(_linear(color) @ [.2126, .7152, .0722])})
    return sorted(result, key=lambda c: -c["sample_fraction"])


def measure_image(image):
    rgb = np.asarray(image, dtype=float) / 255
    luminance = _linear(rgb) @ np.array([.2126, .7152, .0722])
    gray = rgb @ np.array([.2126, .7152, .0722])
    h, w = gray.shape
    gx = np.gradient(gray, axis=1) if w > 1 else np.zeros_like(gray)
    gy = np.gradient(gray, axis=0) if h > 1 else np.zeros_like(gray)
    edges = np.hypot(gx, gy)
    orientation, _ = np.histogram(np.arctan2(gy, gx) % np.pi, bins=8,
        range=(0, np.pi), weights=edges)
    orientation /= max(float(orientation.sum()), 1e-12)
    hsv = np.asarray(image.convert("HSV"), dtype=float)/255
    hue_counts, _ = np.histogram(hsv[:, :, 0], bins=12, range=(0, 1), weights=hsv[:, :, 1])
    hue_counts /= max(float(hue_counts.sum()), 1e-12)
    palette = _palette(rgb)
    contrasts = [{"a": a["hex"], "b": b["hex"],
        "ratio": (max(a["relative_luminance"], b["relative_luminance"])+.05)/
                 (min(a["relative_luminance"], b["relative_luminance"])+.05)}
        for i, a in enumerate(palette) for b in palette[i+1:]]
    # Spectral residual, a classical signal novelty proxy. Not gaze/subject detection.
    fft = np.fft.fft2(gray)
    log_amplitude = np.log(np.abs(fft)+1e-9)
    smooth = sum(np.roll(np.roll(log_amplitude, y, 0), x, 1)
                 for y in (-1, 0, 1) for x in (-1, 0, 1))/9
    saliency = np.abs(np.fft.ifft2(np.exp(log_amplitude-smooth+1j*np.angle(fft))))**2
    if np.std(gray) < 1e-9:
        saliency[:] = 0
    total = float(saliency.sum())
    yy, xx = np.mgrid[:h, :w]
    centroid = ([float((xx*saliency).sum()/total/max(w-1, 1)),
                 float((yy*saliency).sum()/total/max(h-1, 1))] if total > 0 else None)
    centered = gray-gray.mean()
    power = np.abs(np.fft.fft2(centered))**2
    fy, fx = np.meshgrid(np.fft.fftfreq(h), np.fft.fftfreq(w), indexing="ij")
    radius = np.hypot(fx, fy)
    power_total = max(float(power.sum()), 1e-12)
    bands = [float(power[(radius >= a) & (radius < b)].sum()/power_total)
             for a, b in [(0, .0625), (.0625, .25), (.25, 1)]]
    quant = np.minimum((gray*16).astype(int), 15)
    glcms = []
    for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        if h <= dy or w <= abs(dx):
            continue
        left = quant[:h-dy, :w-dx] if dx >= 0 else quant[:h-dy, -dx:]
        right = quant[dy:, dx:] if dx >= 0 else quant[dy:, :w+dx]
        glcms.append(np.bincount((left.ravel()*16+right.ravel()), minlength=256).reshape(16, 16))
    matrix = sum(glcms) if glcms else np.eye(16)
    matrix = (matrix+matrix.T).astype(float)
    matrix /= matrix.sum()
    ii, jj = np.mgrid[:16, :16]
    padded = np.pad(gray, 1, mode="edge")
    lbp = np.zeros_like(quant, dtype=np.uint8)
    for bit, (dy, dx) in enumerate([(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]):
        lbp |= ((padded[1+dy:1+dy+h, 1+dx:1+dx+w] >= gray).astype(np.uint8) << bit)
    lbp_hist = np.bincount(lbp.ravel(), minlength=256)/lbp.size
    cells = []
    for row, ys in enumerate(np.array_split(np.arange(h), min(4, h))):
        for col, xs in enumerate(np.array_split(np.arange(w), min(4, w))):
            region = np.ix_(ys, xs)
            cells.append({"row": row, "column": col, "mean_luminance": float(luminance[region].mean()),
                "mean_saturation": float(hsv[:, :, 1][region].mean()),
                "edge_density": float((edges[region] > .08).mean()),
                "saliency_fraction": float(saliency[region].sum()/total) if total else 0})
    return {"colour": {"palette": palette, "palette_method": "deterministic Lab k-means, <=8192 samples; 6 centres; approximate RGB means",
        "mean_saturation": float(hsv[:, :, 1].mean()), "hue_histogram_12": hue_counts.tolist(),
        "relative_luminance_percentiles": np.quantile(luminance, [0, .05, .5, .95, 1]).tolist(),
        "luminance_std": float(luminance.std()), "dark_fraction_below_0_05": float((luminance < .05).mean()),
        "light_fraction_above_0_8": float((luminance > .8).mean()), "palette_pair_contrasts": contrasts,
        "contrast_caveat": "Palette-pair ratios do not establish actual text/background adjacency or accessibility conformance."},
        "texture": {"grayscale_entropy_bits": _entropy(gray*255),
            "lbp_histogram_256": lbp_hist.tolist(), "lbp_entropy_bits": _entropy(lbp),
            "glcm_contrast": float((matrix*(ii-jj)**2).sum()),
            "glcm_homogeneity": float((matrix/(1+(ii-jj)**2)).sum()),
            "glcm_energy": float((matrix**2).sum()),
            "method": "8-neighbour LBP; symmetric 16-level GLCM pooled over four one-pixel offsets"},
        "composition": {"edge_density_above_0_08": float((edges > .08).mean()),
            "gradient_orientation_histogram_8": orientation.tolist(),
            "horizontal_mirror_mae": float(np.abs(gray-gray[:, ::-1]).mean()),
            "vertical_mirror_mae": float(np.abs(gray-gray[::-1]).mean()),
            "spectral_residual_centroid_xy": centroid, "grid": cells,
            "saliency_caveat": "Spectral residual measures signal novelty, not gaze, semantics or guaranteed negative space."},
        "frequency": {"radial_power_fractions_low_mid_high": bands,
            "bands_cycles_per_pixel": [[0, .0625], [.0625, .25], [.25, 1]],
            "method": "Mean-removed grayscale 2D FFT; depends on analysis size"}}


def analyse_image(path, *, models=(), model_paths=None, allow_download=False,
                  max_edge=768, background="#ffffff", device="cpu"):
    """Reusable PixelCue API. Measurements plus explicit independent model receipts."""
    from .visual_models import run_models
    image, source = load_image(path, max_edge=max_edge, background=background)
    try:
        return {"schema": "pixelcue.visual-analysis/1", "analysis_version": ANALYSIS_VERSION,
            "source": source, "measurements": measure_image(image),
            "models": run_models(image, models, model_paths or {}, allow_download=allow_download, device=device),
            "limits": ["Measurements describe a composited, bounded analysis raster, not untouched source pixels.",
                "One image does not establish recurring brand rules, lineage, intent, rights or audience response.",
                "Motion, responsive behaviour, font identity, consent and actual legibility require separate evidence."]}
    finally:
        image.close()
