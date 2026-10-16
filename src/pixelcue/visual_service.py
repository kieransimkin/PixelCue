"""Composable visual profile service; reuses PixelCue's existing VLM service."""
from __future__ import annotations

import threading

from .visual_analysis import analyse_image


class VisualAnalysisService:
    def __init__(self, keyword_service=None):
        self.keyword_service = keyword_service
        self._lock = threading.Lock()

    def analyse(self, path, *, models=(), model_paths=None, keyword_models=(),
                allow_download=False, max_edge=768, background="#ffffff", device="cpu"):
        # Bound GPU work across concurrent analysis requests. Keyword service owns its lock.
        with self._lock:
            result = analyse_image(path, models=models, model_paths=model_paths,
                allow_download=allow_download, max_edge=max_edge, background=background, device=device)
            for profile in dict.fromkeys(keyword_models):
                receipt = {"name": f"pixelcue-tagger:{profile}", "evidence_class": "interpretation",
                           "allow_download": allow_download}
                if not allow_download:
                    receipt.update(status="unavailable", reason="Existing PixelCue taggers may download weights; allow_download must be explicit.")
                elif self.keyword_service is None:
                    receipt.update(status="unavailable", reason="No PixelCue keyword service supplied.")
                else:
                    try:
                        receipt.update(status="completed", result={"tags": self.keyword_service.extract_keywords(path, model=profile),
                            "caveat": "PixelCue VLM tags are model interpretations, not verified subject facts."})
                    except Exception as exc:
                        receipt.update(status="unavailable", reason=f"{type(exc).__name__}: {str(exc)[:800]}")
                result["models"].append(receipt)
            return result


def compare_profiles(profiles):
    """Axis-specific distances, no invented overall brand or aesthetic score."""
    import numpy as np

    def js(a, b):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if a.sum() == 0 or b.sum() == 0:
            return None
        a, b = a/a.sum(), b/b.sum()
        midpoint = (a+b)/2
        def kl(p):
            keep = p > 0
            return float((p[keep]*np.log2(p[keep]/midpoint[keep])).sum())
        return (kl(a)+kl(b))/2

    pairs = []
    for i, a in enumerate(profiles):
        for j in range(i+1, len(profiles)):
            b = profiles[j]
            ca, cb = a["measurements"], b["measurements"]
            pair = {"a": i, "b": j, "exact_file_duplicate": a["source"]["sha256"] == b["source"]["sha256"],
                "same_analysis_raster": a["source"]["analysed_pixels_sha256"] == b["source"]["analysed_pixels_sha256"],
                "hue_js_divergence_bits": js(ca["colour"]["hue_histogram_12"], cb["colour"]["hue_histogram_12"]),
                "texture_lbp_js_divergence_bits": js(ca["texture"]["lbp_histogram_256"], cb["texture"]["lbp_histogram_256"]),
                "saturation_absolute_difference": abs(ca["colour"]["mean_saturation"]-cb["colour"]["mean_saturation"]),
                "model_cosine_similarity": []}
            for ma in a["models"]:
                for mb in b["models"]:
                    va, vb = ma.get("result", {}).get("embedding"), mb.get("result", {}).get("embedding")
                    if (va and vb and len(va) == len(vb) and ma["name"] == mb["name"] and
                        ma.get("checkpoint") == mb.get("checkpoint") and ma.get("resolved_revision") and
                        ma.get("resolved_revision") == mb.get("resolved_revision")):
                        denom = np.linalg.norm(va)*np.linalg.norm(vb)
                        if denom:
                            pair["model_cosine_similarity"].append({"name": ma["name"],
                                "cosine_similarity": float(np.dot(va, vb)/denom)})
            pairs.append(pair)
    return {"pairs": pairs, "limits": ["Distances describe separate measured axes; similarity does not establish a shared identity.",
        "Same analysis raster is only a derivative/lineage candidate. Crops and near-duplicates require review.",
        "Absent model revision prevents embedding comparisons. Roles, dates, authority and independent lineages require supplied evidence."]}
