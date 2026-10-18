"""Bounded model-major visual batches: load each neural adapter once, not per frame.

This is a PixelCue API, not a parallel model implementation in PrismMoves.
Single-image APIs and their default prompt bank remain unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

from .model_paths import project_models_dir
from .visual_analysis import ANALYSIS_VERSION, load_image, measure_image
from .visual_models import PROFILES, PROMPTS, _version

BATCH_VERSION = "1.0.0"
# Deliberately spatial only: a picture of a spiral is NOT evidence of rotation.
VISUALISER_PROMPTS = {
    "motif": ["abstract concentric rings", "abstract radial flower-like geometry",
              "a field of small glowing particles", "a geometric grid or lattice",
              "striped or wave-like abstract patterns", "branching fractal patterns",
              "soft cloud-like abstract forms", "a largely featureless image",
              "a recognisable photographic image", "an abstract pattern without a clear motif"],
    "light": ["luminous neon outlines against darkness", "soft diffuse glow",
              "flat even colours without glow", "bright washed-out white highlights",
              "dark subdued colours", "sharp high-contrast black and white"],
    "surface": ["smooth digital gradients", "grainy speckled texture", "crisp thin lines",
                "soft blurred edges", "blocky pixelated shapes", "complex fine filigree"],
    "structure": ["radially symmetric abstract geometry", "bilaterally mirrored abstract geometry",
                  "an asymmetric abstract composition", "a repeating tiled abstract pattern",
                  "a mostly empty composition"],
}


def prompt_digest(bank):
    return hashlib.sha256(json.dumps(bank, sort_keys=True).encode()).hexdigest()


class TransformerSession:
    """Exactly one selected model in memory; deterministic per-image generation."""
    def __init__(self, name, checkpoint, *, allow_download, device, cache_dir, prompts):
        import torch
        from transformers import (AutoImageProcessor, AutoModel, AutoProcessor,
                                  BlipForConditionalGeneration, CLIPModel)
        self.torch, self.name, self.prompts, self.device = torch, name, prompts, device
        kwargs = {"local_files_only": not allow_download, "trust_remote_code": False,
                  "cache_dir": str(cache_dir)}
        processor = AutoImageProcessor if name == "dinov2" else AutoProcessor
        self.processor = processor.from_pretrained(checkpoint, **kwargs)
        cls = {"clip": CLIPModel, "dinov2": AutoModel, "blip": BlipForConditionalGeneration}[name]
        self.model = cls.from_pretrained(checkpoint, **kwargs).to(device).eval()
        self.revision = getattr(self.model.config, "_commit_hash", None)
        self.model_type = getattr(self.model.config, "model_type", None)

    def infer(self, image):
        torch, name, model = self.torch, self.name, self.model
        with torch.inference_mode():
            if name == "clip":
                texts = [p for values in self.prompts.values() for p in values]
                inputs = self.processor(text=texts, images=image, return_tensors="pt", padding=True).to(self.device)
                output = model(**inputs)
                scores = (output.image_embeds @ output.text_embeds.T)[0].float().cpu().tolist()
                groups, start = {}, 0
                for axis, prompts in self.prompts.items():
                    groups[axis] = sorted([{"prompt": text, "cosine_similarity": float(score)}
                                          for text, score in zip(prompts, scores[start:start+len(prompts)])],
                                         key=lambda row: -row["cosine_similarity"])
                    start += len(prompts)
                return {"axes": groups, "embedding": output.image_embeds[0].float().cpu().tolist(), "prompt_bank": self.prompts,
                        "prompt_bank_sha256": prompt_digest(self.prompts),
                        "caveat": "Prompt-relative hypotheses; no calibrated probability or motion inference."}
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            if name == "dinov2":
                vector = torch.nn.functional.normalize(model(**inputs).last_hidden_state[:, 0, :][0], dim=0)
                return {"embedding": vector.float().cpu().tolist(), "caveat": "No labels inferred from embedding."}
            tokens = model.generate(**inputs, max_new_tokens=96, do_sample=False)
            return {"caption": self.processor.batch_decode(tokens, skip_special_tokens=True)[0],
                    "caveat": "Single-frame caption hypothesis, not a movement observation."}

    def close(self):
        del self.model
        if self.device == "cuda" and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def analyse_batch(paths, *, models=("clip", "blip"), keyword_models=(), model_paths=None,
                  allow_download=False, models_dir=None, device="cpu", max_edge=384,
                  background="#000000", prompt_profile="visualiser", session_factory=None,
                  keyword_service=None):
    """Analyse <=64 selected rasters; each model is loaded once for this call.

    Callers should hash/configure caches and select frames BEFORE this method.
    Missing models have independent receipts. No frame sequence is inferred here.
    ``session_factory`` is an injectable test seam, not a hosted inference route.
    """
    paths = list(paths)
    if not paths or len(paths) > 64:
        raise ValueError("A semantic batch needs 1..64 explicit images")
    if set(models) - PROFILES.keys():
        raise ValueError("Unknown visual model")
    if device not in {"cpu", "cuda", "mps"}:
        raise ValueError("Unsupported device")
    if prompt_profile not in {"visualiser", "general"}:
        raise ValueError("Unknown prompt profile")
    root = project_models_dir(models_dir)
    prompts = VISUALISER_PROMPTS if prompt_profile == "visualiser" else PROMPTS
    session_factory = session_factory or TransformerSession
    model_paths = model_paths or {}
    if set(model_paths) - set(models):
        raise ValueError("Model path overrides must name a selected model")
    loaded, items = [], []
    try:
        for path in paths:
            try:
                image, source = load_image(path, max_edge=max_edge, background=background)
                analysis = {"schema": "pixelcue.visual-analysis/1", "analysis_version": ANALYSIS_VERSION,
                            "source": source, "measurements": measure_image(image), "models": [],
                            "limits": ["Captions and prompt matches are hypotheses, not motion or intent."]}
                items.append({"status": "completed", "name": Path(path).name, "analysis": analysis})
                loaded.append(image)
            except (OSError, ValueError):
                items.append({"status": "unavailable", "name": Path(path).name,
                              "reason": "Image unavailable or invalid"})
                loaded.append(None)
        for name in dict.fromkeys(models):
            installed = root / name
            checkpoint = model_paths.get(name) or (str(installed) if (installed / "config.json").is_file()
                                                   else PROFILES[name]["checkpoint"])
            base = {"name": name, "checkpoint": checkpoint, "evidence_class": "interpretation",
                    "allow_download": allow_download, "device": device,
                    "dependencies": {n: _version(n) for n in ("torch", "transformers", "pytesseract")},
                    "prompt_profile": prompt_profile, "prompt_bank_sha256": prompt_digest(prompts)}
            started, session, failure = time.monotonic(), None, None
            try:
                if name != "ocr":
                    session = session_factory(name, checkpoint, allow_download=allow_download,
                                              device=device, cache_dir=root/".cache"/"huggingface", prompts=prompts)
                base["load_seconds"] = round(time.monotonic()-started, 3)
            except (ImportError, OSError, ValueError, RuntimeError) as exc:
                failure = type(exc).__name__ + ": selected model unavailable"
            try:
                for item, image in zip(items, loaded):
                    if image is None:
                        continue
                    receipt = dict(base)
                    started = time.monotonic()
                    if failure:
                        receipt.update(status="unavailable", reason=failure)
                    else:
                        try:
                            if name == "ocr":
                                # Optional, normally disabled for an abstract visualiser.
                                from .visual_models import run_models
                                receipt = run_models(image, ["ocr"], {}, models_dir=root)[0]
                            else:
                                result = session.infer(image)
                                manifest = Path(checkpoint) / "pixelcue-model.json"
                                pin = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {}
                                receipt.update(status="completed", result=result,
                                               resolved_revision=session.revision or pin.get("revision"),
                                               model_type=session.model_type)
                        except (ImportError, OSError, ValueError, RuntimeError, KeyError) as exc:
                            receipt.update(status="unavailable", reason=type(exc).__name__ + ": model inference failed")
                    receipt["seconds"] = round(time.monotonic()-started, 3)
                    item["analysis"]["models"].append(receipt)
            finally:
                if session is not None:
                    session.close()
        if keyword_models and allow_download and keyword_service is None:
            try:
                from .service import KeywordExtractionService
                keyword_service = KeywordExtractionService()
            except (ImportError, OSError, RuntimeError):
                keyword_service = None
        for item, path in zip(items, paths):
            if item["status"] != "completed":
                continue
            for name in dict.fromkeys(keyword_models):
                receipt = {"name": "pixelcue-tagger:"+name, "evidence_class": "interpretation",
                           "allow_download": allow_download}
                started = time.monotonic()
                if not allow_download:
                    receipt.update(status="unavailable", reason="Tagger download permission not given")
                elif keyword_service is None:
                    receipt.update(status="unavailable", reason="PixelCue keyword service unavailable")
                else:
                    try:
                        receipt.update(status="completed", result={"tags": keyword_service.extract_keywords(path, model=name)})
                    except Exception as exc:
                        receipt.update(status="unavailable", reason=type(exc).__name__ + ": tagger failed")
                receipt["seconds"] = round(time.monotonic()-started, 3)
                item["analysis"]["models"].append(receipt)
        return {"schema": "pixelcue.visual-batch/1", "batch_version": BATCH_VERSION,
                "prompt_profile": prompt_profile, "prompt_bank_sha256": prompt_digest(prompts),
                "items": items, "strategy": "one model loaded at a time for the selected batch"}
    finally:
        for image in loaded:
            if image is not None:
                image.close()
