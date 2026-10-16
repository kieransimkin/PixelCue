"""Explicit, lazy visual model adapters; no remote inference or implicit downloads."""
from __future__ import annotations

import importlib.metadata
import time

PROFILES = {
    "clip": {"checkpoint": "openai/clip-vit-base-patch32", "task": "prompt-relative visual language similarity"},
    "dinov2": {"checkpoint": "facebook/dinov2-small", "task": "visual embedding for reference comparison"},
    "blip": {"checkpoint": "Salesforce/blip-image-captioning-base", "task": "scene caption hypothesis"},
    "ocr": {"checkpoint": "tesseract", "task": "text and bounding box hypotheses"},
}
PROMPTS = {
    "medium": ["a photograph", "a painting", "a vector illustration", "a 3D render", "a collage", "a graphic poster"],
    "light": ["soft diffuse lighting", "hard dramatic lighting", "neon lighting", "flat even lighting", "backlit silhouettes"],
    "material": ["paper and ink texture", "metallic surfaces", "glass surfaces", "organic natural textures", "smooth digital surfaces"],
    "tone": ["a playful cheerful image", "a calm contemplative image", "a tense ominous image", "an energetic celebratory image", "a melancholy image"],
    "setting": ["an urban setting", "a natural landscape", "an indoor room", "outer space", "an abstract composition"],
}


def _version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _transformer(name, image, checkpoint, allow_download, device):
    import torch
    from transformers import AutoImageProcessor, AutoModel, AutoProcessor, BlipForConditionalGeneration, CLIPModel
    kwargs = {"local_files_only": not allow_download, "trust_remote_code": False}
    if device not in ("cpu", "cuda", "mps"):
        raise ValueError("device must be cpu, cuda or mps")
    processor = (AutoImageProcessor if name == "dinov2" else AutoProcessor).from_pretrained(checkpoint, **kwargs)
    cls = {"clip": CLIPModel, "dinov2": AutoModel, "blip": BlipForConditionalGeneration}[name]
    model = cls.from_pretrained(checkpoint, **kwargs).to(device).eval()
    try:
        with torch.inference_mode():
            if name == "clip":
                texts = [p for group in PROMPTS.values() for p in group]
                inputs = processor(text=texts, images=image, return_tensors="pt", padding=True).to(device)
                out = model(**inputs)
                scores = (out.image_embeds @ out.text_embeds.T)[0].float().cpu().tolist()
                groups, start = {}, 0
                for axis, prompts in PROMPTS.items():
                    groups[axis] = sorted([{"prompt": p, "cosine_similarity": float(s)}
                        for p, s in zip(prompts, scores[start:start+len(prompts)])],
                        key=lambda v: -v["cosine_similarity"])
                    start += len(prompts)
                return {"axes": groups, "embedding": out.image_embeds[0].float().cpu().tolist(),
                    "prompt_bank": PROMPTS, "caveat": "Relative to this prompt bank; cosine similarity is not probability or calibrated confidence."}, model
            inputs = processor(images=image, return_tensors="pt").to(device)
            if name == "dinov2":
                vector = model(**inputs).last_hidden_state[:, 0, :][0]
                vector = torch.nn.functional.normalize(vector, dim=0)
                return {"embedding": vector.float().cpu().tolist(), "caveat": "Embedding has no semantic labels; compare only matching model/revision/preprocessing."}, model
            tokens = model.generate(**inputs, max_new_tokens=96, do_sample=False)
            return {"caption": processor.batch_decode(tokens, skip_special_tokens=True)[0],
                    "caveat": "Generated caption may hallucinate subjects, text, materials and intent."}, model
    finally:
        # Do not retain all selected models simultaneously in a long-lived service.
        del model


def run_models(image, names, paths, *, allow_download=False, device="cpu"):
    unknown = set(names) - PROFILES.keys()
    if unknown:
        raise ValueError(f"Unknown visual model(s): {', '.join(sorted(unknown))}")
    receipts = []
    for name in dict.fromkeys(names):
        checkpoint = paths.get(name, PROFILES[name]["checkpoint"])
        receipt = {"name": name, "checkpoint": checkpoint, "task": PROFILES[name]["task"],
            "evidence_class": "interpretation", "allow_download": allow_download,
            "device": device, "dependencies": {n: _version(n) for n in ("torch", "transformers", "pytesseract")}}
        started = time.monotonic()
        try:
            if name == "ocr":
                import pytesseract
                data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, timeout=30)
                words = [{"text": text, "confidence": float(data["conf"][i]),
                    "box_xywh": [int(data[k][i]) for k in ("left", "top", "width", "height")]}
                    for i, text in enumerate(data["text"]) if text.strip() and float(data["conf"][i]) >= 0]
                receipt.update(status="completed", result={"words": words,
                    "caveat": "OCR confidence is engine-specific; boxes use the analysis raster; no font identification."},
                    resolved_revision=str(pytesseract.get_tesseract_version()))
            else:
                result, model = _transformer(name, image, checkpoint, allow_download, device)
                receipt.update(status="completed", result=result,
                    resolved_revision=getattr(model.config, "_commit_hash", None),
                    model_type=getattr(model.config, "model_type", None))
                del model
        except (ImportError, OSError, RuntimeError, ValueError, KeyError) as exc:
            receipt.update(status="unavailable", reason=f"{type(exc).__name__}: {str(exc)[:800]}")
        receipt["seconds"] = round(time.monotonic()-started, 3)
        receipts.append(receipt)
    return receipts
