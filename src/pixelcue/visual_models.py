"""Explicit, lazy visual model adapters; no remote inference or implicit downloads."""
from __future__ import annotations

import importlib.metadata
import time
from pathlib import Path
import json

from .model_paths import project_models_dir

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


def _transformer(name, image, checkpoint, allow_download, device, cache_dir=None):
    """Single-image compatibility wrapper over the same reusable batch adapter."""
    from .visual_batch import TransformerSession
    root = cache_dir or project_models_dir()/".cache"/"huggingface"
    session = TransformerSession(name, checkpoint, allow_download=allow_download,
                                 device=device, cache_dir=root, prompts=PROMPTS)
    try:
        # The caller retains the model briefly for its original metadata contract.
        return session.infer(image), session.model
    finally:
        session.close()


def run_models(image, names, paths, *, allow_download=False, device="cpu", models_dir=None):
    unknown = set(names) - PROFILES.keys()
    if unknown:
        raise ValueError(f"Unknown visual model(s): {', '.join(sorted(unknown))}")
    receipts = []
    root = project_models_dir(models_dir)
    for name in dict.fromkeys(names):
        installed = root/name
        checkpoint = paths.get(name) or (str(installed) if (installed/"config.json").is_file()
                                        else PROFILES[name]["checkpoint"])
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
                result, model = _transformer(name, image, checkpoint, allow_download, device,
                                             cache_dir=root/".cache"/"huggingface")
                manifest = Path(checkpoint)/"pixelcue-model.json"
                pinned = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {}
                receipt.update(status="completed", result=result,
                    resolved_revision=getattr(model.config, "_commit_hash", None) or pinned.get("revision"),
                    model_type=getattr(model.config, "model_type", None))
                del model
        except (ImportError, OSError, RuntimeError, ValueError, KeyError) as exc:
            receipt.update(status="unavailable", reason=f"{type(exc).__name__}: {str(exc)[:800]}")
        receipt["seconds"] = round(time.monotonic()-started, 3)
        receipts.append(receipt)
    return receipts


def install_models(names, *, models_dir=None):
    """Explicit installation; bounded to selected profiles and framework files."""
    import hashlib
    from huggingface_hub import HfApi, snapshot_download
    root = project_models_dir(models_dir)
    if set(names)-{"clip", "dinov2", "blip"}:
        raise ValueError("Installer supports clip, dinov2 and blip; OCR uses a separate Tesseract installation")
    installed = []
    for name in dict.fromkeys(names):
        checkpoint = PROFILES[name]["checkpoint"]
        info = HfApi().model_info(checkpoint)
        files = [entry.rfilename for entry in info.siblings]
        has_safe = any(f.endswith(".safetensors") for f in files)
        weights = "*.safetensors" if has_safe else "pytorch_model*.bin"
        target = root/name
        snapshot_download(checkpoint, revision=info.sha, local_dir=target,
            cache_dir=root/".cache"/"huggingface", max_workers=2,
            allow_patterns=[weights, "*.json", "*.txt", "*.model", "*.vocab"])
        hashes = {}
        for path in sorted(target.rglob("*")):
            if path.is_file() and ".cache" not in path.relative_to(target).parts and path.name != "pixelcue-model.json":
                with path.open("rb") as stream:
                    hashes[path.relative_to(target).as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
        receipt = {"profile": name, "checkpoint": checkpoint, "revision": info.sha,
            "files_sha256": hashes, "licence": getattr(info.card_data, "license", None)}
        (target/"pixelcue-model.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
        installed.append(receipt)
    return installed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Explicitly install selected PixelCue visual models into project-local models/")
    parser.add_argument("models", nargs="+", choices=["clip", "dinov2", "blip"])
    parser.add_argument("--models-dir", type=Path, default=None)
    args = parser.parse_args()
    for receipt in install_models(args.models, models_dir=args.models_dir):
        print(receipt["profile"], receipt["revision"])
