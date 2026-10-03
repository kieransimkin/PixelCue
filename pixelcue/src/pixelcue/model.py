from __future__ import annotations

import importlib.metadata
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Callable

from PIL import Image

DEFAULT_MODEL_ID = "heavlav/llama-joycaption-beta-one-hf-llava-4bit"
MODEL_ID = os.environ.get("PIXELCUE_JOYCAPTION_MODEL", DEFAULT_MODEL_ID)
MODEL_APPROX_GB = 5.96
MODEL_MANIFEST_NAME = ".pixelcue-model-manifest.json"

# PixelCue wants observable byte progress. hf-xet stores active transfer chunks
# outside local_dir, which makes local byte accounting opaque, so use the normal
# Hugging Face HTTP/LFS downloader unless the user explicitly chose otherwise.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

TAG_PROMPT = """Write a list of Booru-like tags for this image.
Output ONLY comma-separated tags, with no prose.
Include concrete subject, object, setting, action, clothing, visual-style, and composition tags where useful.
You MUST classify the image as SFW, suggestive, or NSFW.
If the image is NSFW, include the exact tag NSFW.
If any person's face is visibly present, include the exact tag FaceIdentity.
Do not identify who a person is. FaceIdentity only means that a visible human face is present.
Keep the tag list concise and avoid duplicate tags."""


class JoyCaptionError(RuntimeError):
    pass


def parse_tags(text: str) -> list[str]:
    text = text.strip()
    text = re.sub(r"^```(?:text|json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    bits = re.split(r"[,;\n]+", text)
    out: list[str] = []
    seen: set[str] = set()

    for bit in bits:
        tag = bit.strip().strip("[](){}\"'`").strip()
        tag = re.sub(r"^\s*[-*]\s*", "", tag)
        tag = re.sub(r"^\d+\.\s*", "", tag)
        if not tag:
            continue

        low = tag.casefold()
        if low == "nsfw" or re.fullmatch(r"(rating:)?\s*nsfw", low):
            tag = "NSFW"
        elif low in {"faceidentity", "face_identity", "face identity"}:
            tag = "FaceIdentity"
        elif low in {"sfw", "rating:sfw", "rating: sfw"}:
            tag = "SFW"
        elif low in {"suggestive", "rating:suggestive", "rating: suggestive"}:
            tag = "suggestive"

        key = tag.casefold()
        if key not in seen:
            seen.add(key)
            out.append(tag)

    if re.search(r"\bnsfw\b", text, re.I) and "nsfw" not in seen:
        out.append("NSFW")
    if re.search(r"\bface[_ ]?identity\b", text, re.I) and "faceidentity" not in seen:
        out.append("FaceIdentity")
    return out


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "unknown"


def default_model_dir() -> Path:
    from platformdirs import user_cache_dir
    return Path(user_cache_dir("PixelCue", "DanceFlow")) / "models" / "joycaption-beta-one-4bit"


class JoyCaption4Bit:
    def __init__(
        self,
        model_id: str = MODEL_ID,
        model_dir: str | Path | None = None,
        status_callback: Callable[[str], None] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        self.model_id = model_id
        self.model_dir = Path(
            os.environ.get("PIXELCUE_MODEL_DIR", str(model_dir or default_model_dir()))
        ).expanduser()
        self.status_callback = status_callback or (lambda _: None)
        self.progress_callback = progress_callback or (lambda _done, _total: None)

        self.processor = None
        self.model = None
        self.device = None
        self._downloaded_path: Path | None = None
        self._repo_files: list[tuple[str, int]] = []
        self._repo_total_bytes: int = 0

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def _status(self, message: str) -> None:
        self.status_callback(message)

    def _progress(self, done: int, total: int) -> None:
        self.progress_callback(max(0, int(done)), max(0, int(total)))

    def diagnostics(self) -> str:
        bits = [
            f"model_id={self.model_id}",
            f"model_dir={self.model_dir}",
            f"torch={_version('torch')}",
            f"transformers={_version('transformers')}",
            f"bitsandbytes={_version('bitsandbytes')}",
            f"accelerate={_version('accelerate')}",
            f"huggingface-hub={_version('huggingface-hub')}",
        ]
        try:
            import torch
            bits.append(f"cuda_available={torch.cuda.is_available()}")
            bits.append(f"torch_cuda={torch.version.cuda}")
            if torch.cuda.is_available():
                idx = torch.cuda.current_device()
                bits.append(f"cuda_device={idx}")
                bits.append(f"gpu={torch.cuda.get_device_name(idx)}")
        except Exception as e:
            bits.append(f"torch_diagnostics_error={e}")
        return "\n".join(bits)

    @property
    def manifest_path(self) -> Path:
        return self.model_dir / MODEL_MANIFEST_NAME

    def _write_manifest(self, files: list[tuple[str, int]], total: int) -> None:
        payload = {
            "model_id": self.model_id,
            "total_bytes": int(total),
            "files": [{"path": name, "size": int(size)} for name, size in files],
        }
        try:
            self.manifest_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except OSError:
            pass

    def _read_manifest(self) -> tuple[list[tuple[str, int]], int] | None:
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if payload.get("model_id") != self.model_id:
                return None
            files = [
                (str(item["path"]), int(item["size"]))
                for item in payload.get("files", [])
                if int(item.get("size", 0)) >= 0
            ]
            total = int(payload.get("total_bytes", 0))
            if files and total > 0:
                return files, total
        except Exception:
            pass
        return None

    def _fetch_repository_manifest(self) -> tuple[list[tuple[str, int]], int]:
        """Fetch exact file sizes from Hugging Face metadata."""
        from huggingface_hub import HfApi

        self._status(f"Querying JoyCaption repository metadata: {self.model_id}")
        info = HfApi().model_info(self.model_id, files_metadata=True)

        files: list[tuple[str, int]] = []
        for sibling in info.siblings or []:
            name = getattr(sibling, "rfilename", None)
            size = getattr(sibling, "size", None)
            if name and size is not None:
                files.append((str(name), int(size)))

        total = sum(size for _, size in files)
        if total <= 0:
            raise RuntimeError("Hugging Face did not return model file sizes.")

        self._repo_files = files
        self._repo_total_bytes = total
        self._write_manifest(files, total)
        return files, total

    def _get_repository_manifest(self) -> tuple[list[tuple[str, int]], int]:
        try:
            return self._fetch_repository_manifest()
        except Exception:
            cached = self._read_manifest()
            if cached:
                self._repo_files, self._repo_total_bytes = cached
                return cached
            raise

    def _completed_final_file_bytes(self, files: list[tuple[str, int]]) -> int:
        done = 0
        for rel, expected in files:
            p = self.model_dir / Path(rel)
            try:
                if p.is_file():
                    done += min(int(p.stat().st_size), expected)
            except OSError:
                pass
        return done

    def _partial_download_bytes(self) -> int:
        """Count Hugging Face local_dir .incomplete transfer files."""
        cache_root = self.model_dir / ".cache" / "huggingface" / "download"
        if not cache_root.exists():
            return 0

        total = 0
        try:
            for p in cache_root.rglob("*.incomplete"):
                try:
                    total += int(p.stat().st_size)
                except OSError:
                    pass
        except OSError:
            pass
        return total

    def _downloaded_bytes(self, files: list[tuple[str, int]], total: int) -> int:
        # Completed files plus resumable partial files. Hugging Face removes/renames
        # .incomplete files on completion, so these do not normally double count.
        done = self._completed_final_file_bytes(files) + self._partial_download_bytes()
        return min(max(done, 0), total)

    def _is_complete(self, files: list[tuple[str, int]]) -> bool:
        for rel, expected in files:
            p = self.model_dir / Path(rel)
            try:
                if not p.is_file() or int(p.stat().st_size) != expected:
                    return False
            except OSError:
                return False
        return bool(files)

    def ensure_downloaded(self) -> Path:
        """Materialise the model locally while reporting exact byte progress."""
        if self._downloaded_path and self._downloaded_path.exists():
            return self._downloaded_path

        from huggingface_hub import snapshot_download

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._status(f"Checking JoyCaption 4-bit model cache: {self.model_dir}")

        try:
            files, total = self._get_repository_manifest()
        except Exception as e:
            raise JoyCaptionError(
                f"Could not obtain JoyCaption model size/file metadata: "
                f"{type(e).__name__}: {e}\n\n"
                f"Repository: {self.model_id}\nDestination: {self.model_dir}\n\n"
                f"{self.diagnostics()}"
            ) from e

        done = self._downloaded_bytes(files, total)
        self._progress(done, total)

        if self._is_complete(files):
            self._downloaded_path = self.model_dir
            self._progress(total, total)
            self._status(f"JoyCaption 4-bit: cached and complete at {self.model_dir}")
            return self.model_dir

        self._status(f"Downloading JoyCaption 4-bit model to {self.model_dir}")

        result: dict[str, object] = {}
        finished = threading.Event()

        def do_download() -> None:
            try:
                result["path"] = snapshot_download(
                    repo_id=self.model_id,
                    local_dir=str(self.model_dir),
                    local_files_only=False,
                    max_workers=4,
                )
            except BaseException as exc:
                result["error"] = exc
            finally:
                finished.set()

        download_thread = threading.Thread(
            target=do_download,
            name="pixelcue-huggingface-download",
            daemon=True,
        )
        download_thread.start()

        # snapshot_download has no byte callback. Poll its final and .incomplete
        # files so the GUI still gets continuously updated byte counts.
        last_done = -1
        while not finished.wait(timeout=0.25):
            done = self._downloaded_bytes(files, total)
            if done != last_done:
                last_done = done
                self._progress(done, total)

        download_thread.join()
        done = self._downloaded_bytes(files, total)
        self._progress(done, total)

        error = result.get("error")
        if error is not None:
            raise JoyCaptionError(
                f"JoyCaption model download failed: "
                f"{type(error).__name__}: {error}\n\n"
                f"Downloaded: {done:,} / {total:,} bytes\n"
                f"Destination: {self.model_dir}\n"
                f"Repository: {self.model_id}\n\n{self.diagnostics()}"
            ) from error

        if not self._is_complete(files):
            done = self._downloaded_bytes(files, total)
            raise JoyCaptionError(
                "JoyCaption download finished but the local checkpoint is incomplete.\n\n"
                f"Downloaded: {done:,} / {total:,} bytes\n"
                f"Destination: {self.model_dir}\n"
                f"Repository: {self.model_id}"
            )

        self._downloaded_path = Path(str(result.get("path") or self.model_dir))
        self._progress(total, total)
        self._status(f"JoyCaption 4-bit: download complete at {self.model_dir}")
        return self._downloaded_path

    def load(self) -> None:
        if self.loaded:
            return

        model_path = self.ensure_downloaded()
        self._status("Verifying checkpoint and loading JoyCaption 4-bit onto GPU …")

        try:
            import torch
            from transformers import AutoModelForMultimodalLM, AutoProcessor

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "A CUDA-capable NVIDIA GPU was not detected by PyTorch. "
                    "Install a CUDA-enabled PyTorch build and compatible NVIDIA driver."
                )

            self.processor = AutoProcessor.from_pretrained(
                str(model_path),
                local_files_only=True,
            )
            self.model = AutoModelForMultimodalLM.from_pretrained(
                str(model_path),
                device_map="auto",
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
            self.model.eval()
            self.device = self.model.device
            self._status(
                f"JoyCaption 4-bit: ready on "
                f"{torch.cuda.get_device_name(torch.cuda.current_device())}"
            )
        except Exception as e:
            self.model = None
            self.processor = None
            raise JoyCaptionError(
                f"JoyCaption model load failed: {type(e).__name__}: {e}\n\n"
                f"{self.diagnostics()}"
            ) from e

    def tags_for_image(self, image: Image.Image) -> list[str]:
        self.load()

        try:
            import torch

            assert self.processor is not None
            assert self.model is not None

            image = image.convert("RGB")
            convo = [
                {"role": "system", "content": "You are a helpful image captioner."},
                {"role": "user", "content": TAG_PROMPT},
            ]
            prompt = self.processor.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.processor(
                text=[prompt],
                images=[image],
                return_tensors="pt",
            ).to(self.model.device)

            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    suppress_tokens=None,
                    use_cache=True,
                    temperature=0.6,
                    top_k=None,
                    top_p=0.9,
                )[0]

            response_ids = generated[inputs["input_ids"].shape[1]:]
            text = self.processor.tokenizer.decode(
                response_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()

            if not text:
                raise RuntimeError("JoyCaption returned an empty response.")

            tags = parse_tags(text)
            if not tags:
                raise RuntimeError(
                    f"Could not parse any tags from JoyCaption response: {text!r}"
                )
            return tags

        except JoyCaptionError:
            raise
        except Exception as e:
            raise JoyCaptionError(
                f"JoyCaption inference failed: {type(e).__name__}: {e}\n\n"
                f"{self.diagnostics()}"
            ) from e
