from __future__ import annotations

import importlib.metadata
import os
import re
from pathlib import Path
from typing import Callable

from PIL import Image

# A real pre-quantized NF4 checkpoint, rather than downloading the ~17 GB
# unquantized checkpoint and quantizing it every time it is loaded.
DEFAULT_MODEL_ID = "heavlav/llama-joycaption-beta-one-hf-llava-4bit"
MODEL_ID = os.environ.get("PIXELCUE_JOYCAPTION_MODEL", DEFAULT_MODEL_ID)
MODEL_APPROX_GB = 5.96

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
    ) -> None:
        self.model_id = model_id
        self.model_dir = Path(
            os.environ.get("PIXELCUE_MODEL_DIR", str(model_dir or default_model_dir()))
        ).expanduser()
        self.status_callback = status_callback or (lambda _: None)

        self.processor = None
        self.model = None
        self.device = None
        self._downloaded_path: Path | None = None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def _status(self, message: str) -> None:
        self.status_callback(message)

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

    def ensure_downloaded(self) -> Path:
        """Explicitly materialise the model in PixelCue's own cache directory."""
        if self._downloaded_path and self._downloaded_path.exists():
            return self._downloaded_path

        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        self.model_dir.mkdir(parents=True, exist_ok=True)

        # First check without network access. This lets the GUI truthfully say
        # whether the checkpoint was already present.
        try:
            local = snapshot_download(
                repo_id=self.model_id,
                local_dir=str(self.model_dir),
                local_files_only=True,
            )
            path = Path(local)
            self._downloaded_path = path
            self._status(f"JoyCaption 4-bit: cached at {path}")
            return path
        except Exception:
            pass

        self._status(
            f"Downloading JoyCaption 4-bit model (~{MODEL_APPROX_GB:.2f} GB) to "
            f"{self.model_dir} …"
        )

        try:
            local = snapshot_download(
                repo_id=self.model_id,
                local_dir=str(self.model_dir),
                local_files_only=False,
                max_workers=4,
            )
        except Exception as e:
            raise JoyCaptionError(
                f"JoyCaption model download failed: {type(e).__name__}: {e}\n\n"
                f"Destination: {self.model_dir}\n"
                f"Repository: {self.model_id}\n\n{self.diagnostics()}"
            ) from e

        path = Path(local)
        self._downloaded_path = path
        self._status(f"JoyCaption 4-bit: downloaded to {path}")
        return path

    def load(self) -> None:
        if self.loaded:
            return

        model_path = self.ensure_downloaded()
        self._status("Loading JoyCaption 4-bit model onto GPU …")

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
            # The checkpoint contains its own BitsAndBytes NF4 quantization_config.
            # Do not pass a second BitsAndBytesConfig here.
            self.model = AutoModelForMultimodalLM.from_pretrained(
                str(model_path),
                device_map="auto",
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
            self.model.eval()
            self.device = self.model.device
            self._status(
                f"JoyCaption 4-bit: ready on {torch.cuda.get_device_name(torch.cuda.current_device())}"
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

            # This is the JoyCaption/LLaVA formatting pattern recommended by the
            # upstream model card: build a chat string, then pass image+text to
            # the processor in one call.
            convo = [
                {
                    "role": "system",
                    "content": "You are a helpful image captioner.",
                },
                {
                    "role": "user",
                    "content": TAG_PROMPT,
                },
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
                raise RuntimeError(f"Could not parse any tags from JoyCaption response: {text!r}")
            return tags

        except JoyCaptionError:
            raise
        except Exception as e:
            raise JoyCaptionError(
                f"JoyCaption inference failed: {type(e).__name__}: {e}\n\n"
                f"{self.diagnostics()}"
            ) from e
