from __future__ import annotations

import re
from typing import Iterable

from PIL import Image

MODEL_ID = "fancyfeast/llama-joycaption-beta-one-hf-llava"

TAG_PROMPT = """Write a list of Booru-like tags for this image.
Output ONLY comma-separated tags, with no prose.
Include concrete subject, object, setting, action, clothing, visual-style, and composition tags where useful.
You MUST classify the image as SFW, suggestive, or NSFW.
If the image is NSFW, include the exact tag NSFW.
If any person's face is visibly present, include the exact tag FaceIdentity.
Do not identify who a person is. FaceIdentity only means that a visible human face is present.
Keep the tag list concise and avoid duplicate tags."""


def parse_tags(text: str) -> list[str]:
    """Parse JoyCaption's tag-like response defensively."""
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
        elif low == "faceidentity" or low == "face_identity":
            tag = "FaceIdentity"
        elif low in {"sfw", "rating:sfw", "rating: sfw"}:
            tag = "SFW"
        elif low in {"suggestive", "rating:suggestive", "rating: suggestive"}:
            tag = "suggestive"

        key = tag.casefold()
        if key not in seen:
            seen.add(key)
            out.append(tag)

    # Ensure exact spelling if those special labels appeared anywhere in the raw response.
    if re.search(r"\bnsfw\b", text, re.I) and "nsfw" not in seen:
        out.append("NSFW")
    if re.search(r"\bface[_ ]?identity\b", text, re.I) and "faceidentity" not in seen:
        out.append("FaceIdentity")
    return out


class JoyCaption4Bit:
    """Lazy-loading JoyCaption Beta One inference wrapper."""

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.model_id = model_id
        self.processor = None
        self.model = None
        self.device = None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if self.loaded:
            return

        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration

        if not torch.cuda.is_available():
            raise RuntimeError(
                "A CUDA GPU was not detected. This app is configured to load JoyCaption in "
                "bitsandbytes 4-bit NF4 mode. Install a CUDA-enabled PyTorch build and a "
                "compatible NVIDIA driver/GPU."
            )

        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            quantization_config=quant,
            device_map="auto",
            torch_dtype=compute_dtype,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.compute_dtype = compute_dtype

    def tags_for_image(self, image: Image.Image) -> list[str]:
        self.load()
        import torch

        assert self.processor is not None
        assert self.model is not None
        image = image.convert("RGB")

        convo = [
            {"role": "system", "content": "You are a precise image tagger."},
            {"role": "user", "content": TAG_PROMPT},
        ]
        prompt = self.processor.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[prompt],
            images=[image],
            return_tensors="pt",
        )

        # Quantized models can have a mixed device map; input embeddings normally begin
        # on the device of the first model parameters.
        moved = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                value = value.to(self.device)
                if key == "pixel_values":
                    value = value.to(self.compute_dtype)
            moved[key] = value

        with torch.inference_mode():
            generated = self.model.generate(
                **moved,
                max_new_tokens=256,
                do_sample=False,
                use_cache=True,
            )[0]

        input_len = moved["input_ids"].shape[1]
        response_ids = generated[input_len:]
        text = self.processor.tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        return parse_tags(text)
