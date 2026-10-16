"""Optional real HF-interface checks using tiny randomly initialised local models.

These tests verify adapters, not pretrained semantic accuracy. No network/weights.
"""
import json
import os

import numpy as np
from PIL import Image
import pytest

if os.environ.get("PIXELCUE_TEST_MODEL_INTERFACES") != "1":
    pytest.skip("Set PIXELCUE_TEST_MODEL_INTERFACES=1 for optional local model interface checks", allow_module_level=True)

torch = pytest.importorskip("torch")
torch.set_num_threads(1)
transformers = pytest.importorskip("transformers")

from pixelcue.visual_models import run_models


def test_real_dinov2_local_checkpoint(tmp_path):
    from transformers import BitImageProcessor, Dinov2Config, Dinov2Model
    torch.manual_seed(1)
    config = Dinov2Config(hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
        intermediate_size=64, image_size=32, patch_size=8)
    Dinov2Model(config).save_pretrained(tmp_path)
    BitImageProcessor(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32}).save_pretrained(tmp_path)
    receipts = run_models(Image.new("RGB", (32, 32), "red"), ["dinov2"], {"dinov2": str(tmp_path)})
    assert receipts[0]["status"] == "completed", receipts
    assert len(receipts[0]["result"]["embedding"]) == 32
    assert np.linalg.norm(receipts[0]["result"]["embedding"]) == pytest.approx(1)


def test_real_clip_local_checkpoint(tmp_path):
    from transformers import CLIPConfig, CLIPImageProcessor, CLIPModel, CLIPProcessor, CLIPTextConfig, CLIPTokenizer, CLIPVisionConfig
    torch.manual_seed(2)
    text = CLIPTextConfig(vocab_size=3, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, max_position_embeddings=77,
        bos_token_id=0, eos_token_id=1, pad_token_id=1)
    vision = CLIPVisionConfig(hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, image_size=32, patch_size=16)
    config = CLIPConfig(text_config=text.to_dict(), vision_config=vision.to_dict(), projection_dim=16)
    CLIPModel(config).save_pretrained(tmp_path)
    (tmp_path/"vocab.json").write_text(json.dumps({"<|startoftext|>": 0, "<|endoftext|>": 1, "a</w>": 2}))
    (tmp_path/"merges.txt").write_text("#version: 0.2\n")
    tokenizer = CLIPTokenizer(str(tmp_path/"vocab.json"), str(tmp_path/"merges.txt"), model_max_length=77)
    processor = CLIPProcessor(tokenizer=tokenizer, image_processor=CLIPImageProcessor(
        size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32}))
    processor.save_pretrained(tmp_path)
    receipts = run_models(Image.new("RGB", (32, 32), "blue"), ["clip"], {"clip": str(tmp_path)})
    assert receipts[0]["status"] == "completed", receipts
    result = receipts[0]["result"]
    assert len(result["embedding"]) == 16
    assert set(result["axes"]) == {"medium", "light", "material", "tone", "setting"}
    assert all(-1.001 <= c["cosine_similarity"] <= 1.001 for group in result["axes"].values() for c in group)


def test_real_blip_local_checkpoint(tmp_path):
    from transformers import BertTokenizer, BlipConfig, BlipForConditionalGeneration, BlipImageProcessor, BlipProcessor
    torch.manual_seed(3)
    config = BlipConfig(text_config={"vocab_size": 8, "hidden_size": 32, "intermediate_size": 64,
        "num_hidden_layers": 1, "num_attention_heads": 4, "max_position_embeddings": 128,
        "encoder_hidden_size": 32, "bos_token_id": 2, "eos_token_id": 3, "sep_token_id": 3, "pad_token_id": 0},
        vision_config={"hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1,
            "num_attention_heads": 4, "image_size": 32, "patch_size": 16})
    model = BlipForConditionalGeneration(config)
    model.save_pretrained(tmp_path)
    (tmp_path/"vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\nblue\nimage\n")
    tokenizer = BertTokenizer(str(tmp_path/"vocab.txt"))
    BlipProcessor(image_processor=BlipImageProcessor(size={"height": 32, "width": 32}), tokenizer=tokenizer).save_pretrained(tmp_path)
    receipts = run_models(Image.new("RGB", (32, 32), "blue"), ["blip"], {"blip": str(tmp_path)})
    assert receipts[0]["status"] == "completed", receipts
    assert isinstance(receipts[0]["result"]["caption"], str)
