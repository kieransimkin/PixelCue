"""Real Transformers interfaces with tiny random local fixtures, never model downloads.

These verify loading/inference and API compatibility, not pretrained accuracy.
"""

import os

import pytest
from PIL import Image

if os.environ.get("PIXELCUE_TEST_MODEL_INTERFACES") != "1":
    pytest.skip("Enable optional local model interface checks", allow_module_level=True)

torch = pytest.importorskip("torch")
torch.set_num_threads(1)
transformers = pytest.importorskip("transformers")

from pixelcue.visual_models import run_models  # noqa: E402 - optional runtime gate above


def test_real_mobilenet_local_safetensors(tmp_path):
    from transformers import (
        MobileNetV2Config,
        MobileNetV2ForImageClassification,
        MobileNetV2ImageProcessor,
    )

    config = MobileNetV2Config(
        depth_multiplier=0.1,
        min_depth=8,
        num_labels=3,
        id2label={0: "fixture-a", 1: "fixture-b", 2: "fixture-c"},
    )
    MobileNetV2ForImageClassification(config).save_pretrained(tmp_path)
    MobileNetV2ImageProcessor(
        size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32}
    ).save_pretrained(tmp_path)
    receipts = run_models(
        Image.new("RGB", (32, 32), "red"),
        ["mobilenet"],
        {"mobilenet": str(tmp_path)},
        models_dir=tmp_path / "models",
    )
    assert receipts[0]["status"] == "completed", receipts
    categories = receipts[0]["result"]["categories"]
    assert len(categories) == 3
    assert sum(row["score"] for row in categories) == pytest.approx(1, abs=1e-5)


@pytest.mark.parametrize("profile", ["rtdetr", "layout"])
def test_real_rtdetr_local_safetensors(tmp_path, profile):
    from transformers import RTDetrImageProcessor, RTDetrResNetConfig
    from transformers import RTDetrConfig, RTDetrForObjectDetection
    from transformers import RTDetrV2Config, RTDetrV2ForObjectDetection

    backbone = RTDetrResNetConfig(
        embedding_size=8,
        hidden_sizes=[8, 16, 32, 64],
        depths=[1, 1, 1, 1],
        layer_type="basic",
        out_indices=[2, 3, 4],
    )
    options = dict(
        backbone_config=backbone.to_dict(),
        use_pretrained_backbone=False,
        use_timm_backbone=False,
        encoder_hidden_dim=32,
        encoder_layers=1,
        encoder_in_channels=[16, 32, 64],
        encoder_attention_heads=4,
        encoder_ffn_dim=64,
        d_model=32,
        decoder_in_channels=[32, 32, 32],
        decoder_layers=2,
        decoder_attention_heads=4,
        decoder_ffn_dim=64,
        num_queries=5,
        num_labels=2,
        disable_custom_kernels=True,
        id2label={0: "fixture-region", 1: "fixture-object"},
    )
    config = (RTDetrV2Config if profile == "layout" else RTDetrConfig)(**options)
    cls = (
        RTDetrV2ForObjectDetection if profile == "layout" else RTDetrForObjectDetection
    )
    cls(config).save_pretrained(tmp_path)
    RTDetrImageProcessor(size={"height": 64, "width": 64}).save_pretrained(tmp_path)
    receipts = run_models(
        Image.new("RGB", (64, 64), "blue"),
        [profile],
        {profile: str(tmp_path)},
        models_dir=tmp_path / "models",
    )
    assert receipts[0]["status"] == "completed", receipts
    assert receipts[0]["result"]["coordinate_space"] == "analysis_raster"
    assert receipts[0]["result"]["dimensions"] == [64, 64]
