from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_no_third_party_prequantized_checkpoint_in_model_source():
    source = (
        ROOT / "src" / "pixelcue" / "model.py"
    ).read_text(encoding="utf-8")
    assert "heavlav/llama-joycaption-beta-one-hf-llava-4bit" not in source


def test_visual_modules_are_explicitly_excluded_from_quantization():
    source = (
        ROOT / "src" / "pixelcue" / "model.py"
    ).read_text(encoding="utf-8")
    start = source.index("quantization_config = BitsAndBytesConfig(")
    end = source.index(
        "self.model = LlavaForConditionalGeneration.from_pretrained(",
        start,
    )
    block = source[start:end]
    assert '"vision_tower"' in block
    assert '"multi_modal_projector"' in block
