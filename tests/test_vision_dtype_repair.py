from pathlib import Path
import ast

from pixelcue.model import JoyCaption4Bit, DEFAULT_MODEL_ID


ROOT = Path(__file__).parents[1]


def test_uses_official_joycaption_checkpoint():
    assert DEFAULT_MODEL_ID == "fancyfeast/llama-joycaption-beta-one-hf-llava"


def test_visual_validation_is_a_real_model_method():
    assert callable(
        getattr(JoyCaption4Bit, "_validate_unquantized_visual_modules", None)
    )


def test_load_uses_nf4_and_skips_visual_modules():
    source = (
        ROOT / "src" / "pixelcue" / "model.py"
    ).read_text(encoding="utf-8")

    assert "BitsAndBytesConfig(" in source
    assert "load_in_4bit=True" in source
    assert 'bnb_4bit_quant_type="nf4"' in source
    assert "bnb_4bit_compute_dtype=torch.bfloat16" in source
    assert "bnb_4bit_use_double_quant=True" in source
    assert '"vision_tower"' in source
    assert '"multi_modal_projector"' in source
    assert '"model.vision_tower"' in source
    assert '"model.multi_modal_projector"' in source
    assert "llm_int8_skip_modules" in source


def test_load_validates_visual_modules_after_model_creation():
    source = (
        ROOT / "src" / "pixelcue" / "model.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "JoyCaption4Bit"
    )
    load = next(
        n for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "load"
    )
    segment = ast.get_source_segment(source, load)

    create_pos = segment.index(
        "LlavaForConditionalGeneration.from_pretrained("
    )
    validate_pos = segment.index(
        "self._validate_unquantized_visual_modules()"
    )
    assert create_pos < validate_pos


def test_inference_casts_pixels_to_vision_dtype_and_device():
    source = (
        ROOT / "src" / "pixelcue" / "model.py"
    ).read_text(encoding="utf-8")
    assert "vision_device, vision_dtype = self._vision_input_location()" in source
    assert 'inputs["pixel_values"].to(' in source
    assert "dtype=vision_dtype" in source
    assert "language_device = self._language_input_device()" in source
