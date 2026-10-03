from pathlib import Path
import ast

from pixelcue.model import JoyCaption4Bit


ROOT = Path(__file__).parents[1]


def test_component_resolver_is_real_model_api():
    assert callable(getattr(JoyCaption4Bit, "_resolve_component", None))
    assert callable(getattr(JoyCaption4Bit, "_vision_tower", None))
    assert callable(getattr(JoyCaption4Bit, "_multimodal_projector", None))
    assert callable(getattr(JoyCaption4Bit, "_language_model", None))


def test_resolver_supports_direct_and_nested_layouts():
    source = (ROOT / "src" / "pixelcue" / "model.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "JoyCaption4Bit")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_resolve_component")
    segment = ast.get_source_segment(source, fn)

    assert 'getattr(self.model, name, None)' in segment
    assert 'getattr(self.model, "model", None)' in segment
    assert "getattr(inner, name, None)" in segment


def test_quantization_skip_list_covers_transformers_5_names():
    source = (ROOT / "src" / "pixelcue" / "model.py").read_text(encoding="utf-8")
    assert '"model.vision_tower"' in source
    assert '"model.multi_modal_projector"' in source


def test_visual_and_language_device_helpers_use_resolvers():
    source = (ROOT / "src" / "pixelcue" / "model.py").read_text(encoding="utf-8")
    assert "vision = self._vision_tower()" in source
    assert "language_model = self._language_model()" in source
    assert "self._multimodal_projector().parameters()" in source
