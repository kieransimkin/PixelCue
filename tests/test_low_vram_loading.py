from pathlib import Path
import ast

from pixelcue.model import JoyCaption4Bit


ROOT = Path(__file__).parents[1]


def test_low_vram_helpers_are_real_methods():
    assert callable(getattr(JoyCaption4Bit, "_available_cuda_memory_bytes", None))
    assert callable(getattr(JoyCaption4Bit, "_low_vram_max_memory", None))
    assert callable(getattr(JoyCaption4Bit, "_offload_dir", None))


def test_model_loader_uses_auto_device_map_and_memory_budget():
    source = (ROOT / "src" / "pixelcue" / "model.py").read_text(encoding="utf-8")
    assert 'device_map="auto"' in source
    assert "max_memory=max_memory" in source
    assert "offload_state_dict=True" in source
    assert 'offload_folder=str(self._offload_dir())' in source
    assert "device_map=0" not in source


def test_loader_reserves_vram_headroom():
    source = (ROOT / "src" / "pixelcue" / "model.py").read_text(encoding="utf-8")
    assert "DEFAULT_GPU_MEMORY_FRACTION = 0.84" in source
    assert "reserve = 768 * 1024 * 1024" in source
    assert "torch.cuda.mem_get_info(0)" in source


def test_nf4_visual_exclusions_are_retained_with_offload():
    source = (ROOT / "src" / "pixelcue" / "model.py").read_text(encoding="utf-8")
    assert '"vision_tower"' in source
    assert '"multi_modal_projector"' in source
    assert '"model.vision_tower"' in source
    assert '"model.multi_modal_projector"' in source
