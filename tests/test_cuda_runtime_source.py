from pathlib import Path
from pixelcue.model import JoyCaption4Bit, cuda_pytorch_install_command


ROOT = Path(__file__).parents[1]


def test_runtime_preflight_is_a_real_class_method():
    assert callable(getattr(JoyCaption4Bit, "validate_runtime", None))
    assert "validate_runtime" in JoyCaption4Bit.__dict__


def test_loader_uses_llava_class_not_missing_multimodal_auto_class():
    source = (ROOT / "src" / "pixelcue" / "model.py").read_text(encoding="utf-8")
    assert "LlavaForConditionalGeneration" in source
    assert "AutoModelForMultimodalLM" not in source


def test_cuda_install_command_targets_active_interpreter_and_cuda_index():
    command = cuda_pytorch_install_command()
    assert "pip install" in command
    assert "download.pytorch.org/whl/cu132" in command
    assert "torch" in command


def test_normal_package_dependencies_do_not_install_torch_cpu_wheel():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"torch>=2.4"' not in pyproject


def test_preflight_happens_before_snapshot_download():
    source = (ROOT / "src" / "pixelcue" / "model.py").read_text(encoding="utf-8")
    start = source.index("def ensure_downloaded(self)")
    section = source[start:source.index("def load(self)", start)]
    assert section.index("self.validate_runtime()") < section.index("snapshot_download")
