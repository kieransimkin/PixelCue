from pathlib import Path

ROOT=Path(__file__).parents[1]

def test_torch_is_project_dependency_from_explicit_cuda_index():
    p=(ROOT/"pyproject.toml").read_text(encoding="utf-8")
    assert '"torch>=2.14.1"' in p
    assert 'name = "pytorch-cu132"' in p
    assert 'url = "https://download.pytorch.org/whl/cu132"' in p
    assert 'torch = { index = "pytorch-cu132" }' in p
