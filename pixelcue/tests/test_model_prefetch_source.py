from pathlib import Path


def test_scanner_starts_model_prefetch_before_traversal():
    source = (Path(__file__).parents[1] / "src" / "pixelcue" / "scanner.py").read_text(encoding="utf-8")
    run_pos = source.index("def run(self)")
    prefetch_pos = source.index("self._start_model_prefetch()", run_pos)
    loop_pos = source.index("while True:", run_pos)
    assert prefetch_pos < loop_pos


def test_transformers_has_no_forced_v4_upper_bound():
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    assert '"transformers>=4.53"' in pyproject
    assert 'transformers>=4.53,<5' not in pyproject
