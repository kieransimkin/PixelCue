from pixelcue.model import DEFAULT_MODEL_ID, MODEL_APPROX_GB, default_model_dir


def test_default_model_is_explicit_4bit_checkpoint():
    assert DEFAULT_MODEL_ID == "fancyfeast/llama-joycaption-beta-one-hf-llava"
    assert MODEL_APPROX_GB >= 10.0


def test_model_cache_is_within_owning_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PIXELCUE_MODELS_DIR", raising=False)
    assert default_model_dir() == tmp_path / "models" / "joycaption-beta-one-source"
