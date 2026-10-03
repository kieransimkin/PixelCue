from pixelcue.model import DEFAULT_MODEL_ID, MODEL_APPROX_GB, default_model_dir


def test_default_model_is_explicit_4bit_checkpoint():
    assert DEFAULT_MODEL_ID == "fancyfeast/llama-joycaption-beta-one-hf-llava"
    assert MODEL_APPROX_GB >= 10.0


def test_model_cache_is_pixelcue_owned():
    path = str(default_model_dir()).casefold()
    assert "pixelcue" in path
    assert "joycaption" in path
