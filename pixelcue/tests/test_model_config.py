from pixelcue.model import DEFAULT_MODEL_ID, MODEL_APPROX_GB, default_model_dir


def test_default_model_is_explicit_4bit_checkpoint():
    assert "4bit" in DEFAULT_MODEL_ID.casefold()
    assert MODEL_APPROX_GB < 8.0


def test_model_cache_is_pixelcue_owned():
    path = str(default_model_dir()).casefold()
    assert "pixelcue" in path
    assert "joycaption" in path
