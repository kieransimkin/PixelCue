from pixelcue.cli import model_profile_from_argv

def test_model_option_equals_form():
    assert model_profile_from_argv(["pixelcue","--model=smolvlm-256m"]) == "smolvlm-256m"

def test_model_option_separate_form():
    assert model_profile_from_argv(["pixelcue","--model","smolvlm2-500m"]) == "smolvlm2-500m"

def test_bad_model_falls_back():
    assert model_profile_from_argv(["pixelcue","--model","not-real"]) == "joycaption"
