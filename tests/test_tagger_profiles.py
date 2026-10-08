from pathlib import Path

from pixelcue.model import TAGGER_PROFILES, SmolVLMTagger, create_image_tagger

ROOT=Path(__file__).parents[1]

def test_lightweight_profiles_exist():
    for key in ["smolvlm-256m", "smolvlm-500m", "smolvlm2-256m", "smolvlm2-500m"]:
        assert key in TAGGER_PROFILES
        assert TAGGER_PROFILES[key].backend == "smolvlm"

def test_official_huggingface_repositories_are_used():
    assert TAGGER_PROFILES["smolvlm-256m"].repo_id == "HuggingFaceTB/SmolVLM-256M-Instruct"
    assert TAGGER_PROFILES["smolvlm-500m"].repo_id == "HuggingFaceTB/SmolVLM-500M-Instruct"
    assert TAGGER_PROFILES["smolvlm2-256m"].repo_id == "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
    assert TAGGER_PROFILES["smolvlm2-500m"].repo_id == "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

def test_factory_builds_smol_backend_without_loading_weights():
    tagger=create_image_tagger("smolvlm-256m")
    assert isinstance(tagger, SmolVLMTagger)
    assert tagger.model is None

def test_gui_has_model_choice():
    source=(ROOT/"src"/"pixelcue"/"gui.py").read_text(encoding="utf-8")
    assert "QComboBox" in source
    assert "available_tagger_profiles()" in source
    assert 'QLabel("Tagger:")' in source
    assert "model_profile_id=profile_id" in source
