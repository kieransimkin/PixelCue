from pathlib import Path

from pixelcue.model_paths import project_models_dir
from pixelcue.model import default_model_dir


def test_project_model_root_and_override(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PIXELCUE_MODELS_DIR", raising=False)
    assert project_models_dir() == tmp_path/"models"
    assert default_model_dir() == tmp_path/"models"/"joycaption-beta-one-source"
    monkeypatch.setenv("PIXELCUE_MODELS_DIR", str(tmp_path/"composed"/"models"))
    assert project_models_dir() == tmp_path/"composed"/"models"
    assert project_models_dir(tmp_path/"explicit"/"models") == tmp_path/"explicit"/"models"
