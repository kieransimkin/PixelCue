"""Project-local model storage shared by PixelCue analysis and taggers."""
import os
from pathlib import Path


def project_models_dir(value=None):
    return Path(value or os.environ.get("PIXELCUE_MODELS_DIR") or Path.cwd()/"models").expanduser().resolve()
