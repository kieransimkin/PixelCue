from pathlib import Path
import ast


ROOT = Path(__file__).parents[1]


def test_video_sampler_caps_frames_at_five():
    source = (ROOT / "src" / "pixelcue" / "media.py").read_text(encoding="utf-8")
    assert "max_frames: int = 5" in source
    assert "min(int(max_frames), 5)" in source


def test_video_sampler_uses_lightweight_visual_fingerprints():
    source = (ROOT / "src" / "pixelcue" / "media.py").read_text(encoding="utf-8")
    assert 'resize((16, 9))' in source
    assert "_fingerprint_distance" in source
    assert "candidate_count = min(7" in source
