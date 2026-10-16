import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from pixelcue.visual_analysis import analyse_image, load_image, measure_image
from pixelcue.visual_service import VisualAnalysisService, compare_profiles


def saved(tmp_path, image, name="image.png"):
    path = tmp_path/name
    image.save(path)
    return path


@pytest.mark.parametrize("color,expected", [("black", 0), ("white", 1), ("red", .2126)])
def test_linear_luminance_known_colors(tmp_path, color, expected):
    result = analyse_image(saved(tmp_path, Image.new("RGB", (32, 32), color)))
    measurements = result["measurements"]
    assert measurements["colour"]["relative_luminance_percentiles"][2] == pytest.approx(expected)
    assert measurements["composition"]["edge_density_above_0_08"] == 0
    assert measurements["texture"]["grayscale_entropy_bits"] == 0
    assert measurements["composition"]["spectral_residual_centroid_xy"] is None
    assert len(measurements["colour"]["palette"]) == 1
    json.dumps(result, allow_nan=False)


def test_alpha_hidden_rgb_is_not_visible_evidence(tmp_path):
    path = saved(tmp_path, Image.new("RGBA", (32, 32), (255, 0, 0, 0)))
    p = analyse_image(path)
    assert p["measurements"]["colour"]["palette"][0]["hex"] == "#ffffff"
    assert p["source"]["alpha"]["fully_transparent_fraction"] == 1
    dark = analyse_image(path, background="#000000")
    assert dark["measurements"]["colour"]["palette"][0]["hex"] == "#000000"


def test_edges_entropy_frequency_and_determinism(tmp_path):
    rng = np.random.default_rng(42)
    path = saved(tmp_path, Image.fromarray(rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)))
    a, b = analyse_image(path), analyse_image(path)
    assert a == b
    assert a["measurements"]["texture"]["grayscale_entropy_bits"] > 7
    assert sum(a["measurements"]["frequency"]["radial_power_fractions_low_mid_high"]) == pytest.approx(1)
    assert sum(x["sample_fraction"] for x in a["measurements"]["colour"]["palette"]) == pytest.approx(1)
    assert a["source"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_resize_and_exif_orientation(tmp_path):
    image = Image.new("RGB", (120, 60), "green")
    exif = image.getexif()
    exif[274] = 6
    path = tmp_path/"rotated.jpg"
    image.save(path, exif=exif)
    raster, source = load_image(path, max_edge=64)
    assert source["dimensions"] == [60, 120]
    assert raster.size == (32, 64)


@pytest.mark.parametrize("size", [(1, 1), (1, 8), (8, 1), (2, 2)])
def test_tiny_images_finite(tmp_path, size):
    result = analyse_image(saved(tmp_path, Image.new("RGB", size)))
    json.dumps(result, allow_nan=False)


def test_animated_first_frame_and_corrupt_file(tmp_path):
    path = tmp_path/"animation.gif"
    Image.new("RGB", (16, 16), "red").save(path, save_all=True,
        append_images=[Image.new("RGB", (16, 16), "blue")])
    p = analyse_image(path)
    assert p["source"]["frames"] == 2
    assert "Only frame 0" in p["source"]["notes"][0]
    (tmp_path/"broken.png").write_text("not an image")
    with pytest.raises(OSError):
        analyse_image(tmp_path/"broken.png")


def test_bounds_before_processing(tmp_path):
    path = saved(tmp_path, Image.new("RGB", (20, 20)))
    with pytest.raises(ValueError):
        load_image(path, max_pixels=100)
    with pytest.raises(ValueError):
        load_image(path, max_edge=4096)
    with pytest.raises(ValueError, match="Unknown"):
        analyse_image(path, models=["made-up"])


def test_failed_model_is_explicit_and_measurements_survive(tmp_path, monkeypatch):
    from pixelcue import visual_models
    def unavailable(*args, **kwargs):
        raise OSError("weights not cached")
    monkeypatch.setattr(visual_models, "_transformer", unavailable)
    p = analyse_image(saved(tmp_path, Image.new("RGB", (16, 16))), models=["clip", "dinov2", "blip"])
    assert len(p["models"]) == 3
    assert all(m["status"] == "unavailable" for m in p["models"])
    assert all(m["allow_download"] is False for m in p["models"])


def test_keyword_service_delegation_and_download_guard(tmp_path):
    class Keywords:
        def extract_keywords(self, path, *, model):
            assert model == "smolvlm-256m"
            return ["synthetic blue geometry"]
    path = saved(tmp_path, Image.new("RGB", (32, 32), "blue"))
    service = VisualAnalysisService(Keywords())
    guarded = service.analyse(path, keyword_models=["smolvlm-256m"])
    assert guarded["models"][0]["status"] == "unavailable"
    result = service.analyse(path, keyword_models=["smolvlm-256m"], allow_download=True)
    assert result["models"][0]["result"]["tags"] == ["synthetic blue geometry"]


def test_collection_distances_do_not_invent_identity(tmp_path):
    a = analyse_image(saved(tmp_path, Image.new("RGB", (32, 32), "red")))
    b = analyse_image(saved(tmp_path, Image.new("RGB", (32, 32), "blue"), "blue.png"))
    pairs = compare_profiles([a, a, b])["pairs"]
    assert pairs[0]["exact_file_duplicate"]
    assert pairs[0]["hue_js_divergence_bits"] == 0
    assert pairs[1]["hue_js_divergence_bits"] == pytest.approx(1)
    assert "brand_score" not in pairs[0]
