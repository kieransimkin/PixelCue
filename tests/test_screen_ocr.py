"""Actual screen-text regression; explicit OCR environment required."""

import shutil
from pathlib import Path

import pytest
from PIL import Image


@pytest.mark.skipif(
    shutil.which("tesseract") is None, reason="Separate Tesseract installation required"
)
def test_canvas_label_survives_multiscale_ocr_with_original_coordinates():
    pytest.importorskip("pytesseract")
    from pixelcue.visual_models import run_models

    with Image.open(Path(__file__).parent / "fixtures/window-canvas.jpg") as image:
        receipt = run_models(image, ["ocr"], {}, ocr_profile="screen")[0]
    assert receipt["status"] == "completed"
    labels = [
        word
        for word in receipt["result"]["words"]
        if word["text"].casefold() == "launch" and word["confidence"] >= 50
    ]
    assert len(labels) == 1
    x, y, width, height = labels[0]["box_xywh"]
    assert 350 < x + width / 2 < 650 and 310 < y + height / 2 < 412
    assert 2 <= receipt["ocr_passes"] <= 3


@pytest.mark.skipif(
    shutil.which("tesseract") is None, reason="Separate Tesseract installation required"
)
def test_native_dark_control_label_is_observed_in_window_coordinates():
    pytest.importorskip("pytesseract")
    from pixelcue.visual_models import run_models

    with Image.open(Path(__file__).parent / "fixtures/native-window.png") as image:
        receipt = run_models(image, ["ocr"], {}, ocr_profile="screen")[0]
    assert receipt["status"] == "completed"
    labels = [
        w
        for w in receipt["result"]["words"]
        if w["text"].casefold() == "activate" and w["confidence"] >= 50
    ]
    assert len(labels) == 1
    x, y, width, height = labels[0]["box_xywh"]
    assert 180 <= x + width / 2 <= 460 and 125 <= y + height / 2 <= 215
