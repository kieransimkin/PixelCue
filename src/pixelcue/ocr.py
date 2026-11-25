"""Bounded Tesseract observations; optional screen-scale diversity, no UI authority."""

from __future__ import annotations


def read_words(image, *, profile="document"):
    import pytesseract

    if profile not in {"document", "screen"}:
        raise ValueError("Unknown OCR profile")
    words = []
    variants = [(image, 1.0, 1.0)]
    scaled = None
    if profile == "screen" and min(image.size) >= 256:
        # Tesseract can discard a large flat-colour button as a graphic at one
        # scale. A second bounded raster recovers some of these labels. This is
        # the same engine/input, not independent model consensus.
        from PIL import Image

        scaled = image.resize(
            (max(1, image.width // 2), max(1, image.height // 2)),
            Image.Resampling.BICUBIC,
        )
        variants.append(
            (scaled, image.width / scaled.width, image.height / scaled.height)
        )
    try:
        for raster, sx, sy in variants:
            data = pytesseract.image_to_data(
                raster, output_type=pytesseract.Output.DICT, timeout=30
            )
            for i, text in enumerate(data["text"]):
                confidence = float(data["conf"][i])
                if not text.strip() or confidence < 0:
                    continue
                x, y, width, height = [
                    int(data[key][i]) for key in ("left", "top", "width", "height")
                ]
                box = [
                    round(x * sx),
                    round(y * sy),
                    round(width * sx),
                    round(height * sy),
                ]
                word = {"text": text, "confidence": confidence, "box_xywh": box}
                duplicate = None
                for j, previous in enumerate(words):
                    px, py, pw, ph = previous["box_xywh"]
                    cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
                    if (
                        text.casefold() == previous["text"].casefold()
                        and px <= cx <= px + pw
                        and py <= cy <= py + ph
                    ):
                        duplicate = j
                        break
                if duplicate is None:
                    words.append(word)
                elif confidence > words[duplicate]["confidence"]:
                    words[duplicate] = word
        return words, len(variants)
    finally:
        if scaled is not None:
            scaled.close()


def line_regions(words, *, minimum_confidence=50):
    """Group adjacent word boxes on a shared baseline; not semantic UI controls."""
    groups = []
    for word in sorted(words, key=lambda row: row["box_xywh"][1]):
        if word["confidence"] < minimum_confidence:
            continue
        x, y, w, h = word["box_xywh"]
        selected = None
        for group in groups:
            left = min(row["box_xywh"][0] for row in group)
            right = max(row["box_xywh"][0] + row["box_xywh"][2] for row in group)
            py, ph = group[0]["box_xywh"][1], group[0]["box_xywh"][3]
            if (
                abs((y + h / 2) - (py + ph / 2)) < max(h, ph) * 0.65
                and max(left - x - w, x - right) <= max(h, ph) * 2.5
            ):
                selected = group
                break
        if selected is None:
            groups.append([word])
        else:
            selected.append(word)
    regions = []
    for group in groups:
        group.sort(key=lambda row: row["box_xywh"][0])
        left = min(row["box_xywh"][0] for row in group)
        top = min(row["box_xywh"][1] for row in group)
        right = max(row["box_xywh"][0] + row["box_xywh"][2] for row in group)
        bottom = max(row["box_xywh"][1] + row["box_xywh"][3] for row in group)
        regions.append(
            {
                "text": " ".join(row["text"] for row in group),
                "box_xywh": [left, top, right - left, bottom - top],
                "minimum_confidence": min(row["confidence"] for row in group),
            }
        )
    regions.sort(key=lambda row: (row["box_xywh"][1], row["box_xywh"][0]))
    return regions
