"""Bounded Tesseract observations; optional screen-scale diversity, no UI authority."""

from __future__ import annotations


def _screen_atlas(image):
    """At most four dark connected regions in a <=256px search raster."""
    import numpy as np
    from PIL import Image, ImageFilter, ImageOps, ImageStat

    small = image.convert("L")
    small.thumbnail((256, 256))
    threshold = small.point(lambda value: 255 if value < 135 else 0)
    eroded = threshold.filter(ImageFilter.MinFilter(5))
    mask = np.asarray(eroded).copy() > 0
    threshold.close()
    eroded.close()
    scale_x, scale_y = image.width / small.width, image.height / small.height
    small.close()
    height, width = mask.shape
    regions = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x]:
                continue
            stack = [(x, y)]
            mask[y, x] = False
            left = right = x
            top = bottom = y
            area = 0
            while stack:
                px, py = stack.pop()
                left, right = min(left, px), max(right, px)
                top, bottom = min(top, py), max(bottom, py)
                area += 1
                for nx, ny in ((px - 1, py), (px + 1, py), (px, py - 1), (px, py + 1)):
                    if 0 <= nx < width and 0 <= ny < height and mask[ny, nx]:
                        mask[ny, nx] = False
                        stack.append((nx, ny))
            if (
                area >= 40
                and right - left >= 12
                and bottom - top >= 6
                and (right - left) * (bottom - top) < width * height * 0.4
            ):
                regions.append(
                    (
                        area,
                        (
                            max(0, int((left - 2) * scale_x)),
                            max(0, int((top - 2) * scale_y)),
                            min(image.width, round((right + 3) * scale_x)),
                            min(image.height, round((bottom + 3) * scale_y)),
                        ),
                    )
                )
    tiles = []
    for _, bounds in sorted(regions, reverse=True)[:4]:
        crop = image.crop(bounds).convert("L")
        if ImageStat.Stat(crop).mean[0] < 128:
            replacement = ImageOps.invert(crop)
            crop.close()
            crop = replacement
        replacement = ImageOps.autocontrast(crop)
        crop.close()
        crop = replacement
        original = crop.size
        crop.thumbnail((768, 256))
        tiles.append((crop, bounds, original))
    if not tiles:
        return None, []
    atlas = Image.new(
        "RGB",
        (max(t[0].width for t in tiles) + 32, sum(t[0].height + 32 for t in tiles)),
        "white",
    )
    mappings, cursor = [], 16
    for crop, bounds, original in tiles:
        atlas.paste(crop, (16, cursor))
        mappings.append(
            (
                16,
                cursor,
                crop.width,
                crop.height,
                bounds[0],
                bounds[1],
                original[0] / crop.width,
                original[1] / crop.height,
            )
        )
        cursor += crop.height + 32
        crop.close()
    return atlas, mappings


def read_words(image, *, profile="document"):
    import pytesseract

    if profile not in {"document", "screen"}:
        raise ValueError("Unknown OCR profile")
    words = []
    variants = [(image, 1.0, 1.0, None)]
    scaled = atlas = None
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
            (scaled, image.width / scaled.width, image.height / scaled.height, None)
        )
        atlas, mappings = _screen_atlas(image)
        if atlas is not None:
            variants.append((atlas, 1.0, 1.0, mappings))
    try:
        for raster, sx, sy, mappings in variants:
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
                if mappings is not None:
                    matched = [
                        m
                        for m in mappings
                        if m[0] <= x + width / 2 <= m[0] + m[2]
                        and m[1] <= y + height / 2 <= m[1] + m[3]
                    ]
                    if not matched:
                        continue
                    ax, ay, _, _, ox, oy, mx, my = matched[0]
                    box = [
                        round(ox + (x - ax) * mx),
                        round(oy + (y - ay) * my),
                        round(width * mx),
                        round(height * my),
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
        if atlas is not None:
            atlas.close()


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
