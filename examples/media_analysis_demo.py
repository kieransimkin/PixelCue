"""Create an original synthetic page/video and retain actual PixelCue OCR evidence.

Requires PixelCue's analysis extra, Tesseract on PATH, and PyAV. No downloads.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pixelcue.media_analysis import analyse_media


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    image = Image.new("RGB", (960, 640), "#f6f3eb")
    draw = ImageDraw.Draw(image)
    title = ImageFont.load_default(size=46)
    body = ImageFont.load_default(size=27)
    draw.rectangle((32, 30, 928, 142), fill="#14364b")
    draw.text((58, 59), "PIXELCUE TOOL DEMO", font=title, fill="white")
    draw.text((58, 184), "Local image and video analysis", font=body, fill="#14364b")
    draw.text((58, 244), "Objects  Categories  Layout  OCR", font=body, fill="#14364b")
    draw.text(
        (58, 310), "This is an original synthetic fixture.", font=body, fill="#14364b"
    )
    draw.text((58, 370), "Recognised words are source data.", font=body, fill="#14364b")
    draw.text((58, 430), "They never grant tool authority.", font=body, fill="#14364b")
    draw.rounded_rectangle((58, 514, 900, 594), radius=12, fill="#d6e5dd")
    draw.text(
        (80, 540),
        "Hash-bound input. Explicit model receipts.",
        font=body,
        fill="#14364b",
    )
    source = args.out / "media-analysis-page.png"
    image.save(source)
    result = analyse_media(source, models=["ocr"], max_edge=960)
    (args.out / "media-analysis-page.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    receipt = result["items"][0]["analysis"]["models"][0]
    if receipt["status"] != "completed":
        raise SystemExit(
            "OCR unavailable; inspect the retained receipt before claiming a demo"
        )
    words = receipt["result"]["words"]
    for word in words:
        x, y, width, height = word["box_xywh"]
        draw.rectangle((x, y, x + width, y + height), outline="#d43c46", width=2)
    image.save(args.out / "media-analysis-ocr.png")

    import av

    video = args.out / "media-analysis-fixture.mp4"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("libx264", rate=5)
        stream.width, stream.height, stream.pix_fmt = 480, 320, "yuv420p"
        for index in range(15):
            frame = image.resize((480, 320))
            overlay = ImageDraw.Draw(frame)
            overlay.rectangle(
                (8 + index * 25, 300, 28 + index * 25, 316), fill="#d43c46"
            )
            for packet in stream.encode(av.VideoFrame.from_image(frame)):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    video_result = analyse_media(video, models=["ocr"], max_frames=3, max_edge=480)
    (args.out / "media-analysis-video.json").write_text(
        json.dumps(video_result, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "image_status": result["status"],
                "ocr_words": len(words),
                "tesseract": receipt["resolved_revision"],
                "video_status": video_result["status"],
                "video_frames": video_result["coverage"]["sampled_frames"],
            }
        )
    )


if __name__ == "__main__":
    main()
