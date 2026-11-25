"""Small, task-specific local vision adapters shared by image and video APIs."""

from __future__ import annotations

import math

SPECIALIST_PROFILES = {
    "rtdetr": {
        "checkpoint": "PekingU/rtdetr_r18vd",
        "task": "object detection",
        "backend": "object-detection",
    },
    "mobilenet": {
        "checkpoint": "google/mobilenet_v2_1.0_224",
        "task": "ImageNet image categorisation",
        "backend": "image-classification",
    },
    "layout": {
        "checkpoint": "docling-project/docling-layout-heron",
        "task": "document layout detection",
        "backend": "object-detection",
    },
}


def detection_result(prediction, labels, size, *, threshold=0.35, maximum=50):
    """Validate model output and retain bounded boxes in the analysis raster."""
    width, height = size
    rows = []
    for score, label, box in zip(
        prediction["scores"].tolist(),
        prediction["labels"].tolist(),
        prediction["boxes"].tolist(),
        strict=True,
    ):
        if not math.isfinite(score) or not 0 <= score <= 1 or len(box) != 4:
            raise ValueError("Invalid detector score or box")
        if not all(math.isfinite(value) for value in box):
            raise ValueError("Non-finite detector box")
        if score < threshold:
            continue
        x0, y0, x1, y1 = [
            max(0.0, min(float(value), limit))
            for value, limit in zip(box, (width, height, width, height), strict=True)
        ]
        if x1 <= x0 or y1 <= y0:
            continue
        rows.append(
            {
                "label": str(labels.get(int(label), labels.get(str(label), label)))[
                    :160
                ],
                "label_id": int(label),
                "score": round(float(score), 6),
                "box_xyxy": [round(v, 3) for v in (x0, y0, x1, y1)],
                "box_normalized_xyxy": [
                    round(v / limit, 6)
                    for v, limit in zip(
                        (x0, y0, x1, y1), (width, height, width, height), strict=True
                    )
                ],
            }
        )
    rows.sort(key=lambda row: -row["score"])
    return {
        "detections": rows[:maximum],
        "total_detections": len(rows),
        "truncated": len(rows) > maximum,
        "threshold": threshold,
        "coordinate_space": "analysis_raster",
        "dimensions": [width, height],
        "caveat": "Model hypotheses; scores are not calibrated certainty. Layout boxes do not establish reading order, table cells or UI controls.",
    }


class SpecialistSession:
    """Load one detector/classifier once for a bounded batch; never hosted."""

    def __init__(self, name, checkpoint, *, allow_download, device, cache_dir, prompts):
        del prompts
        import torch
        from transformers import (
            AutoImageProcessor,
            AutoModelForImageClassification,
            AutoModelForObjectDetection,
        )

        self.torch, self.device, self.name = torch, device, name
        options = {
            "local_files_only": not allow_download,
            "trust_remote_code": False,
            "cache_dir": str(cache_dir),
        }
        self.processor = AutoImageProcessor.from_pretrained(checkpoint, **options)
        cls = (
            AutoModelForImageClassification
            if name == "mobilenet"
            else AutoModelForObjectDetection
        )
        self.model = (
            cls.from_pretrained(checkpoint, use_safetensors=True, **options)
            .to(device)
            .eval()
        )
        self.revision = getattr(self.model.config, "_commit_hash", None)
        self.model_type = getattr(self.model.config, "model_type", None)

    def infer(self, image):
        torch = self.torch
        with torch.inference_mode():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            output = self.model(**inputs)
            labels = self.model.config.id2label
            if self.name == "mobilenet":
                probabilities = output.logits[0].float().softmax(dim=-1)
                scores, indices = probabilities.topk(min(5, probabilities.shape[-1]))
                if not bool(torch.isfinite(scores).all()):
                    raise ValueError("Non-finite classifier scores")
                return {
                    "categories": [
                        {
                            "label": str(labels.get(i, labels.get(str(i), i)))[:160],
                            "label_id": i,
                            "score": round(s, 6),
                        }
                        for s, i in zip(
                            scores.cpu().tolist(), indices.cpu().tolist(), strict=True
                        )
                    ],
                    "taxonomy": "ImageNet-1k",
                    "caveat": "Whole-image closed-set category hypotheses; not a count or detection of individual objects.",
                }
            predictions = self.processor.post_process_object_detection(
                output, threshold=0.35, target_sizes=[(image.height, image.width)]
            )[0]
            return detection_result(predictions, labels, image.size)

    def close(self):
        del self.model
        if self.device == "cuda" and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
