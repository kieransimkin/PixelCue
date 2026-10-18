# Local temporal and batch APIs

```python
from pixelcue.visual_sequence import analyse_sequences
from pixelcue.visual_batch import analyse_batch

sequence = analyse_sequences([
    {"id": "control:0", "condition_id": "control", "seconds": 0,
     "path": "frame-000.png", "sha256": "...64 lowercase hex characters..."},
    {"id": "control:1", "condition_id": "control", "seconds": 1/15,
     "path": "frame-001.png", "sha256": "...64 lowercase hex characters..."},
])
# Select representative frame paths with a consumer such as PrismMoves, then:
batch = analyse_batch(["representative.png"], models=["clip", "blip"],
                      models_dir="models", allow_download=False)
```

Optional `input_path` and `input_sha256` bind the untreated source raster. Equal
input pixel hashes across a condition identify static-source controls; changed
inputs are not assumed stationary. Source filenames/absolute paths are omitted
from temporal receipts. Models still have inspectable checkpoint metadata, which
consumers should redact before public export.

Sequence limits: 1..32,768 records, at most 512 conditions, a 64..512-pixel
analysis edge and a 0.01..0.5-second allowed gap (default 0.25). Timestamps must
be finite, nonnegative and strictly increasing within each condition. Missing
or changed captures are receipts, not silently substituted frames. Dense motion
uses forward/backward-checked LK tracks, RANSAC similarity and a checked dense
flow residual. Rotation uses image coordinates (positive clockwise). Sparse
sampling, poor tracking and uniform rasters yield unknown motion. Brightness
periodicity requires regular cadence, >=2 seconds and >=3 observed cycles;
this is not beat alignment or a flash-safety test.

The batch API reuses existing decoding/measurements and the shared transformer
adapter. Models are lazy and local-only by default; optional keyword taggers
still require explicit download permission. The default batch uses CLIP/BLIP,
with other supported models opt-in. A failure in one image/model has its own
receipt and does not erase numerical measurements. The spatial visualiser prompt
bank is explicit and hashed; CLIP similarity remains uncalibrated.

No trained checkpoints or images are downloaded by tests. Run:
`python -m pytest -q tests/test_visual_sequence.py tests/test_visual_batch.py
 tests/test_visual_analysis.py tests/test_visual_models_runtime.py`.
Install `pixelcue[sequence,dev]` for sequence tests. Public model accuracy and
real GPU/driver rendering are separate validation tasks.
