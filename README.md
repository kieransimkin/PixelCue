# PixelCue

PixelCue is a local visual-media analysis service and desktop Python tool. It tags images and sampled video/archive content with a selectable local vision-language model (VLM). The desktop app scans a filesystem using an explicit FIFO **todo list**, builds a live tag cloud, and exports a TSV containing every file discovered.

## Behaviour

- The selected starting directory is the first item in a `deque`.
- Items are always taken from the **oldest/front** of the queue.
- Directories are enumerated with `Path.iterdir()` (so hidden entries are included) and every child is appended to the **bottom/back** of the queue.
- Images are tagged with the selected VLM. JoyCaption remains the high-quality default, with much smaller SmolVLM alternatives available.
- `FaceIdentity` images are queued independently for DeepFace attributes, embeddings and identity clustering when the `faces` extra is installed.
- Videos are sampled in a separate worker: PixelCue considers a small set of timestamps and selects at most **five visually diverse frames**; video VLM work is lower priority than normal images.
- Archives are treated as container files. Up to five randomly selected media members are sampled and their tags are unioned onto the archive itself.
- Symlinks are deferred and the GUI asks whether to follow each link without blocking the worker thread. Following directory links is cycle-protected by device/inode identity.
- When scanning finishes, the GUI automatically opens a TSV save dialog.

The TSV columns start with exactly:

1. `filename`
2. `tags`
3. `exif`

and then include path, extension, libmagic MIME/type description, size, timestamps, permissions, owner IDs, symlink target, inferred category, media/archive metadata, and any processing error.

## Image tagging model choice

PixelCue no longer requires the heavyweight JoyCaption model for every scan. The **Tagger** dropdown in the main window selects the VLM before a scan starts. The same choice is available on the command line with `--model`.

Available built-in profiles:

| Model | Profile ID | Size class | Intended use |
| --- | --- | --- | --- |
| JoyCaption Beta One | `joycaption` | ~8B-class source model, NF4 runtime | Best detailed tagging, heaviest option |
| SmolVLM 256M Instruct | `smolvlm-256m` | 256M | Tiny / fastest still-image tagging |
| SmolVLM 500M Instruct | `smolvlm-500m` | 500M | Lightweight quality/speed balance |
| SmolVLM2 256M Video Instruct | `smolvlm2-256m` | 256M | Newer tiny multimodal model used for image tagging |
| SmolVLM2 500M Video Instruct | `smolvlm2-500m` | 500M | Newer lightweight multimodal model |

All profiles receive the same PixelCue keyword-tagging prompt, including the required `NSFW` and `FaceIdentity` labels. The small SmolVLM models may produce less detailed or less consistent tags than JoyCaption, but they download much faster, use far less VRAM/RAM, and are useful for very large collections or quick first-pass indexing.

GUI: choose the model from **Tagger:** before starting the scan.

CLI examples:

```powershell
uv run pixelcue --model smolvlm-256m I:\
uv run pixelcue --model smolvlm-500m I:\
uv run pixelcue --model smolvlm2-500m I:\
uv run pixelcue --model joycaption I:\
```

You can also set the default profile with `PIXELCUE_TAGGER_MODEL`. Each model is cached in its own PixelCue model directory, so changing model does not overwrite another model's cache.

## JoyCaption / GPU

The app loads:

`fancyfeast/llama-joycaption-beta-one-hf-llava`

from the official JoyCaption checkpoint, applying BitsAndBytes NF4 at runtime only to the language-model components.

JoyCaption strongly benefits from an NVIDIA CUDA GPU and uses PixelCue's NF4/CPU-offload path. The SmolVLM profiles are dramatically smaller and can also fall back to CPU, although CUDA remains faster. The selected model is downloaded from Hugging Face on first use and cached locally.

The prompt asks JoyCaption for comma-separated Booru-like tags and explicitly requests:

- `NSFW` for NSFW content.
- `FaceIdentity` whenever a person's visible face is present.
- `SFW` / `suggestive` classification where appropriate.

JoyCaption is a generative VLM, so these labels are model judgments rather than guarantees.

## Installation (uv)

PixelCue is managed with **uv** and pins **Python 3.13** in `.python-version`. You do not need to create or activate a virtual environment manually.

### Prerequisites

- An NVIDIA CUDA-capable GPU is strongly recommended. It is required for PixelCue's JoyCaption NF4 path; the small SmolVLM profiles can run on CPU.
- Install a current NVIDIA driver.
- Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
- On Linux, install libmagic before syncing PixelCue:

```bash
sudo apt install libmagic1
```

RAR archives may additionally require `unrar`, `unar`, or `bsdtar`.

### Full install (recommended)

From the PixelCue checkout:

```powershell
uv python install 3.13
uv sync --extra faces
```

That command creates/synchronizes `.venv` and installs:

- PixelCue and its GUI/media dependencies;
- CUDA-enabled PyTorch from the **PyTorch CUDA 13.2** wheel index configured in `pyproject.toml`;
- local VLM runtime dependencies (JoyCaption plus the lightweight SmolVLM-compatible Transformers backend);
- the stable DeepFace stack: TensorFlow 2.21.0 + tf-keras 2.21.0.

No `pip install torch ...`, manual venv activation, or separate DeepFace install is required.

Run PixelCue directly through uv:

```powershell
uv run pixelcue I:\
```

Or open the GUI without a starting path:

```powershell
uv run pixelcue
```

## REST and Socket.IO service

Start the local service on its safe loopback default:

```powershell
uv run pixelcue-server
```

The service exposes:

- `GET /health` for version and available-model discovery;
- `POST /v1/keywords` for one server-local image, video or archive file;
- a Socket.IO endpoint at `/socket.io` with `keyword_progress` events.

The keyword route accepts a JSON body and deliberately returns a bare JSON list
of strings:

```json
{
  "path": "C:\\Media\\cover.png",
  "model": "smolvlm-256m",
  "request_id": "cover-2026-10-03"
}
```

```json
["night sky", "silhouette", "blue lighting"]
```

To receive progress, connect with a Socket.IO 4.x client, subscribe before the
HTTP request using the same caller-generated request ID, then listen for
`keyword_progress`:

```javascript
const socket = io("http://127.0.0.1:8765");
socket.emit("subscribe", { request_id: "cover-2026-10-03" });
socket.on("keyword_progress", (event) => console.log(event));
```

Model work is serialised and loaded taggers are reused between requests. The
service accepts paths on the machine running PixelCue; it does not treat a path
as a browser upload. Keep the default `127.0.0.1` binding unless the service is
placed behind appropriate authentication and filesystem-access controls.

The first Hugging Face model download can take hours, particularly for
JoyCaption. The REST request intentionally remains open while the Socket.IO
channel reports cache checks, downloaded bytes, model loading and media-analysis
stages. A slow but advancing download is not treated as a service failure.

For a separately served local React development client, set a comma-separated
origin allow-list before starting the server, for example:

```powershell
$env:PIXELCUE_CORS_ORIGINS = 'http://127.0.0.1:5173,http://localhost:5173'
uv run pixelcue-server
```

### Install without DeepFace

If you do not want face analysis:

```powershell
uv python install 3.13
uv sync
uv run pixelcue I:\
```

The selected VLM and the normal filesystem/media scan still work; `FaceIdentity` items simply will not receive DeepFace analysis.

### Lockfile

`uv sync` automatically resolves the project and creates/updates `uv.lock` when necessary. **Commit `uv.lock`** so future installs use exactly the same dependency graph. The lockfile is intentionally not ignored by `.gitignore`.

For CI or a checked-in lockfile, use:

```powershell
uv sync --frozen --extra faces
```

### Development environment

```powershell
uv sync --extra faces --group dev
uv run pytest
```

If dependencies change, refresh the lock explicitly with:

```powershell
uv lock
```

## Model download and diagnostics

PixelCue downloads only the model profile selected for the scan. SmolVLM models are stored in profile-specific cache directories. For the `joycaption` profile, PixelCue explicitly downloads the official JoyCaption Beta One source checkpoint into its own user cache on first use, then applies NF4 4-bit quantization to the language-model components at load time. The model cache check/download starts immediately when a scan starts and runs in parallel with filesystem traversal. PixelCue queries Hugging Face for the exact checkpoint file sizes and displays the exact bytes downloaded / total, human-readable GiB values, and percentage while downloading. The official source checkpoint is approximately **17 GB**.

The GUI shows model state separately from the current filesystem item:

- `Downloading JoyCaption 4-bit model ...`
- `JoyCaption 4-bit: downloaded`
- `Loading JoyCaption 4-bit model onto GPU ...`
- `JoyCaption 4-bit: ready ...`

On Windows the cache location is under the normal per-user application cache returned by `platformdirs`; PixelCue displays the exact path while downloading. You can override it with:

```bash
set PIXELCUE_MODEL_DIR=D:\Models\PixelCue
```

You can set the default profile with `PIXELCUE_TAGGER_MODEL`. `PIXELCUE_JOYCAPTION_MODEL` remains available for overriding the JoyCaption repository specifically.

A model or inference error is no longer silent. The first one appears immediately in the GUI and every error is retained in the **Errors** window with PyTorch/CUDA/Transformers/BitsAndBytes diagnostics. After a backend-level JoyCaption failure, PixelCue continues the filesystem scan but does not repeatedly retry the broken backend for every image.


### 0.4.0 selectable lightweight VLM taggers

PixelCue now exposes multiple local image-tagging VLM profiles in the GUI and CLI. In addition to JoyCaption, the built-in choices include official Hugging Face SmolVLM 256M/500M and SmolVLM2 256M/500M checkpoints. These lightweight models use the same comma-separated keyword prompt and PixelCue special tags but require only a fraction of JoyCaption's model size and memory.

Use the GUI **Tagger** dropdown or `--model <profile-id>`.

## Version notes (historical)

The entries below document earlier releases and may contain commands or dependency advice that has since been superseded. For installation, always use the **Installation (uv)** section above.

### Metadata lookup cannot block downloads

From 0.1.4 onward, PixelCue starts the Hugging Face model transfer immediately and
queries exact repository sizes concurrently with a 10-second request timeout. The
standard JoyCaption checkpoint therefore shows progress against an approximately
5.96 GB denominator immediately; when exact file metadata arrives, the GUI switches
to the exact byte total. A slow `files_metadata=True` API response can no longer
prevent the model download from starting.


### 0.1.5 model worker fix

0.1.5 fixes a packaging regression in 0.1.4 where `ensure_downloaded()` was
accidentally placed outside `JoyCaption4Bit`. PixelCue now has regression tests
that import the class itself and verify `ensure_downloaded()`, `load()`, and
`tags_for_image()` are actual class methods before a package is built.


### 0.1.6 true parallel filesystem discovery

Filesystem traversal and media analysis now use separate queues and workers.

The filesystem QThread only discovers entries, records metadata, classifies files,
and appends images/videos/archives to a FIFO media-analysis queue. It never waits
for JoyCaption. A separate media worker consumes those jobs and waits for the model
when necessary. Therefore a first image/archive encountered during the scan can no
longer stop traversal while the model is downloading.

The GUI reports `files discovered` separately from `Media: N queued / M processed`.
The final TSV dialog still waits until queued media analysis finishes, so exported
tags are complete.


### 0.1.7 CUDA/runtime compatibility fix

PixelCue no longer declares `torch` as a normal PyPI dependency. On Windows,
installing plain `torch` from the default package index can result in a CPU-only
wheel even when an NVIDIA GPU is present.

Before downloading/loading JoyCaption, PixelCue now checks:

- whether PyTorch is installed;
- whether `torch.version.cuda` is present;
- whether `torch.cuda.is_available()` is true;
- what `nvidia-smi` reports for the physical GPU/driver;
- whether bitsandbytes imports successfully.

If Windows sees an NVIDIA GPU but PyTorch is CPU-only, PixelCue gives an exact
command using the active Python interpreter, for example:

```powershell
python -m pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu132
```

JoyCaption is loaded explicitly with `LlavaForConditionalGeneration`, matching
the architecture declared by the checkpoint and supporting Transformers 4.x as
well as newer releases where the generic multimodal auto-class may differ.


### 0.1.8 JoyCaption 4-bit SigLIP dtype repair

The original JoyCaption project documents a Transformers/SigLIP limitation:
the vision tower and multimodal projector should not normally be quantized with
bitsandbytes. Some third-party pre-quantized checkpoints nevertheless contain an
integer-packed SigLIP attention output projection, producing runtime errors such
as:

`self and mat2 must have the same dtype, but got Half and Byte`

PixelCue 0.1.8 keeps the already-downloaded compact 4-bit checkpoint and applies
the known compatibility repair only when that bad integer projection dtype is
detected. Pixel values are also explicitly sent to the vision tower using its
own device and floating dtype, while token inputs are sent to the language-model
device.

This avoids throwing away the existing ~6 GB cache or downloading the ~17 GB
full-precision source model merely to quantize the language model again.


### 0.1.9 media queue ordering by size

When PixelCue compiles the list of media items for JoyCaption analysis, it now
prioritises them by `size_bytes`, processing the largest files first. This
applies to images, videos, and archives placed onto the media-analysis queue.
Files with the same size retain their discovery order. Filesystem discovery is
still independent of JoyCaption and continues in parallel.


### 0.1.10 continue after per-file JoyCaption failures

If JoyCaption fails on an individual image, video, or archive, PixelCue now
records the error on that file and immediately continues with the next queued
media item. A single bad file no longer disables tagging for the rest of the
scan. Only a true model-setup failure (for example the model cannot load at all)
prevents later items from being tagged.


### 0.1.11 libmagic-gated media queue

PixelCue now performs content-based MIME identification **before** a filesystem
file can enter the JoyCaption media queue. A `.jpg`, `.mp4`, `.zip`, etc. suffix
by itself is no longer sufficient when libmagic/the system `file` database has
identified the contents as something else. This prevents files such as Windows
Recycle Bin `$I...JPG` metadata records from being sent to Pillow/JoyCaption.

The original extension is still recorded in the TSV, but queue classification
uses the detected MIME type. The TSV's `extra_metadata` also records whether the
type came from `libmagic`, the `file` command, or was unavailable.


### 0.1.11 content-based queue admission

PixelCue now treats names and extensions as metadata only. Before a filesystem
file is admitted to the JoyCaption queue, it must be identified by libmagic
(the same database used by the `file` utility) as `image/*`, `video/*`, or a
supported archive MIME type. A file named `something.JPG` that actually contains
Windows Recycle Bin metadata or arbitrary binary data will therefore never be
sent to JoyCaption.

The same principle applies inside archives: member bytes are probed with
libmagic before they count as image/video candidates. Archive type itself is
selected from MIME when available rather than from `.zip`, `.rar`, etc.

The repository now also contains a `.gitignore` with `*.tsv`, so generated scan
inventories are never accidentally committed.


### 0.1.12 official JoyCaption NF4 loading path

PixelCue no longer uses the third-party whole-model pre-quantized JoyCaption
checkpoint. That checkpoint quantizes parts of the SigLIP vision tower and can
fail at inference with errors such as `Half and Byte` or `BFloat16 and Byte`.

PixelCue now follows JoyCaption's own NF4 loading strategy:

- download the official `fancyfeast/llama-joycaption-beta-one-hf-llava` checkpoint;
- apply BitsAndBytes NF4 during model loading;
- use `bnb_4bit_compute_dtype=torch.bfloat16`;
- use double quantization;
- explicitly exclude `vision_tower` and `multi_modal_projector` from
  bitsandbytes quantization;
- validate after loading that the visual path does not contain integer-packed
  parameters;
- send `pixel_values` to the vision tower using its actual floating dtype/device,
  and language inputs to the language-model input device.

This requires a larger one-time source-checkpoint download, but the in-memory
language model is still 4-bit and is compatible with an 8 GB-class GPU. The old
third-party ~6 GB cache is not used by 0.1.12 and may be deleted manually if disk
space is needed.


### 0.1.13 minimum JoyCaption file size

After content-based MIME identification, PixelCue now excludes files smaller
than **500,000 bytes** from the JoyCaption analysis queue. These files are still
catalogued normally and remain in the final TSV with their MIME, category and
other metadata. Their `extra_metadata` records that JoyCaption was skipped due
to the size threshold.

Files of exactly 500,000 bytes or larger remain eligible. The existing
largest-file-first priority then applies to the eligible media queue.


### 0.1.14 Transformers 5.x LLaVA component layout

Transformers 5.x wraps the actual `LlavaModel` inside
`LlavaForConditionalGeneration.model`. PixelCue now resolves the vision tower,
multimodal projector, and language model across both layouts:

- older: `model.vision_tower`
- current: `model.model.vision_tower`

The NF4 exclusion list also contains both old and current qualified module names,
so the visual path remains floating-point regardless of the installed supported
Transformers layout.


### 0.1.15 low-VRAM loading and CPU offload

On 8 GiB GPUs, forcing the entire JoyCaption model onto CUDA can run out of VRAM
while Transformers is still constructing and quantizing the checkpoint.
PixelCue now uses Accelerate's automatic placement with an explicit memory
budget.

By default it reserves roughly 16% of physical VRAM (and at least 768 MiB of
currently free VRAM) for CUDA context, quantization buffers, image activations,
and generation. Model layers that do not fit inside the remaining budget are
offloaded to system RAM. `offload_state_dict=True` also reduces peak host-memory
pressure during loading.

The default CPU placement budget is 24 GiB and can be changed with
`PIXELCUE_CPU_OFFLOAD_GIB`. PixelCue reports the GPU/CPU placement budget while
loading and reports the number of offloaded modules once ready.


### 0.2.0 DeepFace identity pipeline and diverse video sampling

Images tagged `FaceIdentity` by JoyCaption are now appended to a separate
DeepFace queue. This queue runs in its own worker thread independently of
filesystem discovery and JoyCaption media analysis.

For each qualifying image PixelCue selects the largest detected face as the
image's primary identity and stores:

- apparent age;
- gender scores and dominant gender;
- DeepFace race/ethnicity scores and dominant label;
- emotion scores and dominant emotion;
- face confidence/region;
- an ArcFace embedding vector;
- an optional celebrity look-alike.

The demographic fields are model estimates, not ground-truth personal
attributes.

#### Face clustering

Approximately every 10 seconds PixelCue reclusters all completed primary-face
embeddings using cosine distance. Every analysed image belongs to a cluster,
including singleton clusters. The default automatic merge distance is 0.35 and
can be changed with `PIXELCUE_FACE_CLUSTER_DISTANCE`.

Each cluster's representative thumbnail is the actual image whose embedding is
nearest the cluster centroid. The GUI has a dedicated **Face identity clusters**
pane; clicking a cluster opens all images currently assigned to it.

DeepFace attributes and the complete embedding vector are also stored in the
file's TSV `extra_metadata`.

#### Celebrity look-alike database

Celebrity matching requires reference faces; DeepFace does not provide PixelCue
with a universal celebrity identity database automatically. Set:

```text
PIXELCUE_CELEBRITY_DB=C:\path\to\celebrity-faces
```

to a DeepFace-compatible local reference database. With no database configured,
PixelCue records the look-alike field as unavailable rather than inventing a
name.

DeepFace/TensorFlow is kept on CPU by default so it does not compete with
JoyCaption for the RTX GPU's VRAM.

#### Video frames

Video analysis now uses at most **five frames per video**. PixelCue inspects at
most nine timestamps spread across the duration, reduces each candidate to a
16x9 grayscale fingerprint, and greedily selects frames that are furthest from
the already selected frames. This gives broad visual coverage with negligible
comparison cost relative to JoyCaption inference.


### 0.2.2 faster non-blocking video handling

Video sampling now opens each video container only once and performs at most seven sparse seeks to choose at most five visually diverse frames. This removes the repeated container-open overhead from 0.2.1.

Sampled videos also have their own low-priority JoyCaption heap. Images and archives always take precedence, and PixelCue does not begin JoyCaption inference on video frames until filesystem discovery is complete. Large videos therefore cannot jump ahead of normal images merely because the global media policy otherwise favours larger files. Video sampling itself remains in its dedicated background thread.


### 0.2.3 DeepFace dependency preflight

Fresh PixelCue installs include DeepFace. For an existing editable checkout,
adding DeepFace to `pyproject.toml` does not retroactively install it until the
package is reinstalled. PixelCue now detects that situation before processing
face jobs, reports the exact active-interpreter command:

```text
uv sync --extra faces
```

and disables only the DeepFace subsystem for that scan. It reports the missing
dependency once rather than generating one error for every `FaceIdentity`
image; filesystem discovery and JoyCaption continue normally.


### 0.2.4 TensorFlow/Keras DeepFace preflight

PixelCue's face extra now includes `tf-keras`, which DeepFace requires when used
with modern TensorFlow/Keras 3 environments. The DeepFace preflight now checks
the complete runtime stack rather than merely testing whether the `deepface`
package exists:

- DeepFace package availability;
- TensorFlow version;
- `tf_keras` availability for TensorFlow 2.16+;
- an actual `from deepface import DeepFace` compatibility import.

If any part fails, face analysis is disabled once for that scan and PixelCue
prints the exact active-interpreter repair command. JoyCaption and filesystem
processing continue unaffected.


### 0.2.5 stable DeepFace runtime

PixelCue no longer attempts DeepFace under Python 3.14. Stable TensorFlow 2.21.0
publishes Windows wheels through Python 3.13, while Python 3.14 currently gets
TensorFlow 2.22 release-candidate builds and the available `tf-keras` line is
still 2.21.x. Mixing those generations produces TensorFlow internal-API errors.

The `faces` extra is pinned to TensorFlow 2.21.0 + tf-keras 2.21.0 on Python
3.10–3.13. PixelCue itself remains usable on Python 3.14; only face analysis is
disabled there, with a preflight message showing how to create a Python 3.13
environment.


### 0.3.0 uv-managed environment

PixelCue is now uv-first and pins Python **3.13** in `.python-version`.
Use uv to install the interpreter, create/sync `.venv`, and run the application:

```powershell
uv python install 3.13
uv sync --extra faces
uv run pixelcue I:\
```

No manual virtual-environment activation is required. For development:

```powershell
uv sync --extra faces --group dev
uv run pytest
```

`uv sync` generates `uv.lock` when needed. Commit that file so the environment can be reproduced exactly.


### 0.3.1 current uv installation path

The top-level installation instructions are now uv-first. CUDA PyTorch is a normal project dependency routed through uv's explicit CUDA 13.2 PyTorch index, so `uv sync` cannot silently replace it with a CPU-only wheel. The full supported install is now simply:

```powershell
uv python install 3.13
uv sync --extra faces
uv run pixelcue <path>
```

The first sync creates `uv.lock` if it is not already present; commit the resulting lockfile.


### 0.4.1 robust DeepFace detector selection

PixelCue now defaults DeepFace to **RetinaFace** rather than OpenCV's Haar-cascade
detector. The OpenCV backend depends on
`cv2/data/haarcascade_frontalface_default.xml`, which is not present in every
OpenCV wheel/environment.

`PIXELCUE_FACE_DETECTOR=opencv` is still supported, but PixelCue first verifies
that the Haar cascade actually exists. If it does not, PixelCue automatically
falls back to RetinaFace, or MTCNN if RetinaFace is unavailable.

The `faces` uv extra explicitly includes both `retina-face` and `mtcnn`, so:

```powershell
uv sync --extra faces
uv run pixelcue I:\
```

installs a detector path that does not depend on OpenCV's bundled cascade XML.


### 0.4.2 safe top-level gallery lifetime tracking

Closing a tag or face-cluster gallery no longer calls QWidget methods on an
already-destroyed Qt C++ object. PixelCue now tracks each independent top-level
gallery by Python object identity and removes its reference from the relevant
window collection when Qt emits `destroyed`. No `isHidden()`/visibility call is
made during destruction.


### 0.4.3 DeepFace legacy-Keras initialization

PixelCue now sets `TF_USE_LEGACY_KERAS=1` as soon as its face-analysis module is
imported, before TensorFlow, RetinaFace, or DeepFace can initialise Keras.

The DeepFace runtime preflight no longer imports TensorFlow just to read its
version. It uses package metadata instead, then imports DeepFace only after the
legacy-Keras flag is in place. This prevents Keras 3 `KerasTensor` objects from
being mixed with the legacy TensorFlow/Keras models used by DeepFace.


### 0.4.4 FaceIdentity confirmation and detector fallback

`FaceIdentity` from the VLM is now treated as a candidate signal, not proof that
DeepFace must find a face.

For each candidate image PixelCue tries a small detector fallback chain
(starting with the configured detector, then RetinaFace/MTCNN and OpenCV when
available). If none confirms a face:

- no embedding is fabricated;
- the image is not placed in a person cluster;
- no error popup is shown;
- TSV `extra_metadata.deepface.status` is set to `face_not_confirmed`;
- PixelCue immediately continues with the next face job.

This handles small, angled, occluded, blurred, or false-positive VLM face tags
without turning expected detector disagreement into a processing error.


### 0.5.0 cluster-level face attributes

DeepFace work is now split into two phases.

**During scanning**, `FaceIdentity` images only perform face confirmation and
ArcFace embedding extraction. These embeddings drive the continuously-updated
identity clusters. Age, gender, race/ethnicity, emotion and celebrity look-alike
are no longer calculated on every face image.

**After all media and face embeddings are complete**, PixelCue freezes the final
clusters. For each cluster it sorts members by cosine distance to the cluster
centroid and considers at most the three most central images.

It analyses the first two central images. If their stable attributes agree with
sufficient confidence (gender, race/ethnicity and age within the configured
tolerance), PixelCue stops there and skips the third sample. Emotion is
aggregated but is deliberately not an early-stop criterion because expression
can legitimately vary between images of the same person. If configured
celebrity matches exist on both samples, disagreement prevents early stopping.

The final cluster summary is written to every member image's
`extra_metadata.face_cluster` and shown in the face-cluster GUI tooltip/window.


### 0.6.0 responsive GUI, process-isolated clustering, and verbose worker logs

PixelCue now bounds GUI work even on very large scans. The tag cloud materialises
only the top **100** tags initially; as the user scrolls near the bottom it loads
the next 100. Tag events are coalesced and visible buttons are updated in place,
rather than rebuilding thousands of Qt widgets after every image. The face
cluster pane uses the same 100-at-a-time lazy materialisation strategy.

Each face-cluster tile explicitly displays its image count, e.g.
`Person cluster 7 — 23 images`.

CPU-heavy embedding clustering now runs in a persistent **separate process** and
communicates with the scanner over multiprocessing queues. Filesystem discovery,
model preparation, VLM tagging, video sampling, and DeepFace remain off the Qt
main thread; high-frequency filesystem progress signals are throttled. This
removes the main sources of periodic GUI stalls while avoiding unnecessary model
copies in separate processes.

Verbose lifecycle logging is enabled by default on stdout. Worker log lines
include timestamp, PID, thread name, worker name, event, success/failure and
reason. Lifecycle and failure logging covers filesystem scanning, VLM model
preparation, VLM media analysis, video sampling, DeepFace analysis and the face
clustering subprocess. Set `PIXELCUE_VERBOSE=0` to silence it.


### 0.6.1 LZ4 cleanup-warning filtering

Some archive/decompression dependency paths can trigger a benign
`lz4.frame` finalizer warning after an otherwise successful media job:

```text
Exception ignored in: <_io.BufferedReader>
ValueError: I/O operation on closed file.
```

PixelCue now installs a narrow `sys.unraisablehook` at startup. It suppresses
only that specific closed-file `ValueError` when the traceback originates in
`lz4/frame`. All other unraisable exceptions continue through Python's normal
handler.

The first suppressed occurrence is still written once through PixelCue's
verbose stdout logger as a warning, so the condition remains observable without
flooding the console.


### 0.6.2 explicit descending image queue

Still images now use a dedicated, explicitly ordered pending list rather than a
generic heap shared with archives.

Whenever PixelCue discovers an eligible image it:

1. reads its `size_bytes`;
2. compares it with the first pending image;
3. if larger, inserts it at index 0 so it becomes the next candidate;
4. otherwise walks down the list until it reaches the first smaller image and
   inserts immediately before it;
5. inserts after existing equal-sized images so ties retain discovery order.

The VLM worker always takes `image_jobs.pop(0)`, so the largest currently
discovered-but-unprocessed image is always next. Archives and sampled videos
have separate queues and cannot disturb image-to-image filesize ordering.

Verbose stdout logging now includes `vlm-media-queue image-insert` events with
the image filesize and resulting queue position.


### 0.6.3 instant tag-cloud search

The tag cloud now has an instant, case-insensitive **Filter tags…** field.

Filtering operates on PixelCue's in-memory tag index rather than materialising
every tag as a Qt widget. The existing lazy paging still applies to search
results:

- first 100 matching tags are rendered immediately;
- scrolling near the bottom materialises the next 100 matches;
- changing the search resets to the first page and scrolls to the top;
- clearing the field instantly restores the normal frequency-ranked cloud;
- the count label shows matching tags versus total tags.

This keeps searches responsive even when the scan has accumulated thousands of
distinct tags.


### 0.7.0 cumulative tag filtering inside galleries

Both thumbnail gallery types now support live, cumulative tag filtering.

#### Tag galleries

If the main cloud opens a gallery for `rock`, that tag remains a locked
requirement. The popup's **Add required tag** field searches only tags which
actually occur among the base gallery images.

Typing `clear water` previews:

```text
rock AND clear water
```

and therefore shows only images carrying both tags. Press Enter or **Add** to
commit the additional requirement; further tags can then be added in the same
way. **Clear added filters** returns to the original clicked tag.

#### Person-cluster galleries

Face-cluster galleries use the identical filter UI. Cluster membership is the
base image set, and any added tag requirements narrow that set with AND
semantics.

The gallery search offers case-insensitive substring autocomplete, while
committed filters resolve to actual tag names. Thumbnail rendering is lazy in
pages of 100 so filtering a large gallery does not create every Qt thumbnail
widget at once.


### 0.8.0 permanent analysis history and restart/resume

PixelCue now keeps analysis results in a permanent SQLite cache instead of
relying on the final TSV as the only record. By default the database is stored
in PixelCue's per-user application-data directory as `analysis-cache.sqlite3`.
Set `PIXELCUE_ANALYSIS_DB` to use another location.

Each cached result is tied to the file's absolute path, byte size and nanosecond
modification timestamp. On a later scan, if those still match, PixelCue restores
the previous result and skips the expensive model stage. If the file changed,
it receives a new analysis while the older database row remains historical.

Persisted stages include:

- VLM image/video/archive tags, keyed by the selected tagger profile;
- DeepFace face-confirmation result and ArcFace embedding vector;
- negative `face_not_confirmed` results, so false-positive `FaceIdentity` images
  are not repeatedly sent through DeepFace;
- final per-image DeepFace attribute samples used for cluster-level age, gender,
  race/ethnicity, emotion and celebrity look-alike aggregation.

Face clusters themselves are recomputed from the restored embeddings because
cluster membership can change as additional images are discovered. Expensive
attribute samples are reused from the database when possible.

Model preparation is now lazy: if a restart can satisfy all discovered media
from the permanent cache, PixelCue does not load/download the VLM at all. The
filesystem is still rescanned so new or changed files are discovered, but
already-analysed unchanged files are restored immediately and the scan continues
with only cache misses queued for analysis.


### 0.8.1 reliable instant tag filtering

The main tag-cloud search now renders directly from the live search field instead
of waiting behind the coalesced background tag refresh timer. This fixes a race
visible during active scans/cache restoration where entering a filter such as
`outdoor` could leave the cloud showing the unfiltered top 100 tags.

Background tag arrivals remain throttled, but every refresh re-reads the current
`QLineEdit` text. Incoming tag events also no longer overwrite the filtered count
label with the global total. Lazy paging still applies to the filtered result set.


### 0.8.2 more tolerant person clustering

Person clustering is deliberately less strict by default.

The ArcFace cosine-distance assignment cutoff has changed from **0.35** to
**0.50**. PixelCue then performs a second cluster-to-cluster merge pass using
recomputed centroids with a default additional margin of **0.06**, giving an
effective default merge cutoff of **0.56**.

This helps photographs of the same person remain together when appearance
changes because of age, pose, expression, lighting, camera quality or partial
occlusion. The second pass also reduces fragmentation caused by discovery order
and centroid drift.

Both values remain configurable:

```powershell
$env:PIXELCUE_FACE_CLUSTER_DISTANCE="0.50"
$env:PIXELCUE_FACE_CLUSTER_MERGE_MARGIN="0.06"
```

Larger values merge more aggressively; smaller values split identities more
strictly.

The persistent analysis database stores the ArcFace embeddings, not fixed
cluster assignments, so upgrading to this release automatically reclusters
previously analysed images without rerunning DeepFace on them.

### 0.8.4 thumbmoves extracted as reusable library

The operating-system thumbnail-cache integration now lives in a separate,
independently installable Python project: **`thumbmoves`** (import
`thumbmoves`). PixelCue contains only a small logging adapter.

The reusable package exposes `get_cached_thumbnail()`,
`get_cached_thumbnail_image()`, `get_cached_thumbnail_bytes()` and
`backend_info()`, with Windows Shell and Freedesktop/XDG cache backends behind
the same API. macOS reports its Quick Look capability but deliberately returns a
cache miss for the strict cache-only API because Apple does not publish a
cache-only Quick Look query.

Until the package is published to PyPI, PixelCue vendors the complete standalone
project under `packages/thumbmoves` and `uv` installs it through a local
source. The same directory can be copied out or published independently without
any PixelCue dependency.


### 0.8.5 ThumbMoves rename

The reusable cross-platform OS thumbnail-cache library is now named
**ThumbMoves**.

- distribution / PyPI name: `thumbmoves`
- Python import: `thumbmoves`
- CLI: `thumbmoves`
- vendored PixelCue source: `packages/thumbmoves`

PixelCue imports the same cache-first API with:

```python
from thumbmoves import backend_info, get_cached_thumbnail
```

The underlying Windows Shell, Freedesktop/XDG and macOS capability behavior is
unchanged; this release is the DanceFlow-oriented package rename.

## Starting a scan from the command line

With no argument, PixelCue opens normally and lets you choose a starting folder:

```powershell
uv run pixelcue
```

If the first argument is a valid filesystem path, PixelCue opens the GUI and immediately starts scanning from that path:

```powershell
uv run pixelcue "C:\Users\me\Pictures"
uv run pixelcue "I:\Photo Archive"
uv run pixelcue .\single-image.jpg
```

The path may be a directory, ordinary file, or symbolic link. Invalid paths are ignored and PixelCue falls back to the normal folder-selection screen.

## Notes on archive support

Built in / installed support:

- ZIP
- TAR
- TAR.GZ / TGZ
- TAR.BZ2 / TBZ2
- TAR.XZ / TXZ
- 7Z
- RAR

Archive members are **not** inserted into the main filesystem queue and are not individually listed in the final TSV because they are not standalone filesystem files. The containing archive is listed and receives the union of tags from its sampled media members.

## Safety / scale

This can traverse a very large tree and may inspect sensitive media. Run it only on filesystems you are authorized to scan. The tool does not upload scanned media anywhere; JoyCaption inference runs locally after the model has been downloaded.

The scanner deliberately does **not** calculate hashes for every file because that would force a full read of all files and significantly increase I/O.

### 0.8.3 operating-system thumbnail-cache reuse

PixelCue now attempts to reuse an existing operating-system thumbnail before it
opens the original media solely to create a gallery thumbnail.

There is no single maintained Python package which provides one portable API to
the Windows, macOS and Freedesktop thumbnail caches. PixelCue therefore uses a
small platform abstraction:

- **Windows:** `IShellItemImageFactory::GetImage` with `SIIGBF_INCACHEONLY` and
  `SIIGBF_THUMBNAILONLY`, via `comtypes`. This asks Explorer's shared thumbnail
  cache only and will not extract a new thumbnail on a cache miss.
- **Linux / Freedesktop desktops:** direct lookup in the shared XDG thumbnail
  cache (`$XDG_CACHE_HOME/thumbnails`, normally `~/.cache/thumbnails`) using the
  specification's MD5-of-canonical-file-URI key. URI and modification-time PNG
  metadata are checked when present.
- **macOS:** PixelCue detects the PyObjC QuickLookThumbnailing framework as the
  native system route. Apple does not expose a strict cache-only option
  equivalent to Windows, so the current cache-only helper does not force a new
  Quick Look generation on a miss.

OS cache lookup happens in PixelCue's scanner/media worker, never on the Qt UI
thread. On a miss PixelCue falls back to its existing Pillow/video-frame
thumbnail path. Cache hits and misses are visible in verbose stdout as
`worker=thumbmoves` events.
