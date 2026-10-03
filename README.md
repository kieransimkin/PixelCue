# PixelCue

PixelCue is a desktop Python tool that scans a filesystem using an explicit FIFO **todo list**, tags images/videos with **JoyCaption Beta One loaded in 4-bit NF4**, samples media inside archives, builds a live tag cloud, and exports a TSV containing every file discovered.

## Behaviour

- The selected starting directory is the first item in a `deque`.
- Items are always taken from the **oldest/front** of the queue.
- Directories are enumerated with `Path.iterdir()` (so hidden entries are included) and every child is appended to the **bottom/back** of the queue.
- Images are tagged with JoyCaption.
- `FaceIdentity` images are queued independently for DeepFace attributes, embeddings and identity clustering when the `faces` extra is installed.
- Videos are sampled in a separate worker: PixelCue considers a small set of timestamps and selects at most **five visually diverse frames**; video JoyCaption work is lower priority than normal images.
- Archives are treated as container files. Up to five randomly selected media members are sampled and their tags are unioned onto the archive itself.
- Symlinks are deferred and the GUI asks whether to follow each link without blocking the worker thread. Following directory links is cycle-protected by device/inode identity.
- When scanning finishes, the GUI automatically opens a TSV save dialog.

The TSV columns start with exactly:

1. `filename`
2. `tags`
3. `exif`

and then include path, extension, libmagic MIME/type description, size, timestamps, permissions, owner IDs, symlink target, inferred category, media/archive metadata, and any processing error.

## JoyCaption / GPU

The app loads:

`fancyfeast/llama-joycaption-beta-one-hf-llava`

from the official JoyCaption checkpoint, applying BitsAndBytes NF4 at runtime only to the language-model components.

An NVIDIA CUDA GPU is strongly recommended. The first media item triggers the model download/load from Hugging Face. Model weights can be several GB and are cached by Hugging Face.

The prompt asks JoyCaption for comma-separated Booru-like tags and explicitly requests:

- `NSFW` for NSFW content.
- `FaceIdentity` whenever a person's visible face is present.
- `SFW` / `suggestive` classification where appropriate.

JoyCaption is a generative VLM, so these labels are model judgments rather than guarantees.

## Installation (uv)

PixelCue is managed with **uv** and pins **Python 3.13** in `.python-version`. You do not need to create or activate a virtual environment manually.

### Prerequisites

- An NVIDIA CUDA-capable GPU is required for JoyCaption.
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
- JoyCaption runtime dependencies;
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

### Install without DeepFace

If you do not want face analysis:

```powershell
uv python install 3.13
uv sync
uv run pixelcue I:\
```

JoyCaption and the normal filesystem/media scan still work; `FaceIdentity` items simply will not receive DeepFace analysis.

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

PixelCue explicitly downloads the official JoyCaption Beta One source checkpoint into its own user cache on first use, then applies NF4 4-bit quantization to the language-model components at load time. The model cache check/download starts immediately when a scan starts and runs in parallel with filesystem traversal. PixelCue queries Hugging Face for the exact checkpoint file sizes and displays the exact bytes downloaded / total, human-readable GiB values, and percentage while downloading. The official source checkpoint is approximately **17 GB**.

The GUI shows model state separately from the current filesystem item:

- `Downloading JoyCaption 4-bit model ...`
- `JoyCaption 4-bit: downloaded`
- `Loading JoyCaption 4-bit model onto GPU ...`
- `JoyCaption 4-bit: ready ...`

On Windows the cache location is under the normal per-user application cache returned by `platformdirs`; PixelCue displays the exact path while downloading. You can override it with:

```bash
set PIXELCUE_MODEL_DIR=D:\Models\PixelCue
```

You can also override the model repository with `PIXELCUE_JOYCAPTION_MODEL`.

A model or inference error is no longer silent. The first one appears immediately in the GUI and every error is retained in the **Errors** window with PyTorch/CUDA/Transformers/BitsAndBytes diagnostics. After a backend-level JoyCaption failure, PixelCue continues the filesystem scan but does not repeatedly retry the broken backend for every image.


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
