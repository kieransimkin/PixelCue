# PixelCue

PixelCue is a desktop Python tool that scans a filesystem using an explicit FIFO **todo list**, tags images/videos with **JoyCaption Beta One loaded in 4-bit NF4**, samples media inside archives, builds a live tag cloud, and exports a TSV containing every file discovered.

## Behaviour

- The selected starting directory is the first item in a `deque`.
- Items are always taken from the **oldest/front** of the queue.
- Directories are enumerated with `Path.iterdir()` (so hidden entries are included) and every child is appended to the **bottom/back** of the queue.
- Images are tagged with JoyCaption.
- Videos are sampled at `0s, 30s, 60s, ...`; tags are unioned across sampled frames.
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

`heavlav/llama-joycaption-beta-one-hf-llava-4bit`

from a pre-quantized BitsAndBytes NF4 checkpoint.

An NVIDIA CUDA GPU is strongly recommended. The first media item triggers the model download/load from Hugging Face. Model weights can be several GB and are cached by Hugging Face.

The prompt asks JoyCaption for comma-separated Booru-like tags and explicitly requests:

- `NSFW` for NSFW content.
- `FaceIdentity` whenever a person's visible face is present.
- `SFW` / `suggestive` classification where appropriate.

JoyCaption is a generative VLM, so these labels are model judgments rather than guarantees.

## Linux prerequisites

`python-magic` uses `libmagic` (the same database/tool behind the Unix `file` command):

```bash
sudo apt install libmagic1
```

RAR support may additionally require an external extractor such as `unrar`, `unar`, or `bsdtar`, depending on the archive.

## Install

Create a virtual environment, then install a CUDA-enabled PyTorch build appropriate to your system before installing this package if necessary.

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows

pip install --upgrade pip
pip install -e .
pixelcue
```



## Model download and diagnostics

PixelCue explicitly downloads its default 4-bit JoyCaption checkpoint into its own user cache on first use. The model cache check/download starts immediately when a scan starts and runs in parallel with filesystem traversal. PixelCue queries Hugging Face for the exact checkpoint file sizes and displays the exact bytes downloaded / total, human-readable GiB values, and percentage while downloading. The checkpoint is approximately **5.96 GB**.

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

## Starting a scan from the command line

With no argument, PixelCue opens normally and lets you choose a starting folder:

```bash
pixelcue
```

If the first argument is a valid filesystem path, PixelCue opens the GUI and immediately starts scanning from that path:

```bash
pixelcue ~/Pictures
pixelcue "/mnt/archive/Photos 2025"
pixelcue ./single-image.jpg
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
