from __future__ import annotations

import importlib.metadata
import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image

DEFAULT_MODEL_ID = "fancyfeast/llama-joycaption-beta-one-hf-llava"
MODEL_ID = os.environ.get("PIXELCUE_JOYCAPTION_MODEL", DEFAULT_MODEL_ID)
MODEL_APPROX_GB = 17.0
# Hugging Face displays the repository as 5.96 GB. This is a decimal-byte
# fallback used only until/if exact per-file metadata becomes available.
MODEL_FALLBACK_TOTAL_BYTES = 17_000_000_000
MODEL_METADATA_TIMEOUT_SECONDS = 10.0

# Leave headroom for CUDA context, bitsandbytes temporary buffers, image
# activations and generation KV cache. On an 8 GiB card this budgets ~6.75 GiB
# for persistent model placement and lets Accelerate offload overflow to RAM.
DEFAULT_GPU_MEMORY_FRACTION = 0.84
DEFAULT_CPU_OFFLOAD_GIB = 24
MODEL_MANIFEST_NAME = ".pixelcue-model-manifest.json"

# PixelCue wants observable byte progress. hf-xet stores active transfer chunks
# outside local_dir, which makes local byte accounting opaque, so use the normal
# Hugging Face HTTP/LFS downloader unless the user explicitly chose otherwise.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "10")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

TAG_PROMPT = """Write a list of Booru-like tags for this image.
Output ONLY comma-separated tags, with no prose.
Include concrete subject, object, setting, action, clothing, visual-style, and composition tags where useful.
You MUST classify the image as SFW, suggestive, or NSFW.
If the image is NSFW, include the exact tag NSFW.
If any person's face is visibly present, include the exact tag FaceIdentity.
Do not identify who a person is. FaceIdentity only means that a visible human face is present.
Keep the tag list concise and avoid duplicate tags."""


@dataclass(frozen=True)
class TaggerProfile:
    id: str
    label: str
    repo_id: str
    backend: str
    description: str
    approximate_parameters: str


TAGGER_PROFILES: dict[str, TaggerProfile] = {
    "joycaption": TaggerProfile(
        id="joycaption",
        label="JoyCaption Beta One (best tags, heavy)",
        repo_id=DEFAULT_MODEL_ID,
        backend="joycaption",
        description="Highest-detail tagging; runtime NF4 with CPU offload.",
        approximate_parameters="~8B-class",
    ),
    "smolvlm-256m": TaggerProfile(
        id="smolvlm-256m",
        label="SmolVLM 256M (tiny / fastest)",
        repo_id="HuggingFaceTB/SmolVLM-256M-Instruct",
        backend="smolvlm",
        description="Very small image VLM for fast keyword tagging.",
        approximate_parameters="256M",
    ),
    "smolvlm-500m": TaggerProfile(
        id="smolvlm-500m",
        label="SmolVLM 500M (light / better quality)",
        repo_id="HuggingFaceTB/SmolVLM-500M-Instruct",
        backend="smolvlm",
        description="Small image VLM with a useful quality/speed balance.",
        approximate_parameters="500M",
    ),
    "smolvlm2-256m": TaggerProfile(
        id="smolvlm2-256m",
        label="SmolVLM2 256M (tiny, newer)",
        repo_id="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        backend="smolvlm",
        description="Newer tiny multimodal model; used here for still-image tagging.",
        approximate_parameters="256M",
    ),
    "smolvlm2-500m": TaggerProfile(
        id="smolvlm2-500m",
        label="SmolVLM2 500M (light, newer)",
        repo_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        backend="smolvlm",
        description="Newer 500M multimodal model; still lightweight on the 4060 Ti.",
        approximate_parameters="500M",
    ),
}

DEFAULT_TAGGER_PROFILE = os.environ.get("PIXELCUE_TAGGER_MODEL", "joycaption")
if DEFAULT_TAGGER_PROFILE not in TAGGER_PROFILES:
    DEFAULT_TAGGER_PROFILE = "joycaption"


def available_tagger_profiles() -> list[TaggerProfile]:
    return list(TAGGER_PROFILES.values())



class JoyCaptionError(RuntimeError):
    pass


def parse_tags(text: str) -> list[str]:
    text = text.strip()
    text = re.sub(r"^```(?:text|json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    bits = re.split(r"[,;\n]+", text)
    out: list[str] = []
    seen: set[str] = set()

    for bit in bits:
        tag = bit.strip().strip("[](){}\"'`").strip()
        tag = re.sub(r"^\s*[-*]\s*", "", tag)
        tag = re.sub(r"^\d+\.\s*", "", tag)
        if not tag:
            continue

        low = tag.casefold()
        if low == "nsfw" or re.fullmatch(r"(rating:)?\s*nsfw", low):
            tag = "NSFW"
        elif low in {"faceidentity", "face_identity", "face identity"}:
            tag = "FaceIdentity"
        elif low in {"sfw", "rating:sfw", "rating: sfw"}:
            tag = "SFW"
        elif low in {"suggestive", "rating:suggestive", "rating: suggestive"}:
            tag = "suggestive"

        key = tag.casefold()
        if key not in seen:
            seen.add(key)
            out.append(tag)

    if re.search(r"\bnsfw\b", text, re.I) and "nsfw" not in seen:
        out.append("NSFW")
    if re.search(r"\bface[_ ]?identity\b", text, re.I) and "faceidentity" not in seen:
        out.append("FaceIdentity")
    return out


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "unknown"


def default_model_dir() -> Path:
    from platformdirs import user_cache_dir
    return Path(user_cache_dir("PixelCue", "DanceFlow")) / "models" / "joycaption-beta-one-source"




def cuda_pytorch_install_command() -> str:
    """Return a Windows-friendly command for a CUDA-enabled PyTorch wheel.

    CUDA 13.2 is the preferred index for current PyTorch releases. The command
    intentionally uses the active interpreter so PixelCue fixes the environment
    it is actually running from.
    """
    exe = sys.executable
    return (
        f'"{exe}" -m pip install --upgrade --force-reinstall '
        f'torch --index-url https://download.pytorch.org/whl/cu132'
    )


def nvidia_smi_summary() -> str:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
        if proc.stderr.strip():
            return f"nvidia-smi error: {proc.stderr.strip()}"
    except Exception as e:
        return f"nvidia-smi unavailable: {type(e).__name__}: {e}"
    return "nvidia-smi did not report an NVIDIA GPU"


class JoyCaption4Bit:
    def __init__(
        self,
        model_id: str = MODEL_ID,
        model_dir: str | Path | None = None,
        status_callback: Callable[[str], None] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        display_name: str = "JoyCaption Beta One",
        fallback_total_bytes: int | None = MODEL_FALLBACK_TOTAL_BYTES,
    ) -> None:
        self.model_id = model_id
        self.display_name = display_name
        self.fallback_total_bytes = fallback_total_bytes
        self.model_dir = Path(
            os.environ.get("PIXELCUE_MODEL_DIR", str(model_dir or default_model_dir()))
        ).expanduser()
        self.status_callback = status_callback or (lambda _: None)
        self.progress_callback = progress_callback or (lambda _done, _total: None)

        self.processor = None
        self.model = None
        self.device = None
        self._downloaded_path: Path | None = None
        self._repo_files: list[tuple[str, int]] = []
        self._repo_total_bytes: int = 0

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def _status(self, message: str) -> None:
        self.status_callback(message)

    def _progress(self, done: int, total: int) -> None:
        self.progress_callback(max(0, int(done)), max(0, int(total)))

    def diagnostics(self) -> str:
        bits = [
            f"model_id={self.model_id}",
            f"model_dir={self.model_dir}",
            f"torch={_version('torch')}",
            f"transformers={_version('transformers')}",
            f"bitsandbytes={_version('bitsandbytes')}",
            f"accelerate={_version('accelerate')}",
            f"huggingface-hub={_version('huggingface-hub')}",
        ]
        try:
            import torch
            bits.append(f"cuda_available={torch.cuda.is_available()}")
            bits.append(f"torch_cuda={torch.version.cuda}")
            bits.append(f"nvidia_smi={nvidia_smi_summary()}")
            if torch.cuda.is_available():
                idx = torch.cuda.current_device()
                bits.append(f"cuda_device={idx}")
                bits.append(f"gpu={torch.cuda.get_device_name(idx)}")
                try:
                    free_b, total_b = torch.cuda.mem_get_info(idx)
                    bits.append(f"cuda_free_gib={free_b / 1024**3:.2f}")
                    bits.append(f"cuda_total_gib={total_b / 1024**3:.2f}")
                except Exception:
                    pass
        except Exception as e:
            bits.append(f"torch_diagnostics_error={e}")
        return "\n".join(bits)

    @property
    def manifest_path(self) -> Path:
        return self.model_dir / MODEL_MANIFEST_NAME

    def _write_manifest(self, files: list[tuple[str, int]], total: int) -> None:
        payload = {
            "model_id": self.model_id,
            "total_bytes": int(total),
            "files": [{"path": name, "size": int(size)} for name, size in files],
        }
        try:
            self.manifest_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except OSError:
            pass

    def _read_manifest(self) -> tuple[list[tuple[str, int]], int] | None:
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if payload.get("model_id") != self.model_id:
                return None
            files = [
                (str(item["path"]), int(item["size"]))
                for item in payload.get("files", [])
                if int(item.get("size", 0)) >= 0
            ]
            total = int(payload.get("total_bytes", 0))
            if files and total > 0:
                return files, total
        except Exception:
            pass
        return None

    def _fetch_repository_manifest(self) -> tuple[list[tuple[str, int]], int]:
        """Fetch exact file sizes from Hugging Face metadata."""
        from huggingface_hub import HfApi

        self._status(f"Querying {self.display_name} repository metadata: {self.model_id}")
        info = HfApi().model_info(
            self.model_id,
            files_metadata=True,
            timeout=MODEL_METADATA_TIMEOUT_SECONDS,
        )

        files: list[tuple[str, int]] = []
        for sibling in info.siblings or []:
            name = getattr(sibling, "rfilename", None)
            size = getattr(sibling, "size", None)
            if name and size is not None:
                files.append((str(name), int(size)))

        total = sum(size for _, size in files)
        if total <= 0:
            raise RuntimeError("Hugging Face did not return model file sizes.")

        self._repo_files = files
        self._repo_total_bytes = total
        self._write_manifest(files, total)
        return files, total

    def _get_repository_manifest(self) -> tuple[list[tuple[str, int]], int]:
        try:
            return self._fetch_repository_manifest()
        except Exception:
            cached = self._read_manifest()
            if cached:
                self._repo_files, self._repo_total_bytes = cached
                return cached
            raise

    def _completed_final_file_bytes(self, files: list[tuple[str, int]]) -> int:
        done = 0
        for rel, expected in files:
            p = self.model_dir / Path(rel)
            try:
                if p.is_file():
                    done += min(int(p.stat().st_size), expected)
            except OSError:
                pass
        return done

    def _partial_download_bytes(self) -> int:
        """Count Hugging Face local_dir .incomplete transfer files."""
        cache_root = self.model_dir / ".cache" / "huggingface" / "download"
        if not cache_root.exists():
            return 0

        total = 0
        try:
            for p in cache_root.rglob("*.incomplete"):
                try:
                    total += int(p.stat().st_size)
                except OSError:
                    pass
        except OSError:
            pass
        return total

    def _local_payload_bytes(self) -> int:
        """Count payload bytes already materialised in local_dir.

        This deliberately works without repository metadata so download progress can
        start immediately. Completed payload files are counted outside `.cache`, while
        resumable Hugging Face `.incomplete` files are counted inside its download cache.
        """
        total = 0
        try:
            if self.model_dir.exists():
                for p in self.model_dir.rglob("*"):
                    if not p.is_file():
                        continue
                    try:
                        rel_parts = p.relative_to(self.model_dir).parts
                    except ValueError:
                        continue
                    if not rel_parts:
                        continue
                    # Ignore Hugging Face bookkeeping except active partial payloads.
                    if rel_parts[0] == ".cache":
                        if p.name.endswith(".incomplete"):
                            try:
                                total += int(p.stat().st_size)
                            except OSError:
                                pass
                        continue
                    if p.name == MODEL_MANIFEST_NAME:
                        continue
                    try:
                        total += int(p.stat().st_size)
                    except OSError:
                        pass
        except OSError:
            pass
        return max(0, total)

    def _downloaded_bytes(self, files: list[tuple[str, int]], total: int) -> int:
        # Completed files plus resumable partial files. Hugging Face removes/renames
        # .incomplete files on completion, so these do not normally double count.
        done = self._completed_final_file_bytes(files) + self._partial_download_bytes()
        return min(max(done, 0), total)

    def _is_complete(self, files: list[tuple[str, int]]) -> bool:
        for rel, expected in files:
            p = self.model_dir / Path(rel)
            try:
                if not p.is_file() or int(p.stat().st_size) != expected:
                    return False
            except OSError:
                return False
        return bool(files)

    def _available_cuda_memory_bytes(self) -> tuple[int, int]:
        """Return (free, total) CUDA bytes after clearing reclaimable cache."""
        import torch

        torch.cuda.empty_cache()
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        return int(free_bytes), int(total_bytes)

    def _low_vram_max_memory(self) -> dict:
        """Build an Accelerate max_memory budget with deliberate GPU headroom."""
        free_bytes, total_bytes = self._available_cuda_memory_bytes()

        # Budget against physical VRAM, but never claim more than currently free
        # memory. Keep at least 768 MiB free even on unusual card sizes.
        fraction_budget = int(total_bytes * DEFAULT_GPU_MEMORY_FRACTION)
        reserve = 768 * 1024 * 1024
        free_budget = max(0, free_bytes - reserve)
        gpu_budget = min(fraction_budget, free_budget)

        if gpu_budget < 3 * 1024**3:
            raise JoyCaptionError(
                "Not enough free VRAM to load JoyCaption safely.\n\n"
                f"CUDA free={free_bytes / 1024**3:.2f} GiB; "
                f"total={total_bytes / 1024**3:.2f} GiB.\n"
                "Close other GPU-heavy applications and retry."
            )

        cpu_budget_gib = int(
            os.environ.get(
                "PIXELCUE_CPU_OFFLOAD_GIB",
                str(DEFAULT_CPU_OFFLOAD_GIB),
            )
        )

        return {
            0: f"{gpu_budget // (1024**2)}MiB",
            "cpu": f"{cpu_budget_gib}GiB",
        }

    def _offload_dir(self) -> Path:
        path = self.model_dir / ".offload"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def validate_runtime(self) -> None:
        """Fail early with an actionable diagnostic before downloading/loading 6 GB."""
        try:
            import torch
        except Exception as e:
            raise JoyCaptionError(
                "PyTorch is not installed, so JoyCaption cannot use the GPU.\n\n"
                f"Install a CUDA-enabled PyTorch build with:\n"
                f"{cuda_pytorch_install_command()}\n\n"
                f"Python: {sys.version}\n"
                f"Interpreter: {sys.executable}"
            ) from e

        gpu_summary = nvidia_smi_summary()
        torch_cuda = getattr(torch.version, "cuda", None)

        if torch_cuda is None:
            raise JoyCaptionError(
                "PixelCue detected a CPU-only PyTorch installation.\n\n"
                "Windows/NVIDIA may see the GPU correctly, but this Python "
                "environment's torch wheel contains no CUDA runtime.\n\n"
                f"NVIDIA GPU/driver:\n{gpu_summary}\n\n"
                f"torch={getattr(torch, '__version__', 'unknown')}\n"
                f"torch.version.cuda=None\n"
                f"torch.cuda.is_available()={torch.cuda.is_available()}\n\n"
                "Install the CUDA-enabled PyTorch wheel into THIS interpreter:\n"
                f"{cuda_pytorch_install_command()}\n\n"
                "Then restart PixelCue."
            )

        if not torch.cuda.is_available():
            raise JoyCaptionError(
                "A CUDA-enabled PyTorch build is installed, but CUDA is not "
                "available at runtime.\n\n"
                f"NVIDIA GPU/driver:\n{gpu_summary}\n\n"
                f"torch={getattr(torch, '__version__', 'unknown')}\n"
                f"torch.version.cuda={torch_cuda}\n"
                "torch.cuda.is_available()=False\n\n"
                "This usually indicates an NVIDIA driver/runtime mismatch. "
                "Update the NVIDIA driver, then restart PixelCue."
            )

        try:
            import bitsandbytes as bnb
            _ = getattr(bnb, "__version__", "unknown")
        except Exception as e:
            raise JoyCaptionError(
                "CUDA PyTorch is available, but bitsandbytes failed to import.\n\n"
                f"NVIDIA GPU/driver:\n{gpu_summary}\n"
                f"torch CUDA={torch_cuda}\n\n"
                f"bitsandbytes error: {type(e).__name__}: {e}"
            ) from e

    def ensure_downloaded(self) -> Path:
        """Materialise the model locally without blocking on repository metadata.

        Download and exact-size lookup are concurrent. This prevents a slow Hub API
        `files_metadata=True` request from stopping the actual model transfer.
        """
        self.validate_runtime()

        if self._downloaded_path and self._downloaded_path.exists():
            return self._downloaded_path

        from huggingface_hub import snapshot_download

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._status(f"Checking {self.display_name} model cache: {self.model_dir}")

        # A previously saved exact manifest lets us verify a fully cached model
        # without touching the network at all.
        cached_manifest = self._read_manifest()
        if cached_manifest:
            cached_files, cached_total = cached_manifest
            self._repo_files = cached_files
            self._repo_total_bytes = cached_total
            done = self._downloaded_bytes(cached_files, cached_total)
            self._progress(done, cached_total)
            if self._is_complete(cached_files):
                self._downloaded_path = self.model_dir
                self._progress(cached_total, cached_total)
                self._status(
                    f"{self.display_name}: cached and complete at {self.model_dir}"
                )
                return self.model_dir

        # Start both operations immediately. The download does NOT wait for the
        # repository-size query.
        metadata_result: dict[str, object] = {}
        metadata_finished = threading.Event()

        def fetch_metadata() -> None:
            try:
                metadata_result["manifest"] = self._fetch_repository_manifest()
            except BaseException as exc:
                metadata_result["error"] = exc
            finally:
                metadata_finished.set()

        metadata_thread = threading.Thread(
            target=fetch_metadata,
            name="pixelcue-hf-metadata",
            daemon=True,
        )
        metadata_thread.start()

        download_result: dict[str, object] = {}
        download_finished = threading.Event()

        def do_download() -> None:
            try:
                download_result["path"] = snapshot_download(
                    repo_id=self.model_id,
                    local_dir=str(self.model_dir),
                    local_files_only=False,
                    max_workers=4,
                )
            except BaseException as exc:
                download_result["error"] = exc
            finally:
                download_finished.set()

        download_thread = threading.Thread(
            target=do_download,
            name="pixelcue-huggingface-download",
            daemon=True,
        )
        download_thread.start()

        # Default denominator is useful immediately for the standard model.
        if self.fallback_total_bytes:
            total = int(self.fallback_total_bytes)
            total_is_exact = False
        else:
            total = 0
            total_is_exact = False

        self._status(
            f"Downloading {self.display_name} model to {self.model_dir} "
            f"(exact repository size is being queried in parallel)"
        )

        last_done = -1
        last_status_second = -1
        start_time = time.monotonic()
        metadata_reported = False

        while not download_finished.wait(timeout=0.25):
            # Adopt exact metadata as soon as it arrives.
            if metadata_finished.is_set() and not metadata_reported:
                metadata_reported = True
                manifest = metadata_result.get("manifest")
                if manifest is not None:
                    files, exact_total = manifest
                    self._repo_files = list(files)
                    self._repo_total_bytes = int(exact_total)
                    total = int(exact_total)
                    total_is_exact = True
                    self._status(
                        f"Downloading {self.display_name} model to {self.model_dir} "
                        f"(exact size confirmed)"
                    )
                else:
                    error = metadata_result.get("error")
                    self._status(
                        f"{self.display_name} repository-size lookup timed out/failed; "
                        "download is continuing without waiting for it"
                        + (f": {type(error).__name__}" if error else "")
                    )

            if self._repo_files and self._repo_total_bytes > 0:
                done = self._downloaded_bytes(
                    self._repo_files,
                    self._repo_total_bytes,
                )
            else:
                done = self._local_payload_bytes()

            if done != last_done:
                last_done = done
                self._progress(done, total)

            # Make an apparently idle transfer explain itself rather than looking frozen.
            elapsed = int(time.monotonic() - start_time)
            if elapsed >= 15 and elapsed // 15 != last_status_second:
                last_status_second = elapsed // 15
                if done == 0:
                    self._status(
                        "Waiting for Hugging Face to begin model transfer "
                        f"({elapsed}s elapsed; filesystem scan continues)"
                    )

        download_thread.join()

        # One final metadata check, but never wait for it here.
        if metadata_finished.is_set() and not metadata_reported:
            manifest = metadata_result.get("manifest")
            if manifest is not None:
                files, exact_total = manifest
                self._repo_files = list(files)
                self._repo_total_bytes = int(exact_total)
                total = int(exact_total)
                total_is_exact = True

        if self._repo_files and self._repo_total_bytes > 0:
            done = self._downloaded_bytes(
                self._repo_files,
                self._repo_total_bytes,
            )
        else:
            done = self._local_payload_bytes()
        self._progress(done, total)

        error = download_result.get("error")
        if error is not None:
            raise JoyCaptionError(
                f"{self.display_name} model download failed: "
                f"{type(error).__name__}: {error}\n\n"
                f"Downloaded locally: {done:,} bytes"
                + (f" / {total:,} bytes" if total > 0 else "")
                + f"\nDestination: {self.model_dir}\n"
                f"Repository: {self.model_id}\n\n{self.diagnostics()}"
            ) from error

        # snapshot_download returning successfully is authoritative. If an exact
        # manifest was obtained, perform the stronger size-by-size verification too.
        if total_is_exact and self._repo_files:
            if not self._is_complete(self._repo_files):
                exact_done = self._downloaded_bytes(
                    self._repo_files,
                    self._repo_total_bytes,
                )
                raise JoyCaptionError(
                    "JoyCaption download returned successfully but exact repository "
                    "verification found an incomplete checkpoint.\n\n"
                    f"Downloaded: {exact_done:,} / {self._repo_total_bytes:,} bytes\n"
                    f"Destination: {self.model_dir}"
                )
            self._progress(self._repo_total_bytes, self._repo_total_bytes)
        else:
            # With no exact manifest, reflect the completed local payload against
            # the best denominator available. Do not invent exactness.
            final_local = self._local_payload_bytes()
            self._progress(final_local, total)

        self._downloaded_path = Path(
            str(download_result.get("path") or self.model_dir)
        )
        self._status(f"{self.display_name}: download complete at {self.model_dir}")
        return self._downloaded_path

    def _resolve_component(self, name: str):
        """Resolve LLaVA components across Transformers 4.x and 5.x layouts."""
        assert self.model is not None

        direct = getattr(self.model, name, None)
        if direct is not None:
            return direct

        inner = getattr(self.model, "model", None)
        if inner is not None:
            nested = getattr(inner, name, None)
            if nested is not None:
                return nested

        raise JoyCaptionError(
            f"Could not locate LLaVA component {name!r}. "
            f"Transformers={_version('transformers')}; "
            f"model_class={type(self.model).__name__}; "
            f"inner_model_class={type(inner).__name__ if inner is not None else 'None'}"
        )

    def _vision_tower(self):
        return self._resolve_component("vision_tower")

    def _multimodal_projector(self):
        return self._resolve_component("multi_modal_projector")

    def _language_model(self):
        return self._resolve_component("language_model")

    def _vision_input_location(self):
        """Return the device/dtype expected by the resolved LLaVA vision tower."""
        import torch

        assert self.model is not None
        vision = self._vision_tower()

        # Prefer the first floating parameter. This is robust across CLIP/SigLIP
        # internal layouts and avoids hard-coding embeddings.patch_embedding.
        for parameter in vision.parameters():
            dtype = getattr(parameter, "dtype", None)
            if dtype is not None and dtype.is_floating_point:
                return parameter.device, dtype

        return self.model.device, torch.bfloat16

    def _language_input_device(self):
        import torch

        assert self.model is not None
        language_model = self._language_model()

        try:
            return language_model.get_input_embeddings().weight.device
        except Exception:
            pass

        try:
            return language_model.model.embed_tokens.weight.device
        except Exception:
            return torch.device("cuda:0")

    def _validate_unquantized_visual_modules(self) -> None:
        """Assert that JoyCaption's visual path remained floating-point."""
        import torch

        assert self.model is not None

        bad: list[str] = []
        for prefix, module in (
            ("vision_tower", self._vision_tower()),
            ("multi_modal_projector", self._multimodal_projector()),
        ):
            for name, parameter in module.named_parameters():
                dtype = getattr(parameter, "dtype", None)
                if dtype in {torch.uint8, torch.int8}:
                    bad.append(f"{prefix}.{name}={dtype}")
                    if len(bad) >= 20:
                        break
            if len(bad) >= 20:
                break

        if bad:
            raise JoyCaptionError(
                "JoyCaption visual modules were unexpectedly quantized even "
                "though PixelCue explicitly excluded them from NF4. "
                "This Transformers/bitsandbytes combination is incompatible.\n\n"
                + "\n".join(bad)
            )

    def load(self) -> None:
        if self.loaded:
            return

        model_path = self.ensure_downloaded()
        self._status("Verifying checkpoint and loading JoyCaption 4-bit onto GPU …")

        try:
            import torch
            from transformers import (
                AutoProcessor,
                BitsAndBytesConfig,
                LlavaForConditionalGeneration,
            )

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "A CUDA-capable NVIDIA GPU was not detected by PyTorch. "
                    "Install a CUDA-enabled PyTorch build and compatible NVIDIA driver."
                )

            self.processor = AutoProcessor.from_pretrained(
                str(model_path),
                local_files_only=True,
            )
            # Follow JoyCaption's official NF4 loader: quantize the language
            # model at runtime, but NEVER quantize SigLIP or the multimodal
            # projector. Transformers' SigLIP path is known to break when those
            # modules are bitsandbytes-quantized.
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=[
                    # Transformers 4.x checkpoint/module names.
                    "vision_tower",
                    "multi_modal_projector",
                    # Transformers 5.x composite-model names.
                    "model.vision_tower",
                    "model.multi_modal_projector",
                ],
            )

            max_memory = self._low_vram_max_memory()
            self._status(
                "Loading JoyCaption NF4 with low-VRAM placement "
                f"(GPU budget {max_memory[0]}, CPU offload {max_memory['cpu']}) …"
            )

            self.model = LlavaForConditionalGeneration.from_pretrained(
                str(model_path),
                torch_dtype="auto",
                device_map="auto",
                max_memory=max_memory,
                offload_folder=str(self._offload_dir()),
                offload_state_dict=True,
                local_files_only=True,
                low_cpu_mem_usage=True,
                quantization_config=quantization_config,
            )
            self.model.eval()
            self.device = self.model.device

            # Defensive validation: if either excluded module somehow still
            # contains integer-packed ordinary Linear weights, fail at load time
            # with a precise diagnostic rather than during the first image.
            self._validate_unquantized_visual_modules()
            device_map = getattr(self.model, "hf_device_map", {}) or {}
            cpu_modules = sum(
                1 for device in device_map.values()
                if str(device) == "cpu"
            )
            disk_modules = sum(
                1 for device in device_map.values()
                if str(device) == "disk"
            )
            self._status(
                f"JoyCaption 4-bit: ready on "
                f"{torch.cuda.get_device_name(torch.cuda.current_device())}; "
                f"offloaded modules: CPU={cpu_modules}, disk={disk_modules}"
            )
        except Exception as e:
            self.model = None
            self.processor = None
            raise JoyCaptionError(
                f"JoyCaption model load failed: {type(e).__name__}: {e}\n\n"
                f"{self.diagnostics()}"
            ) from e

    def tags_for_image(self, image: Image.Image) -> list[str]:
        self.load()

        try:
            import torch

            assert self.processor is not None
            assert self.model is not None

            image = image.convert("RGB")
            convo = [
                {"role": "system", "content": "You are a helpful image captioner."},
                {"role": "user", "content": TAG_PROMPT},
            ]
            prompt = self.processor.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.processor(
                text=[prompt],
                images=[image],
                return_tensors="pt",
            )

            vision_device, vision_dtype = self._vision_input_location()
            language_device = self._language_input_device()

            # This mirrors JoyCaption's reference path: visual tensors use the
            # vision tower's exact dtype/device, while token tensors go to the
            # language-model input device.
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(
                    device=vision_device,
                    dtype=vision_dtype,
                )
            if "input_ids" in inputs:
                inputs["input_ids"] = inputs["input_ids"].to(language_device)
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"].to(
                    language_device
                )

            with torch.inference_mode():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    suppress_tokens=None,
                    use_cache=True,
                    temperature=0.6,
                    top_k=None,
                    top_p=0.9,
                )[0]

            response_ids = generated[inputs["input_ids"].shape[1]:]
            text = self.processor.tokenizer.decode(
                response_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()

            if not text:
                raise RuntimeError("JoyCaption returned an empty response.")

            tags = parse_tags(text)
            if not tags:
                raise RuntimeError(
                    f"Could not parse any tags from JoyCaption response: {text!r}"
                )
            return tags

        except JoyCaptionError:
            raise
        except Exception as e:
            dtype_details = ""
            try:
                vision_device, vision_dtype = self._vision_input_location()
                projector_dtype = next(
                    self._multimodal_projector().parameters()
                ).dtype
                dtype_details = (
                    f"\nvision_device={vision_device}"
                    f"\nvision_dtype={vision_dtype}"
                    f"\nprojector_dtype={projector_dtype}"
                )
            except Exception:
                pass

            raise JoyCaptionError(
                f"JoyCaption inference failed: {type(e).__name__}: {e}\n\n"
                f"{self.diagnostics()}{dtype_details}"
            ) from e


class SmolVLMTagger(JoyCaption4Bit):
    """Lightweight Transformers VLM backend with the same tagger interface."""

    def __init__(self, profile: TaggerProfile, status_callback=None, progress_callback=None):
        from platformdirs import user_cache_dir

        model_dir = Path(user_cache_dir("PixelCue", "DanceFlow")) / "models" / profile.id
        super().__init__(
            model_id=profile.repo_id,
            model_dir=model_dir,
            status_callback=status_callback,
            progress_callback=progress_callback,
            display_name=profile.label,
            fallback_total_bytes=None,
        )
        self.profile = profile

    def validate_runtime(self) -> None:
        try:
            import torch  # noqa: F401
        except Exception as e:
            raise JoyCaptionError(
                f"PyTorch is required for {self.display_name}: {e}"
            ) from e

    def load(self) -> None:
        if self.loaded:
            return

        model_path = self.ensure_downloaded()
        self._status(f"Loading {self.display_name} …")
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(
                str(model_path), local_files_only=True
            )
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
            self.model = AutoModelForImageTextToText.from_pretrained(
                str(model_path),
                torch_dtype=dtype,
                local_files_only=True,
                low_cpu_mem_usage=True,
                attn_implementation="eager",
            ).to(device)
            self.model.eval()
            self.device = device
            target = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
            self._status(f"{self.display_name}: ready on {target}")
        except Exception as e:
            self.model = None
            self.processor = None
            raise JoyCaptionError(
                f"{self.display_name} model load failed: {type(e).__name__}: {e}\n\n"
                f"{self.diagnostics()}"
            ) from e

    def tags_for_image(self, image: Image.Image) -> list[str]:
        self.load()
        try:
            import torch

            assert self.processor is not None and self.model is not None
            image = image.convert("RGB")
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": TAG_PROMPT},
                ],
            }]
            prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
            for key, value in list(inputs.items()):
                if hasattr(value, "to"):
                    value = value.to(self.device)
                    if key == "pixel_values" and getattr(value, "dtype", None) is not None:
                        value = value.to(
                            torch.bfloat16 if self.device.type == "cuda" else torch.float32
                        )
                    inputs[key] = value

            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=192,
                    do_sample=False,
                    use_cache=True,
                )[0]

            input_len = inputs["input_ids"].shape[1]
            text = self.processor.batch_decode(
                output[input_len:].unsqueeze(0),
                skip_special_tokens=True,
            )[0].strip()
            tags = parse_tags(text)
            if not tags:
                raise RuntimeError(f"No tags parsed from response: {text!r}")
            return tags
        except JoyCaptionError:
            raise
        except Exception as e:
            raise JoyCaptionError(
                f"{self.display_name} inference failed: {type(e).__name__}: {e}\n\n"
                f"{self.diagnostics()}"
            ) from e


def create_image_tagger(
    profile_id: str,
    status_callback=None,
    progress_callback=None,
):
    profile = TAGGER_PROFILES.get(profile_id, TAGGER_PROFILES["joycaption"])
    if profile.backend == "smolvlm":
        return SmolVLMTagger(profile, status_callback, progress_callback)
    return JoyCaption4Bit(
        model_id=profile.repo_id,
        status_callback=status_callback,
        progress_callback=progress_callback,
        display_name=profile.label,
        fallback_total_bytes=MODEL_FALLBACK_TOTAL_BYTES,
    )
