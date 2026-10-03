from __future__ import annotations

import random
import shutil
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Callable

from .media import is_image, is_video
from .metadata import MagicDetector

ARCHIVE_SUFFIXES = (
    ".zip", ".7z", ".rar", ".tar", ".tgz", ".tbz2", ".txz",
    ".tar.gz", ".tar.bz2", ".tar.xz",
)

ARCHIVE_MIMES = {
    "application/zip",
    "application/x-7z-compressed",
    "application/vnd.rar",
    "application/x-rar",
    "application/x-rar-compressed",
    "application/x-tar",
    "application/gzip",
    "application/x-gzip",
    "application/x-bzip2",
    "application/x-xz",
}

PROBE_BYTES = 128 * 1024


def is_archive(path: Path, mime: str = "") -> bool:
    """Content MIME wins completely whenever it is available."""
    mime = (mime or "").strip().lower()
    if mime:
        return mime in ARCHIVE_MIMES

    # Suffix fallback exists only for callers that genuinely have no MIME result.
    name = path.name.lower()
    return any(name.endswith(s) for s in ARCHIVE_SUFFIXES)


def _archive_kind(path: Path, mime: str = "") -> str:
    mime = (mime or "").strip().lower()
    if mime == "application/zip":
        return "zip"
    if mime == "application/x-7z-compressed":
        return "7z"
    if mime in {
        "application/vnd.rar",
        "application/x-rar",
        "application/x-rar-compressed",
    }:
        return "rar"
    if mime in {
        "application/x-tar",
        "application/gzip",
        "application/x-gzip",
        "application/x-bzip2",
        "application/x-xz",
    }:
        return "tar"

    if mime:
        return "unknown"

    # Only callers with no content MIME may fall back to the suffix.
    lower = path.name.lower()
    if lower.endswith(".zip"):
        return "zip"
    if lower.endswith(".7z"):
        return "7z"
    if lower.endswith(".rar"):
        return "rar"
    if lower.endswith((
        ".tar", ".tgz", ".tbz2", ".txz",
        ".tar.gz", ".tar.bz2", ".tar.xz",
    )):
        return "tar"
    return "unknown"


def _safe_member_name(name: str) -> bool:
    pp = PurePosixPath(name.replace("\\", "/"))
    return not pp.is_absolute() and ".." not in pp.parts


def _is_media_bytes(detector: MagicDetector, data: bytes) -> tuple[bool, str]:
    """Return whether member bytes are image/video according to libmagic."""
    mime, _desc = detector.identify_buffer(data)
    # No filename is passed here: extensions have zero influence.
    return is_image(Path("member"), mime) or is_video(Path("member"), mime), mime


@dataclass
class SampledArchive:
    tempdir: tempfile.TemporaryDirectory
    paths: list[Path]
    member_names: list[str]
    total_media_members: int
    total_members: int

    def cleanup(self) -> None:
        self.tempdir.cleanup()


def _safe_output(td: Path, idx: int, member_name: str) -> Path:
    # Keep the original suffix only so downstream tools have a convenient temp
    # filename. It has already passed content-based MIME classification.
    suffix = Path(member_name).suffix.lower()
    return td / f"member-{idx:02d}{suffix}"


def _choose(items: list, sample_size: int) -> list:
    if len(items) <= sample_size:
        return list(items)
    return random.SystemRandom().sample(items, sample_size)


def sample_archive(
    path: Path,
    sample_size: int = 5,
    mime: str = "",
    detector: MagicDetector | None = None,
) -> SampledArchive:
    """Sample actual media members from an archive using content MIME.

    Archive type is selected from libmagic MIME when supplied. Member filenames
    are never used to decide whether an entry is image/video.
    """
    detector = detector or MagicDetector()
    archive_kind = _archive_kind(path, mime)

    if archive_kind == "unknown":
        raise ValueError(
            f"Unsupported archive content type: {mime or 'unknown'} ({path.name})"
        )

    td_obj = tempfile.TemporaryDirectory(prefix="pixelcue-archive-")
    td = Path(td_obj.name)

    extracted: list[Path] = []
    sampled_names: list[str] = []
    total_members = 0
    total_media = 0

    try:
        if archive_kind == "zip":
            with zipfile.ZipFile(path) as zf:
                infos = [
                    i for i in zf.infolist()
                    if not i.is_dir() and _safe_member_name(i.filename)
                ]
                total_members = len(infos)

                media_infos = []
                for info in infos:
                    try:
                        with zf.open(info) as src:
                            header = src.read(PROBE_BYTES)
                        ok, _mime = _is_media_bytes(detector, header)
                        if ok:
                            media_infos.append(info)
                    except Exception:
                        continue

                total_media = len(media_infos)
                for idx, info in enumerate(_choose(media_infos, sample_size)):
                    out = _safe_output(td, idx, info.filename)
                    with zf.open(info) as src, out.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    extracted.append(out)
                    sampled_names.append(info.filename)

        elif archive_kind == "tar":
            with tarfile.open(path, mode="r:*") as tf:
                infos = [
                    i for i in tf.getmembers()
                    if i.isfile() and _safe_member_name(i.name)
                ]
                total_members = len(infos)

                media_infos = []
                for info in infos:
                    try:
                        src = tf.extractfile(info)
                        if src is None:
                            continue
                        with src:
                            header = src.read(PROBE_BYTES)
                        ok, _mime = _is_media_bytes(detector, header)
                        if ok:
                            media_infos.append(info)
                    except Exception:
                        continue

                total_media = len(media_infos)
                for idx, info in enumerate(_choose(media_infos, sample_size)):
                    src = tf.extractfile(info)
                    if src is None:
                        continue
                    out = _safe_output(td, idx, info.name)
                    with src, out.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    extracted.append(out)
                    sampled_names.append(info.name)

        elif archive_kind == "rar":
            import rarfile

            with rarfile.RarFile(path) as rf:
                infos = [
                    i for i in rf.infolist()
                    if not i.isdir() and _safe_member_name(i.filename)
                ]
                total_members = len(infos)

                media_infos = []
                for info in infos:
                    try:
                        with rf.open(info) as src:
                            header = src.read(PROBE_BYTES)
                        ok, _mime = _is_media_bytes(detector, header)
                        if ok:
                            media_infos.append(info)
                    except Exception:
                        continue

                total_media = len(media_infos)
                for idx, info in enumerate(_choose(media_infos, sample_size)):
                    out = _safe_output(td, idx, info.filename)
                    with rf.open(info) as src, out.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    extracted.append(out)
                    sampled_names.append(info.filename)

        elif archive_kind == "7z":
            # py7zr does not expose a stable streaming member API across the
            # supported versions, so probe members one at a time into a private
            # temp directory. Crucially, the decision is still based on libmagic
            # bytes, never the member suffix.
            import py7zr

            with py7zr.SevenZipFile(path, mode="r") as zf:
                names = [
                    n for n in zf.getnames()
                    if _safe_member_name(n) and not n.endswith("/")
                ]
            total_members = len(names)

            media_names: list[str] = []
            probe_root = td / "_probe"
            probe_root.mkdir()

            for probe_idx, name in enumerate(names):
                one = probe_root / f"p{probe_idx:08d}"
                one.mkdir()
                try:
                    # Reopen for each target: some py7zr versions cannot perform
                    # repeated extract() calls on one exhausted archive object.
                    with py7zr.SevenZipFile(path, mode="r") as zf:
                        zf.extract(path=one, targets=[name])

                    candidate = (one / name).resolve()
                    if (
                        one.resolve() not in candidate.parents
                        or not candidate.is_file()
                    ):
                        continue

                    with candidate.open("rb") as src:
                        header = src.read(PROBE_BYTES)
                    ok, _mime = _is_media_bytes(detector, header)
                    if ok:
                        media_names.append(name)
                except Exception:
                    continue
                finally:
                    shutil.rmtree(one, ignore_errors=True)

            total_media = len(media_names)
            chosen = _choose(media_names, sample_size)
            for idx, name in enumerate(chosen):
                one = probe_root / f"chosen-{idx:04d}"
                one.mkdir()
                try:
                    with py7zr.SevenZipFile(path, mode="r") as zf:
                        zf.extract(path=one, targets=[name])
                    candidate = (one / name).resolve()
                    if (
                        one.resolve() not in candidate.parents
                        or not candidate.is_file()
                    ):
                        continue

                    out = _safe_output(td, idx, name)
                    shutil.copy2(candidate, out)
                    extracted.append(out)
                    sampled_names.append(name)
                finally:
                    shutil.rmtree(one, ignore_errors=True)

            shutil.rmtree(probe_root, ignore_errors=True)

        return SampledArchive(
            td_obj,
            extracted,
            sampled_names,
            total_media,
            total_members,
        )

    except Exception:
        td_obj.cleanup()
        raise
