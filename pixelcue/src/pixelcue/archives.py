from __future__ import annotations

import random
import shutil
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

from .media import IMAGE_EXTS, VIDEO_EXTS

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
}


def is_archive(path: Path, mime: str = "") -> bool:
    name = path.name.lower()
    return mime in ARCHIVE_MIMES or any(name.endswith(s) for s in ARCHIVE_SUFFIXES)


def _safe_member_name(name: str) -> bool:
    # Archive member names are POSIX-like even on Windows. Reject absolute and
    # parent-traversal paths before any extractor is allowed to write them.
    pp = PurePosixPath(name.replace("\\", "/"))
    return not pp.is_absolute() and ".." not in pp.parts


def _is_media_name(name: str) -> bool:
    if not _safe_member_name(name):
        return False
    ext = Path(name).suffix.lower()
    return ext in IMAGE_EXTS or ext in VIDEO_EXTS


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
    suffix = Path(member_name).suffix.lower()
    return td / f"member-{idx:02d}{suffix}"


def _sample_names(names: list[str], sample_size: int) -> list[str]:
    media = [n for n in names if _is_media_name(n) and not n.endswith("/")]
    if len(media) <= sample_size:
        return media
    return random.SystemRandom().sample(media, sample_size)


def sample_archive(path: Path, sample_size: int = 5) -> SampledArchive:
    td_obj = tempfile.TemporaryDirectory(prefix="media-todo-archive-")
    td = Path(td_obj.name)
    lower = path.name.lower()
    extracted: list[Path] = []
    sampled_names: list[str] = []
    total_members = 0
    total_media = 0

    try:
        if lower.endswith(".zip"):
            with zipfile.ZipFile(path) as zf:
                infos = [i for i in zf.infolist() if not i.is_dir()]
                total_members = len(infos)
                media_infos = [i for i in infos if _is_media_name(i.filename)]
                total_media = len(media_infos)
                chosen = media_infos if len(media_infos) <= sample_size else random.SystemRandom().sample(media_infos, sample_size)
                for idx, info in enumerate(chosen):
                    out = _safe_output(td, idx, info.filename)
                    with zf.open(info) as src, out.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    extracted.append(out)
                    sampled_names.append(info.filename)

        elif lower.endswith((".tar", ".tgz", ".tbz2", ".txz", ".tar.gz", ".tar.bz2", ".tar.xz")):
            with tarfile.open(path, mode="r:*") as tf:
                infos = [i for i in tf.getmembers() if i.isfile()]
                total_members = len(infos)
                media_infos = [i for i in infos if _is_media_name(i.name)]
                total_media = len(media_infos)
                chosen = media_infos if len(media_infos) <= sample_size else random.SystemRandom().sample(media_infos, sample_size)
                for idx, info in enumerate(chosen):
                    src = tf.extractfile(info)
                    if src is None:
                        continue
                    out = _safe_output(td, idx, info.name)
                    with src, out.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    extracted.append(out)
                    sampled_names.append(info.name)

        elif lower.endswith(".7z"):
            import py7zr
            with py7zr.SevenZipFile(path, mode="r") as zf:
                names = zf.getnames()
                total_members = len(names)
                media_names = [n for n in names if _is_media_name(n)]
                total_media = len(media_names)
                chosen = _sample_names(names, sample_size)
                # py7zr extracts selected targets; validate and copy them into flat safe names.
                zf.extract(path=td, targets=chosen)
                for idx, name in enumerate(chosen):
                    candidate = (td / name).resolve()
                    if td.resolve() not in candidate.parents and candidate != td.resolve():
                        continue
                    if candidate.is_file():
                        safe = _safe_output(td, idx, name)
                        if candidate != safe:
                            shutil.copy2(candidate, safe)
                        extracted.append(safe)
                        sampled_names.append(name)

        elif lower.endswith(".rar"):
            import rarfile
            with rarfile.RarFile(path) as rf:
                infos = [i for i in rf.infolist() if not i.isdir()]
                total_members = len(infos)
                media_infos = [i for i in infos if _is_media_name(i.filename)]
                total_media = len(media_infos)
                chosen = media_infos if len(media_infos) <= sample_size else random.SystemRandom().sample(media_infos, sample_size)
                for idx, info in enumerate(chosen):
                    out = _safe_output(td, idx, info.filename)
                    with rf.open(info) as src, out.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    extracted.append(out)
                    sampled_names.append(info.filename)
        else:
            raise ValueError(f"Unsupported archive format: {path.name}")

        return SampledArchive(td_obj, extracted, sampled_names, total_media, total_members)
    except Exception:
        td_obj.cleanup()
        raise
