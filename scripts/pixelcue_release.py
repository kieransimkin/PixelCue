"""Check PixelCue release identity and packages using the shared ThumbMoves validator."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess

from thumbmoves_release import source_version, validate_dist, write_checksums

ROOT = Path(__file__).resolve().parents[1]
TAG = re.compile(r"^v([0-9]+\.[0-9]+\.[0-9]+)$")


def version(root: Path = ROOT) -> str:
    return source_version(root, distribution="pixelcue", module="pixelcue", tag_pattern=TAG)


def validate_tag(tag: str, root: Path = ROOT, *, check_git: bool = True) -> str:
    match = TAG.fullmatch(tag)
    if not match:
        raise ValueError("PixelCue release tag must be vX.Y.Z; ThumbMoves uses thumbmoves-vX.Y.Z.")
    current = version(root)
    if match.group(1) != current:
        raise ValueError(f"Tag {tag} does not match PixelCue {current}.")
    if check_git:
        def git(*args: str) -> str:
            return subprocess.check_output(["git", *args], cwd=root, text=True).strip()
        if git("rev-parse", "HEAD") != git("rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}"):
            raise ValueError("Checked-out source is not the requested release tag.")
        if git("status", "--porcelain", "--untracked-files=no"):
            raise ValueError("Release source contains tracked changes.")
    return current


def distributions(directory: Path, root: Path = ROOT) -> list[Path]:
    return validate_dist(
        directory, root, distribution="pixelcue", module="pixelcue", tag_pattern=TAG,
        required_modules={f"pixelcue/{name}.py" for name in
                          ("__init__", "main", "gui", "server", "service", "model", "analysis_store", "os_thumbnail")},
        extra_sdist={"packages/thumbmoves/pyproject.toml", "packages/thumbmoves/src/thumbmoves/api.py",
                     "scripts/pixelcue_release.py", "scripts/thumbmoves_release.py"},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag")
    parser.add_argument("--dist", type=Path)
    args = parser.parse_args()
    print(validate_tag(args.tag) if args.tag else version())
    if args.dist:
        paths = distributions(args.dist)
        write_checksums(paths, args.dist / "SHA256SUMS.txt")
        print("Validated:", ", ".join(path.name for path in paths))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"PixelCue release validation failed: {error}") from error
