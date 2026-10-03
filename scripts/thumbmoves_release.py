"""Validate ThumbMoves source metadata and GitHub release artifacts."""
from __future__ import annotations

import argparse
import ast
from email.parser import BytesParser
import hashlib
from pathlib import Path
import re
import subprocess
import tarfile
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "packages" / "thumbmoves"
TAG = re.compile(
    r"^thumbmoves-v([0-9]+\.[0-9]+\.[0-9]+(?:(?:a|b|rc)[0-9]+|\.dev[0-9]+|\.post[0-9]+)?)$"
)


def source_version(package_root: Path = PACKAGE_ROOT) -> str:
    project = tomllib.loads(
        (package_root / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    if project["name"] != "thumbmoves":
        raise ValueError("Distribution name must remain thumbmoves.")

    tree = ast.parse(
        (package_root / "src" / "thumbmoves" / "__init__.py").read_text(
            encoding="utf-8"
        )
    )
    versions = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        )
    ]
    if versions != [project["version"]]:
        raise ValueError("pyproject.toml and thumbmoves.__version__ disagree.")
    if not TAG.fullmatch("thumbmoves-v" + project["version"]):
        raise ValueError(
            "Use a canonical X.Y.Z version, optionally with a/b/rc, .dev or .post suffix."
        )
    return project["version"]


def release_metadata(
    tag: str,
    repository: str,
    root: Path = ROOT,
    *,
    check_git: bool = True,
) -> dict[str, str]:
    match = TAG.fullmatch(tag)
    if not match:
        raise ValueError(
            "Release tag must be thumbmoves-vX.Y.Z (or a canonical Python prerelease)."
        )
    version = source_version(root / "packages" / "thumbmoves")
    if match.group(1) != version:
        raise ValueError(
            f"Tag {tag} does not match ThumbMoves {version}; update both version declarations first."
        )
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("Invalid owner/repository.")

    result = {
        "tag": tag,
        "version": version,
        "prerelease": str(
            bool(re.search(r"(?:a|b|rc|\.dev)[0-9]+$", version))
        ).lower(),
    }
    if check_git:
        def git(*args: str) -> str:
            return subprocess.check_output(
                ["git", *args], cwd=root, text=True
            ).strip()

        head = git("rev-parse", "HEAD")
        tagged = git("rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}")
        if tagged != head:
            raise ValueError("Checked-out commit is not the requested release tag.")
        result["commit"] = head
    return result


def validate_dist(directory: Path, package_root: Path = PACKAGE_ROOT) -> list[Path]:
    version = source_version(package_root)
    wheels = sorted(directory.glob("*.whl"))
    sdists = sorted(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("Expected exactly one ThumbMoves wheel and one source distribution.")

    expected = {"Name": "thumbmoves", "Version": version}
    required_wheel = {
        "thumbmoves/__init__.py",
        "thumbmoves/api.py",
        "thumbmoves/cli.py",
        "thumbmoves/backends/windows.py",
        "thumbmoves/backends/macos.py",
        "thumbmoves/backends/freedesktop.py",
    }
    with zipfile.ZipFile(wheels[0]) as wheel:
        names = set(wheel.namelist())
        metadata_paths = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_paths) != 1:
            raise ValueError("Wheel metadata is missing or ambiguous.")
        metadata = BytesParser().parsebytes(wheel.read(metadata_paths[0]))
        if any(metadata[key] != value for key, value in expected.items()):
            raise ValueError("Wheel name/version does not match the source.")
        missing = required_wheel - names
        if missing:
            raise ValueError(f"Wheel is missing required files: {sorted(missing)}")

    with tarfile.open(sdists[0], "r:gz") as archive:
        names = set(archive.getnames())
        metadata_paths = [
            name for name in names if name.count("/") == 1 and name.endswith("/PKG-INFO")
        ]
        if len(metadata_paths) != 1:
            raise ValueError("Source distribution metadata is missing or ambiguous.")
        metadata_file = archive.extractfile(metadata_paths[0])
        if metadata_file is None:
            raise ValueError("Could not read source distribution metadata.")
        metadata = BytesParser().parsebytes(metadata_file.read())
        if any(metadata[key] != value for key, value in expected.items()):
            raise ValueError("Source distribution name/version does not match the source.")
        prefix = metadata_paths[0].rsplit("/", 1)[0]
        required_sdist = {
            f"{prefix}/LICENSE",
            f"{prefix}/README.md",
            f"{prefix}/pyproject.toml",
            f"{prefix}/src/thumbmoves/api.py",
        }
        missing = required_sdist - names
        if missing:
            raise ValueError(f"Source distribution is missing required files: {sorted(missing)}")

    return wheels + sdists


def write_checksums(paths: list[Path], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
            for path in paths
        ),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("source")
    metadata = subparsers.add_parser("metadata")
    metadata.add_argument("--tag", required=True)
    metadata.add_argument("--repository", required=True)
    metadata.add_argument("--github-output", type=Path)
    dist = subparsers.add_parser("dist")
    dist.add_argument("directory", type=Path)
    dist.add_argument("--checksums", type=Path)
    args = parser.parse_args(argv)

    if args.command == "source":
        print(source_version())
    elif args.command == "metadata":
        result = release_metadata(args.tag, args.repository)
        output = "".join(f"{key}={value}\n" for key, value in result.items())
        print(output, end="")
        if args.github_output:
            with args.github_output.open("a", encoding="utf-8") as stream:
                stream.write(output)
    else:
        paths = validate_dist(args.directory)
        if args.checksums:
            write_checksums(paths, args.checksums)
        print("Validated:", ", ".join(path.name for path in paths))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"ThumbMoves release validation failed: {error}") from error
