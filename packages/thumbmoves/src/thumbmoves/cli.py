from __future__ import annotations

import argparse
from pathlib import Path

from .api import backend_info, get_cached_thumbnail


def _size(value: str) -> tuple[int, int]:
    try:
        width, height = value.lower().split("x", 1)
        return int(width), int(height)
    except Exception as exc:
        raise argparse.ArgumentTypeError("size must look like 320x240") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read an existing OS thumbnail cache entry")
    parser.add_argument("path", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--size", type=_size, default=(320, 240))
    parser.add_argument("--format", default="JPEG")
    args = parser.parse_args(argv)

    info = backend_info()
    result = get_cached_thumbnail(args.path, args.size, output_format=args.format)
    if result is None:
        print(f"MISS backend={info.name} path={args.path}")
        return 1

    output = args.output or Path.cwd() / (args.path.stem + ".thumb." + args.format.lower())
    output.write_bytes(result.data)
    print(f"HIT backend={result.backend} size={result.size[0]}x{result.size[1]} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
