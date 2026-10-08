# ThumbMoves

[![ThumbMoves logo](https://raw.githubusercontent.com/kieransimkin/PixelCue/thumbmoves-v0.1.1/packages/thumbmoves/docs/branding/logo.png)](https://kieransimkin.co.uk/danceflow/)

By **[Kieran Simkin](https://kieransimkin.co.uk/)** · [DanceFlow ecosystem](https://kieransimkin.co.uk/danceflow/) · [Vector logo and usage guide](docs/branding/README.md).

Cache-only thumbnail access for Windows and freedesktop, with explicit cache misses. https://kieransimkin.co.uk/


Part of the DanceFlow ecosystem, **ThumbMoves** is a small cross-platform Python library for retrieving **existing thumbnails from the operating system's thumbnail cache** without opening the original media file.

```python
from thumbmoves import get_cached_thumbnail

result = get_cached_thumbnail("/photos/example.jpg", (320, 240))
if result:
    print(result.backend, result.size)
    open("thumb.jpg", "wb").write(result.data)
```

Or if you only want bytes:

```python
from thumbmoves import get_cached_thumbnail_bytes

jpeg = get_cached_thumbnail_bytes("/photos/example.jpg")
```

## Platform behavior

| Platform | Backend | Strict cache-only? | Behavior |
|---|---|---:|---|
| Windows | Shell `IShellItemImageFactory` | Yes | Requests `SIIGBF_INCACHEONLY | SIIGBF_THUMBNAILONLY` |
| Linux / Unix desktops | Freedesktop thumbnail spec | Yes | Reads `$XDG_CACHE_HOME/thumbnails` / `~/.cache/thumbnails` by canonical file-URI MD5 |
| macOS | Quick Look capability | No public equivalent | Returns a miss rather than trigger thumbnail generation |

The API is intentionally conservative: a cache miss is `None`. It never falls back to decoding the source file. Applications can implement their own fallback after the cache lookup.

## Installation

Install the current release from PyPI:

```bash
python -m pip install thumbmoves
```

The same tested wheel and source distribution are attached to the matching GitHub release with `SHA256SUMS.txt`. Test releases are published separately on TestPyPI.

Or install a checkout for local development:

```bash
python -m pip install -e .[dev]
pytest
```

Windows installs `comtypes` automatically through a platform-scoped dependency. Other platforms do not install it.

Release tags use the `thumbmoves-vX.Y.Z` form because ThumbMoves currently lives in the PixelCue repository.

## CLI

```bash
thumbmoves ~/Pictures/photo.jpg --size 320x240 -o thumb.jpg
```

Exit code `0` means cache hit; `1` means cache miss.

## Design goals

- no dependency on a GUI toolkit;
- no opening the source media file;
- no implicit thumbnail generation in the cache-only API;
- one stable API across platforms;
- Pillow images and encoded bytes both available;
- OS-specific code isolated in backend modules.

## License

MIT.
