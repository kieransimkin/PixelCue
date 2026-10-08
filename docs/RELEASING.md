# PixelCue releases

PixelCue uses `vX.Y.Z` tags. The separate ThumbMoves package retains its
`thumbmoves-vX.Y.Z` tags; publishing PixelCue does not republish ThumbMoves.

Update `pyproject.toml`, `src/pixelcue/__init__.py` and the changelog together.
Run `python -m pytest`, `python -m build`, and
`python scripts/pixelcue_release.py --dist dist` in a Python 3.13 checkout.
Push the reviewed source to GitHub, wait for the model-free CI checks, then
push an immutable version tag. The release workflow verifies source identity,
builds a wheel and source distribution, validates their contents and publishes
them with `SHA256SUMS.txt`. Verify the downloaded assets against that manifest.

The source distribution includes the local ThumbMoves project required by
`uv sync`. The wheel depends on the independently published `thumbmoves` package.
Model checkpoints, scan inventories and databases are excluded. These checks
exercise mocked service and model boundaries; they do not establish model accuracy,
GPU compatibility or successful model downloads on every platform.

## Potential problems

- Windows default text decoding can reject UTF-8 source (for example byte `0x9d`).
  Source-contract tests read files with `encoding="utf-8"`; see the
  [Python Path.read_text documentation](https://docs.python.org/3/library/pathlib.html#pathlib.Path.read_text).
- Root and vendored tests can share filenames. Pytest's `importlib` import mode
  keeps both suites independently collectable; see
  [pytest import modes](https://docs.pytest.org/en/stable/explanation/pythonpath.html#import-modes).
- A build with `--no-isolation` reports missing `wheel` when the active interpreter
  lacks a declared build dependency. Use the normal isolated build, which installs
  the declared backend requirements; see
  [PyPA build guidance](https://build.pypa.io/en/latest/tutorial/getting-started.html).

These remedies were verified during the 8 October 2026 release preparation.
