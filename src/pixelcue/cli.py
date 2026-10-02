from __future__ import annotations

import os
from pathlib import Path


def startup_path_from_argv(argv: list[str]) -> str | None:
    """Return the first command-line argument when it names a valid filesystem entry."""
    if len(argv) < 2:
        return None

    candidate = Path(os.path.expandvars(os.path.expanduser(argv[1])))

    # lexists() treats a symlink itself as a valid filesystem entry even when
    # its target is missing, allowing PixelCue's normal symlink handling.
    if os.path.lexists(candidate):
        try:
            return str(candidate.absolute())
        except OSError:
            return str(candidate)
    return None
