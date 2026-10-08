from __future__ import annotations

import os
from pathlib import Path

from .model import TAGGER_PROFILES, DEFAULT_TAGGER_PROFILE


def model_profile_from_argv(argv: list[str]) -> str:
    """Read --model ID / --model=ID, falling back to configured default."""
    for index, arg in enumerate(argv[1:], start=1):
        if arg.startswith("--model="):
            candidate = arg.split("=", 1)[1].strip()
            return candidate if candidate in TAGGER_PROFILES else DEFAULT_TAGGER_PROFILE
        if arg == "--model" and index + 1 < len(argv):
            candidate = argv[index + 1].strip()
            return candidate if candidate in TAGGER_PROFILES else DEFAULT_TAGGER_PROFILE
    return DEFAULT_TAGGER_PROFILE


def startup_path_from_argv(argv: list[str]) -> str | None:
    """Return the first non-option command-line argument that is a valid path."""
    skip_next = False
    for index, arg in enumerate(argv[1:], start=1):
        if skip_next:
            skip_next = False
            continue
        if arg == "--model":
            skip_next = True
            continue
        if arg.startswith("--model=") or arg.startswith("-"):
            continue

        candidate = Path(os.path.expandvars(os.path.expanduser(arg)))
        if os.path.lexists(candidate):
            try:
                return str(candidate.absolute())
            except OSError:
                return str(candidate)
    return None
