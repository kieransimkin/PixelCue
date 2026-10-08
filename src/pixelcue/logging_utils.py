from __future__ import annotations

import datetime as _dt
import os
import sys
import threading
import traceback
from typing import Any


def verbose_logging_enabled() -> bool:
    value = os.environ.get("PIXELCUE_VERBOSE", "1").strip().casefold()
    return value not in {"0", "false", "no", "off"}


def log_event(
    worker: str,
    event: str,
    reason: str | None = None,
    *,
    level: str = "INFO",
    **fields: Any,
) -> None:
    """Write one structured lifecycle/job line to stdout and flush immediately."""
    if not verbose_logging_enabled():
        return
    now = _dt.datetime.now().astimezone().isoformat(timespec="milliseconds")
    bits = [
        now,
        level.upper(),
        f"pid={os.getpid()}",
        f"thread={threading.current_thread().name}",
        f"worker={worker}",
        f"event={event}",
    ]
    if reason:
        bits.append(f"reason={reason!r}")
    for key, value in fields.items():
        bits.append(f"{key}={value!r}")
    print("[PixelCue] " + " ".join(bits), file=sys.stdout, flush=True)


def log_exception(worker: str, event: str, exc: BaseException, **fields: Any) -> None:
    log_event(
        worker,
        event,
        f"{type(exc).__name__}: {exc}",
        level="ERROR",
        **fields,
    )
    if verbose_logging_enabled():
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()


_ORIGINAL_UNRAISABLE_HOOK = sys.unraisablehook
_unraisable_hook_installed = False
_lz4_closed_stream_warning_logged = False


def _traceback_mentions_lz4_frame(tb) -> bool:
    while tb is not None:
        filename = str(
            getattr(tb.tb_frame.f_code, "co_filename", "")
        ).replace("\\", "/").casefold()
        if "/lz4/frame/" in filename:
            return True
        tb = tb.tb_next
    return False


def _is_benign_lz4_closed_stream(unraisable) -> bool:
    exc = getattr(unraisable, "exc_value", None)
    tb = getattr(unraisable, "exc_traceback", None)

    if not isinstance(exc, ValueError):
        return False

    if "i/o operation on closed file" not in str(exc).casefold():
        return False

    return _traceback_mentions_lz4_frame(tb)


def pixelcue_unraisable_hook(unraisable) -> None:
    """Filter only python-lz4's benign already-closed-stream finalizer."""
    global _lz4_closed_stream_warning_logged

    if _is_benign_lz4_closed_stream(unraisable):
        if not _lz4_closed_stream_warning_logged:
            _lz4_closed_stream_warning_logged = True
            log_event(
                "runtime",
                "suppressed-unraisable",
                "benign python-lz4 cleanup attempted to flush an already-closed stream",
                level="WARNING",
            )
        return

    _ORIGINAL_UNRAISABLE_HOOK(unraisable)


def install_unraisable_hook() -> None:
    global _unraisable_hook_installed
    if _unraisable_hook_installed:
        return
    sys.unraisablehook = pixelcue_unraisable_hook
    _unraisable_hook_installed = True
    log_event(
        "runtime",
        "unraisable-hook-installed",
        "PixelCue runtime cleanup filtering active",
    )
