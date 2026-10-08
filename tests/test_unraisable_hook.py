import types
from pathlib import Path

from pixelcue.logging_utils import _is_benign_lz4_closed_stream

ROOT = Path(__file__).parents[1]


def _capture_tb(filename: str):
    code = compile("raise ValueError('I/O operation on closed file.')", filename, "exec")
    try:
        exec(code, {})
    except ValueError as exc:
        return exc, exc.__traceback__
    raise AssertionError("unreachable")


def test_benign_lz4_closed_stream_is_recognized():
    exc, tb = _capture_tb(r"C:\venv\Lib\site-packages\lz4\frame\__init__.py")
    args = types.SimpleNamespace(exc_value=exc, exc_traceback=tb)
    assert _is_benign_lz4_closed_stream(args)


def test_other_closed_file_valueerror_is_not_suppressed():
    exc, tb = _capture_tb(r"C:\project\ordinary.py")
    args = types.SimpleNamespace(exc_value=exc, exc_traceback=tb)
    assert not _is_benign_lz4_closed_stream(args)


def test_hook_installed_before_qt_app():
    source = (ROOT / "src" / "pixelcue" / "main.py").read_text(encoding="utf-8")
    assert source.index("install_unraisable_hook()") < source.index("QApplication(sys.argv)")


def test_hook_delegates_everything_else():
    source = (ROOT / "src" / "pixelcue" / "logging_utils.py").read_text(encoding="utf-8")
    assert '"/lz4/frame/"' in source
    assert '"i/o operation on closed file"' in source
    assert "_ORIGINAL_UNRAISABLE_HOOK(unraisable)" in source
