from __future__ import annotations

import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from .cli import model_profile_from_argv, startup_path_from_argv
from .gui import MainWindow
from .logging_utils import install_unraisable_hook, log_event


def main() -> int:
    install_unraisable_hook()
    log_event("main", "start", "PixelCue application starting")

    app = QApplication(sys.argv)
    app.setApplicationName("PixelCue")

    startup_path = startup_path_from_argv(sys.argv)
    startup_model = model_profile_from_argv(sys.argv)
    win = MainWindow(
        startup_path=startup_path,
        startup_model_profile=startup_model,
    )
    win.show()

    if startup_path:
        QTimer.singleShot(0, lambda: win.start_scan(startup_path))

    exit_code = app.exec()
    log_event("main", "finish", "Qt event loop exited", exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
