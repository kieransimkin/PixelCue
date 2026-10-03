from __future__ import annotations

import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from .cli import startup_path_from_argv
from .gui import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("PixelCue")

    startup_path = startup_path_from_argv(sys.argv)
    win = MainWindow(startup_path=startup_path)
    win.show()

    # Start once Qt's event loop is active so progress updates and non-modal
    # symlink prompts remain asynchronous from the very beginning.
    if startup_path:
        QTimer.singleShot(0, lambda: win.start_scan(startup_path))

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
