from __future__ import annotations

import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from .gui import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("PixelCue")
    win = MainWindow()
    win.show()
    QTimer.singleShot(0, win.choose_folder)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
