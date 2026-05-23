from __future__ import annotations
import sys
import os

os.environ["PYQTGRAPH_QT_LIB"] = "PySide6"

from src.utils.paths import resource_path

SCREEN_WITDH_MARGIN = 0.3
SCREEN_HEIGHT_MARGIN = 0.3

if sys.platform == "win32":
    import ctypes
    myappid = 'mycompany.csv_plot.0.1'
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    ico_path = resource_path("assets/icon.ico")

elif sys.platform == "darwin":
    ico_path = resource_path("assets/icon.icns")


def main():
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QFont, QIcon
    from PySide6.QtWidgets import QApplication
    import pyqtgraph as pg
    import time

    pg.setConfigOptions(antialias=False)

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)

    if sys.platform == "win32":
        from src.core.font_cache import get_windows_chinese_font_cached
        font_name = get_windows_chinese_font_cached()
        font = QFont(font_name) if font_name else QApplication.font()
        font.setPixelSize(12)
        app.setFont(font)
    elif sys.platform == "darwin":
        font = QApplication.font()
        font.setPixelSize(13)
        app.setFont(font)

    if 'ico_path' in globals() and os.path.exists(ico_path):
        app.setWindowIcon(QIcon(str(ico_path)))

    app.setQuitOnLastWindowClosed(False)
    skip_splash = "--no-splash" in sys.argv

    if skip_splash:
        from src.ui.main_window import MainWindow
        window = MainWindow()
        window.show()
        app.setQuitOnLastWindowClosed(True)
        app._main_window_ref = window
    else:
        from src.ui.splash_screen import SplashScreen
        splash = SplashScreen()
        splash.show()
        app.processEvents()

        _MIN_SPLASH_MS = 800
        delay_arg = next((a for a in sys.argv if a.startswith("--splash-delay=")), None)
        splash_delay = int(delay_arg.split("=")[1]) * 1000 if delay_arg else _MIN_SPLASH_MS

        t0 = time.perf_counter()

        def _do_finish(window):
            splash.finish(window)
            window.show()
            app.setQuitOnLastWindowClosed(True)
            app._main_window_ref = window

        def _create_and_finish():
            from src.ui.main_window import MainWindow
            window = MainWindow()
            elapsed = (time.perf_counter() - t0) * 1000
            remaining = max(0, splash_delay - int(elapsed))
            if remaining == 0:
                _do_finish(window)
            else:
                QTimer.singleShot(remaining, lambda w=window: _do_finish(w))

        QTimer.singleShot(0, _create_and_finish)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
