from __future__ import annotations
import sys
import os

from src.utils.platform_setup import setup_platform, setup_windows_performance

ico_path = setup_platform()
# Windows 性能初始化（timeBeginPeriod(1)，见 docs/windows_smoothness_optimization.md §2.1）：
# 只需在定时器实际触发前生效即可，放模块级保证在事件循环启动前完成。
_applied_win_measures = setup_windows_performance()


def _log_startup_environment() -> None:
    """输出启动环境诊断信息，用于定位 Windows 流畅性问题（§2.2）。"""
    from src.core.logger import get_logger
    from PySide6.QtGui import QGuiApplication

    logger = get_logger("startup")
    app = QGuiApplication.instance()
    if app is None:
        return
    screen = app.primaryScreen()
    logger.info(
        "platform=%s refreshRate=%.1fHz dpr=%.2f win_measures=%s",
        app.platformName(),
        screen.refreshRate() if screen else -1.0,
        screen.devicePixelRatio() if screen else -1.0,
        _applied_win_measures,
    )


def main():
    """CSV Plot 应用程序入口。

    初始化 Qt 应用、设置 pyqtgraph 全局配置、创建主窗口并启动事件循环。
    """
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QFont, QIcon
    from PySide6.QtWidgets import QApplication
    import pyqtgraph as pg
    import time

    pg.setConfigOptions(antialias=False, crashWarning=False)

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    _log_startup_environment()
    # 配对清理 timeBeginPeriod（托盘驻留场景推荐，见 §2.1 cleanup 说明）
    from src.utils.platform_setup import cleanup_windows_performance
    app.aboutToQuit.connect(cleanup_windows_performance)

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
    skip_splash = "--no-splash" in sys.argv or "--clone-window" in sys.argv

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

        MIN_SPLASH_MS = 800
        delay_arg = next((a for a in sys.argv if a.startswith("--splash-delay=")), None)
        if delay_arg:
            parts = delay_arg.split("=", 1)
            if len(parts) > 1 and parts[1]:
                try:
                    splash_delay = int(parts[1]) * 1000
                except ValueError:
                    splash_delay = MIN_SPLASH_MS
            else:
                splash_delay = MIN_SPLASH_MS
        else:
            splash_delay = MIN_SPLASH_MS

        t0 = time.perf_counter()

        # 保存 window 的引用
        window_ref = [None]

        def finish_splash_and_show():
            """完成 Splash，显示主窗口"""
            window = window_ref[0]
            splash.finish(window)
            window.show()
            app.setQuitOnLastWindowClosed(True)
            app._main_window_ref = window

        def create_main_window():
            """在后台构造 MainWindow，构造完成后调度 finish"""
            from src.ui.main_window import MainWindow
            window_ref[0] = MainWindow()
            elapsed = (time.perf_counter() - t0) * 1000
            remaining = max(0, splash_delay - int(elapsed))

            if remaining == 0:
                # 立即结束
                splash.signal_finish()
                QTimer.singleShot(0, finish_splash_and_show)
            else:
                # 延迟到剩余时间后结束
                def delayed_finish():
                    splash.signal_finish()
                    finish_splash_and_show()
                QTimer.singleShot(remaining, delayed_finish)

        # 关键：使用 QTimer.singleShot 启动 MainWindow 构造
        QTimer.singleShot(0, create_main_window)

        # 进入嵌套事件循环，在此期间 Splash 动画持续流畅
        splash.wait_for_completion()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
