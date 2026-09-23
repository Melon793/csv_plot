from __future__ import annotations
import sys
import os

from src.utils.platform_setup import setup_platform

ico_path = setup_platform()


def main():
    """CSV Plot 应用程序入口。

    初始化 Qt 应用、设置 pyqtgraph 全局配置、创建主窗口并启动事件循环。
    """
    from src.core.crash_handler import install_crash_logging
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QFont, QIcon
    from PySide6.QtWidgets import QApplication
    import pyqtgraph as pg
    import time

    # 打包版无控制台，未处理异常必须先落盘，否则闪退零线索
    install_crash_logging()

    pg.setConfigOptions(antialias=False, crashWarning=False)

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)

    # D17：启动时清扫上次会话残留的惰性转换临时目录（三重保险：只扫一层、
    # 只认 <pid>_<rand8> 命名、只删死 PID 或 mtime>72h 的目录），并注册
    # 进程级 atexit 兜底。任何失败都不得阻断启动。
    from src.core.logger import get_logger
    from src.data.temp_cache_dir import TempCacheDir

    try:
        TempCacheDir.register_atexit()
        swept = TempCacheDir.sweep_stale()
        if swept:
            get_logger("app.startup").info("已清扫 %d 个残留惰性转换临时目录", swept)
    except Exception:
        get_logger("app.startup").warning("启动清扫惰性临时目录失败", exc_info=True)

    # D17 心跳：每 30 分钟刷新活动临时目录的 mtime，保证清扫的 72h 陈旧
    # 判据只对真残留生效（长期挂机/跨周末的窗口不会被误删）。
    heartbeat_timer = QTimer()
    heartbeat_timer.setInterval(30 * 60 * 1000)
    heartbeat_timer.timeout.connect(TempCacheDir.touch_all_live)
    heartbeat_timer.start()

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

    if os.path.exists(ico_path):
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
