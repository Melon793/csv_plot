from __future__ import annotations
import sys
import os
import time

from src.utils.platform_setup import setup_platform

ico_path = setup_platform()


def main():
    """CSV Plot 应用程序入口。

    初始化 Qt 应用、设置 pyqtgraph 全局配置、创建主窗口并启动事件循环。
    """
    from src.core.crash_handler import install_crash_logging
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFont, QIcon
    from PySide6.QtWidgets import QApplication
    import pyqtgraph as pg

    # 打包版无控制台，未处理异常必须先落盘，否则闪退零线索
    install_crash_logging()

    pg.setConfigOptions(antialias=False, crashWarning=False)

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

        def window_factory():
            from src.ui.main_window import MainWindow
            return MainWindow()

        _window, error = _run_splash_startup(app, splash, splash_delay, window_factory)
        if error:
            _fail_startup(app, splash, error)

    sys.exit(app.exec())


def _show_main_window(app, splash, window) -> None:
    """收掉启动图并把主窗口交给事件循环。"""
    splash.finish(window)
    window.show()
    app.setQuitOnLastWindowClosed(True)
    app._main_window_ref = window


def _run_splash_startup(app, splash, splash_delay: int, window_factory):
    """启动编排：构造主窗口 → 满足最短展示时长 → 收掉启动图。

    返回 ``(window, error)``，``error`` 为空串表示启动成功。

    两条硬约束（对应「双击图标后只剩启动图、只能杀进程」这类现场事故）：

    1. **任何**路径都要让 ``splash.signal_finish()`` 被调用一次 —— 构造异常
       逃出 Qt 槽只会进 ``sys.excepthook`` 记一条日志，``wait_for_completion``
       的嵌套事件循环将永远等不到退出信号；
    2. ``wait_for_completion`` 自带看门狗，覆盖「没有异常但也没人发信号」
       的意外卡住。
    """
    from PySide6.QtCore import QTimer

    window_ref: list = [None]
    error_ref: list[str] = []
    t0 = time.perf_counter()

    def create_main_window():
        """在事件循环里构造 MainWindow，成功后调度收尾。"""
        try:
            window_ref[0] = window_factory()
        except BaseException:
            import traceback

            from src.core.logger import get_logger

            error_ref.append(traceback.format_exc())
            get_logger("app.boot").critical(
                "MainWindow 构造失败，终止启动:\n%s", error_ref[-1]
            )
            splash.signal_finish()
            return

        remaining = max(0, splash_delay - int((time.perf_counter() - t0) * 1000))
        if remaining == 0:
            # 立即结束
            splash.signal_finish()
            QTimer.singleShot(
                0, lambda: _show_main_window(app, splash, window_ref[0])
            )
        else:
            # 延迟到剩余时间后结束
            def delayed_finish():
                splash.signal_finish()
                _show_main_window(app, splash, window_ref[0])

            QTimer.singleShot(remaining, delayed_finish)

    QTimer.singleShot(0, create_main_window)
    completed = splash.wait_for_completion()

    if error_ref:
        return None, error_ref[0]
    if window_ref[0] is None:
        return None, (
            "启动超时：主窗口在规定时间内没有就绪。"
            if not completed
            else "启动失败：主窗口没有创建成功。"
        )
    if not completed:
        # 看门狗超时但窗口其实已经建好（构造耗时超过阈值）：照常收尾，
        # 只把这条异常时长记进日志
        from src.core.logger import get_logger

        get_logger("app.boot").warning(
            "Splash 看门狗超时，但主窗口已构造完成 → 继续启动"
        )
        _show_main_window(app, splash, window_ref[0])
    return window_ref[0], ""


def _fail_startup(app, splash, error: str) -> None:
    """启动失败：关掉启动图、把原因摆到用户面前，然后以非 0 退出。

    打包版没有控制台，静默退出等于「双击没反应」；而留在事件循环里则是
    「只剩一张启动图」。两种都比不上一个说清楚原因的错误框。
    """
    from src.core.crash_handler import crash_log_path
    from src.core.logger import LOG_FILE_NAME, get_logger
    from PySide6.QtWidgets import QMessageBox

    splash.close()

    # 构造异常时 error 是整段 traceback，弹窗只放最后一行（异常类型 + 消息）
    last_line = error.strip().splitlines()[-1] if error.strip() else error
    reason = (
        f"主窗口初始化失败：{last_line}"
        if error.startswith("Traceback")
        else error
    )

    get_logger("app.boot").critical("启动失败：%s", error)

    QMessageBox.critical(
        None,
        "启动失败",
        f"{reason}\n\n"
        f"完整日志：\n{LOG_FILE_NAME}\n{crash_log_path()}",
    )
    app.exit(1)
    sys.exit(1)


if __name__ == "__main__":
    main()
