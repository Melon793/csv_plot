from __future__ import annotations
import sys
import os
import time
import traceback

from src.core.startup_metrics import get_metrics

_metrics = get_metrics()
_metrics.mark("module_entry")

from src.utils.platform_setup import setup_platform

ico_path = setup_platform()
_metrics.mark("platform_setup_done")


def _configure_pyqtgraph() -> None:
    """Load pyqtgraph only after the animated splash is visible."""
    import pyqtgraph as pg

    pg.setConfigOptions(antialias=False, crashWarning=False)
    _metrics.mark("pyqtgraph_ready")


def _apply_appearance(app, ico_path) -> None:
    """Apply app-wide font/theme immediately before any main-window widget."""
    from PySide6.QtGui import QFont, QIcon
    from src.ui import theme

    # 设了应用级样式表后 app.style() 的 name 会变空，表格选中接管按底层
    # 样式名分流，必须在下面这行之前记录。
    theme.set_base_style_name(app.style().objectName())
    app.setStyleSheet(theme.TOOLTIP_QSS)
    app.setQuitOnLastWindowClosed(False)

    if sys.platform == "win32":
        from src.core.font_cache import get_windows_chinese_font_cached

        font_name = get_windows_chinese_font_cached()
        font = QFont(font_name) if font_name else app.font()
        font.setPixelSize(12)
        app.setFont(font)
    elif sys.platform == "darwin":
        font = app.font()
        font.setPixelSize(13)
        app.setFont(font)

    if ico_path and os.path.exists(ico_path):
        app.setWindowIcon(QIcon(str(ico_path)))
    _metrics.mark("appearance_ready")


def main():
    """CSV Plot 应用程序入口。

    先显示轻量 splash，再在后台预导入主窗口依赖；主窗口仍在 UI 线程构造，
    但每个构造阶段都会把事件循环交还给 splash，避免动画被长任务冻住。
    """
    from src.core.crash_handler import install_crash_logging
    from PySide6.QtCore import Qt, QEventLoop
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    _metrics.mark("main_enter")
    # 打包版无控制台，未处理异常必须先落盘，否则闪退零线索。
    install_crash_logging()

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    _metrics.mark("qapplication_created")

    if ico_path and os.path.exists(ico_path):
        app.setWindowIcon(QIcon(str(ico_path)))

    skip_splash = "--no-splash" in sys.argv or "--clone-window" in sys.argv

    if skip_splash:
        _configure_pyqtgraph()
        _apply_appearance(app, ico_path)
        from src.ui.main_window import MainWindow

        _metrics.mark("main_construct_start")
        window = MainWindow()
        _metrics.mark("main_construct_done")
        window.show()
        app.setQuitOnLastWindowClosed(True)
        app._main_window_ref = window
    else:
        from src.ui.splash_screen import SplashScreen

        splash = SplashScreen()
        splash.show()
        # 让第一帧先真正绘制，再开始任何导入/字体/配置工作。
        app.processEvents()
        _metrics.mark("splash_visible")

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

        def preload_main_window():
            _configure_pyqtgraph()
            from src.ui.main_window import MainWindow

            return MainWindow

        def window_factory():
            _apply_appearance(app, ico_path)
            from src.ui.main_window import MainWindow

            def yield_to_splash():
                # 排除用户输入，避免半初始化窗口被点击/拖拽重入。
                app.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

            return MainWindow(startup_yield=yield_to_splash)

        _window, error = _run_splash_startup(
            app,
            splash,
            splash_delay,
            window_factory,
            preload=preload_main_window,
        )
        if error:
            _fail_startup(app, splash, error)

    _metrics.mark("event_loop_start")
    try:
        exit_code = app.exec()
    finally:
        _metrics.save()
    sys.exit(exit_code)


def _show_main_window(app, splash, window) -> None:
    """收掉启动图并把主窗口交给事件循环。"""
    splash.finish(window)
    window.show()
    app.setQuitOnLastWindowClosed(True)
    app._main_window_ref = window
    _metrics.mark("main_visible")


def _run_splash_startup(app, splash, splash_delay: int, window_factory, *, preload=None):
    """启动编排：后台预导入 → UI 线程构造主窗口 → 满足最短展示时长。

    返回 ``(window, error)``，``error`` 为空串表示启动成功。

    两条硬约束（对应「双击图标后只剩启动图、只能杀进程」这类现场事故）：

    1. **任何**路径都要让 ``splash.signal_finish()`` 被调用一次 —— 构造异常
       逃出 Qt 槽只会进 ``sys.excepthook`` 记一条日志，``wait_for_completion``
       的嵌套事件循环将永远等不到退出信号；
    2. ``wait_for_completion`` 自带看门狗，覆盖「没有异常但也没人发信号」
       的意外卡住。
    """
    from PySide6.QtCore import Qt, QThread, QTimer

    class MainWindowImportThread(QThread):
        def __init__(self, loader, parent=None):
            super().__init__(parent)
            self._loader = loader
            self.error = ""

        def run(self) -> None:
            try:
                self._loader()
            except BaseException:
                self.error = traceback.format_exc()

    window_ref: list = [None]
    error_ref: list[str] = []
    creating = False
    t0 = time.perf_counter()
    _metrics.mark("splash_startup_begin")

    def finish_startup() -> None:
        remaining = max(0, splash_delay - int((time.perf_counter() - t0) * 1000))
        if remaining == 0:
            splash.signal_finish()
            QTimer.singleShot(
                0, lambda: _show_main_window(app, splash, window_ref[0])
            )
        else:
            def delayed_finish():
                splash.signal_finish()
                _show_main_window(app, splash, window_ref[0])

            QTimer.singleShot(remaining, delayed_finish)

    def create_main_window():
        """在事件循环里构造 MainWindow，成功后调度收尾。"""
        nonlocal creating
        if creating or window_ref[0] is not None or error_ref:
            return
        creating = True
        try:
            window_ref[0] = window_factory()
        except BaseException:
            error_ref.append(traceback.format_exc())
            from src.core.logger import get_logger

            get_logger("app.boot").critical(
                "MainWindow 构造失败，终止启动:\n%s", error_ref[-1]
            )
            _metrics.mark("main_construct_failed")
            splash.signal_finish()
            return
        finally:
            creating = False

        _metrics.mark("main_construct_done")
        finish_startup()

    import_thread = None
    if preload is not None:
        import_thread = MainWindowImportThread(preload, parent=splash)

        def preload_finished() -> None:
            # QThread::finished is emitted only after run() has returned.
            error = import_thread.error
            if error:
                error_ref.append(error)
                from src.core.logger import get_logger

                get_logger("app.boot").critical(
                    "MainWindow 预导入失败，终止启动:\n%s", error
                )
                _metrics.mark("main_import_failed")
                splash.signal_finish()
                return
            _metrics.mark("main_import_done")
            create_main_window()

        # 显式排队到创建 splash 的 UI 线程，不能在导入线程构造窗口。
        import_thread.finished.connect(
            preload_finished, Qt.ConnectionType.QueuedConnection
        )
        import_thread.start()
        _metrics.mark("main_import_started")
    else:
        QTimer.singleShot(0, create_main_window)
    completed = splash.wait_for_completion()

    if import_thread is not None and import_thread.isRunning():
        # Never terminate a Python import thread: it may hold an import lock or
        # leave half-initialized modules behind.  The watchdog/external finish
        # path cannot safely continue startup, so persist diagnostics and exit
        # without Python/Qt cleanup or QThread destruction.
        message = "启动看门狗超时时后台导入仍在运行，强制退出以避免损坏模块状态"
        _metrics.save()
        try:
            from src.core.crash_handler import crash_log_path

            with open(crash_log_path(), "a", encoding="utf-8") as crash_file:
                crash_file.write(f"{message}\n")
        except (ImportError, OSError):
            pass
        from src.core.logger import get_logger

        get_logger("app.boot").critical(message)
        os._exit(1)

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
