"""启动编排的兜底：构造异常/意外卡住都不得把进程钉死在启动图上。

现场形态是「双击图标 → 启动图出现 → 什么都没有」：``MainWindow()`` 构造抛异常时，
``signal_finish()`` 永远不会被调用，而 ``wait_for_completion()`` 的嵌套
``QEventLoop`` 没有超时，进程既不显示窗口也不退出，只能去任务管理器杀。
打包版还没有控制台，日志之外用户拿不到任何线索。

修复后有两道防线：
1. 构造异常就地捕获 → 记日志 → 照样发 ``signal_finish`` → 由 ``_fail_startup``
   弹框说明原因并以非 0 退出；
2. ``wait_for_completion`` 自带看门狗，覆盖「没抛异常但也没人发信号」的卡住。
"""

from __future__ import annotations

import time

import pytest
from PySide6.QtWidgets import QMessageBox, QWidget

import csv_plot
from src.ui.splash_screen import SplashScreen


@pytest.fixture()
def splash(qapp, qtbot):
    sp = SplashScreen()
    qtbot.addWidget(sp)
    # 别真跑 16ms 重绘定时器：offscreen 下没意义，还会拖慢用例
    sp.timer.stop()
    sp._is_shown = False
    yield sp
    sp.close()


@pytest.fixture()
def restore_quit_policy(qapp):
    original = qapp.quitOnLastWindowClosed()
    yield qapp
    qapp.setQuitOnLastWindowClosed(original)


class TestWaitForCompletionWatchdog:
    def test_returns_true_when_finished_signalled(self, splash, qtbot):
        from PySide6.QtCore import QTimer

        timer = QTimer(splash)
        timer.setSingleShot(True)
        timer.timeout.connect(splash.signal_finish)
        timer.start(10)

        assert splash.wait_for_completion(timeout_ms=3000) is True

    def test_watchdog_breaks_silent_hang(self, splash):
        """没人发 finished 也必须返回，而不是永久阻塞。"""
        t0 = time.perf_counter()

        assert splash.wait_for_completion(timeout_ms=80) is False

        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"看门狗没有按时收口，实际耗时 {elapsed:.2f}s"
        assert splash._event_loop is None, "嵌套事件循环必须已经拆掉"

    def test_timeout_zero_disables_watchdog(self, splash, qtbot):
        """timeout_ms<=0 = 不设看门狗（旧行为，供特殊场景保留）。"""
        from PySide6.QtCore import QTimer

        timer = QTimer(splash)
        timer.setSingleShot(True)
        timer.timeout.connect(splash.signal_finish)
        timer.start(10)

        assert splash.wait_for_completion(timeout_ms=0) is True


class TestRunSplashStartup:
    def test_constructor_exception_does_not_hang(self, splash, restore_quit_policy):
        """P1-1 主诉：构造抛异常 → 拿到 traceback，事件循环照常退出。"""

        def boom():
            raise RuntimeError("构造主窗口时炸了")

        t0 = time.perf_counter()
        window, error = csv_plot._run_splash_startup(
            restore_quit_policy, splash, 0, boom
        )
        elapsed = time.perf_counter() - t0

        assert window is None
        assert "RuntimeError: 构造主窗口时炸了" in error
        assert elapsed < 5.0, f"启动异常仍然卡住了 {elapsed:.2f}s"

    def test_success_path_shows_window(self, splash, restore_quit_policy, qtbot):
        fake_window = QWidget()
        qtbot.addWidget(fake_window)

        window, error = csv_plot._run_splash_startup(
            restore_quit_policy, splash, 0, lambda: fake_window
        )

        assert error == ""
        assert window is fake_window
        # splash_delay=0 时收尾走 singleShot(0)，真进程里由随后的 app.exec()
        # 落地；测试里手动推一轮事件循环
        qtbot.wait(50)
        assert window.isVisible() is True
        assert restore_quit_policy._main_window_ref is fake_window
        assert restore_quit_policy.quitOnLastWindowClosed() is True

    def test_watchdog_timeout_without_window_is_an_error(
        self, splash, restore_quit_policy, monkeypatch
    ):
        """构造既不返回也不抛（卡死）→ 看门狗兜住，报「启动超时」。"""
        monkeypatch.setattr(
            SplashScreen, "wait_for_completion", lambda self, timeout_ms=0: False
        )

        window, error = csv_plot._run_splash_startup(
            restore_quit_policy, splash, 0, lambda: QWidget()
        )

        assert window is None
        assert "启动超时" in error


class TestFailStartup:
    def test_shows_reason_and_exits_nonzero(
        self, splash, restore_quit_policy, monkeypatch
    ):
        dialogs = []

        def fake_critical(parent, title, text, *args, **kwargs):
            dialogs.append((title, text))
            return QMessageBox.StandardButton.Ok

        monkeypatch.setattr(QMessageBox, "critical", staticmethod(fake_critical))
        # 别真调 QCoreApplication.exit()：它会把会话级 QApplication 的 quitNow
        # 置起来，同一进程里后续用例的事件循环全部失效（BLF e2e 会整片超时）
        exit_codes = []
        monkeypatch.setattr(
            restore_quit_policy, "exit", lambda code=0: exit_codes.append(code)
        )

        with pytest.raises(SystemExit) as exc_info:
            csv_plot._fail_startup(
                restore_quit_policy,
                splash,
                "Traceback (most recent call last):\n"
                '  File "main_window.py", line 1, in __init__\n'
                "RuntimeError: 构造主窗口时炸了\n",
            )

        assert exc_info.value.code == 1
        assert exit_codes == [1]
        assert len(dialogs) == 1
        title, text = dialogs[0]
        assert title == "启动失败"
        # 弹窗只放最后一行，但必须带上日志位置，否则打包版依旧零线索
        assert "RuntimeError: 构造主窗口时炸了" in text
        assert "csv_plot.log" in text
        assert "csv_plot_crash.log" in text
        assert splash._is_shown is False
