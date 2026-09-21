"""延迟回调的销毁安全：窗口析构后打到已删控件的 RuntimeError 不得逃逸出槽。

缺陷形态（`src/ui/file_loader_manager.py`）：`_restore_cursor_state_after_reload`
末尾用**两参** `QTimer.singleShot(50, self._post_reload_ui_refresh)` 排延迟回调，
Qt 无从知道这个回调归谁 —— 窗口销毁不取消它。回调在窗口析构后才投递时：

1. `widget.setUpdatesEnabled(True)` 抛 `RuntimeError: ... already deleted`，
   被回调自身的 `except Exception` 吞掉；
2. **异常处理器自己**又去 `self._safety_timer.stop()`（C++ 已随窗口销毁），
   再抛一次、这次没人兜 —— 被 PySide6 交给 sys.excepthook，pytest-qt 记成
   `ERROR at teardown`（用例名随机、77 passed 仍全绿）。

本文件钉住三道防线（缺一条就回到上面那条链）：
- 排延迟回调一律用带 context 的三参重载（Qt 在 context 销毁时自动取消）；
- 回调摸控件前先看 `mw._is_being_destroyed` —— pytest-qt 收尾时窗口是
  「已 close 但 Python 包装器还活着」，三参重载在这个窗口里**不会**取消；
- 清理期的 `_safety_timer.stop()` 自身对已删 C++ 对象免疫。
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QPushButton, QWidget

import src.ui.file_loader_manager as flm_mod
from src.ui.file_loader_manager import FileLoaderManager


class _Recorder:
    """记录属性写入与方法调用；用于断言「一个控件都没摸」。"""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def _record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return self
        return _record

    def __setattr__(self, name, value):
        if name == "calls":
            object.__setattr__(self, name, value)
            return
        self.calls.append((f"set:{name}", value))
        object.__setattr__(self, name, value)


class _FakeMainWindow(QWidget):
    """只覆盖被测方法用到的成员（真实 MainWindow 见 tests/e2e）。"""

    def __init__(self):
        super().__init__()
        self.load_btn = QPushButton(self)
        self.reload_btn = _Recorder()
        self._is_loading_new_data = False
        self._is_being_destroyed = False
        self._data_version = 7
        self.plot_widgets = []


@pytest.fixture()
def manager(qtbot):
    mw = _FakeMainWindow()
    qtbot.addWidget(mw)
    return mw, FileLoaderManager(mw)


class TestContextOverloadCancelsPendingCallback:
    def test_deleted_context_suppresses_callback_but_two_arg_still_fires(self, qtbot):
        """三参重载的 context 一旦销毁，Qt 自动取消；两参重载照常触发（缺陷形态）。

        PySide6 不强引用 context（tmp/probe_ctx_keepalive.py 实测：丢 Python 引用
        即销毁 C++），所以这里 deleteLater 就等价于「窗口真被析构」。
        """
        fired = []
        ctx = QWidget()
        QTimer.singleShot(50, ctx, lambda: fired.append("with_ctx"))
        QTimer.singleShot(50, lambda: fired.append("no_ctx"))
        ctx.deleteLater()

        qtbot.wait(300)

        assert fired == ["no_ctx"]


class TestStopSafetyTimerIsDestructionSafe:
    def test_deleted_cpp_timer_does_not_raise(self, manager):
        """异常处理器内那次 stop() 不能逃逸（原缺陷的真正逃逸点）。"""
        import shiboken6

        _, mgr = manager
        timer = QTimer(mgr.mw)
        shiboken6.delete(timer)  # C++ 已删、Python 包装器还在 = 窗口析构后的现场
        mgr._safety_timer = timer

        mgr._stop_safety_timer()  # 修复前：RuntimeError: QTimer already deleted

        assert mgr._safety_timer is None


class TestPostReloadRefreshHonoursDestroyedFlag:
    def test_destroyed_window_touches_no_widget(self, manager):
        mw, mgr = manager
        widget = _Recorder()
        widget.plot_item = None
        widget.calls.clear()  # 布景本身的赋值不算「被摸」
        mw.plot_widgets = [SimpleNamespace(plot_widget=widget)]
        timer = QTimer(mw)
        timer.start(3000)
        mgr._safety_timer = timer
        mgr._post_reload_pending_version = mw._data_version
        mw._is_being_destroyed = True

        mgr._post_reload_ui_refresh()

        assert widget.calls == [], "窗口销毁中不得摸任何控件"
        assert mw.reload_btn.calls == []
        # 兜底定时器必须停掉：留着它 3 秒后还会去摸已删的 reload_btn
        assert mgr._safety_timer is None
        assert timer.isActive() is False

    def test_alive_window_still_refreshes(self, manager):
        """对照组：守卫不能把正常路径也吃掉。"""
        mw, mgr = manager
        widget = _Recorder()
        widget.plot_item = None
        mw.plot_widgets = [SimpleNamespace(plot_widget=widget)]
        mgr._post_reload_pending_version = mw._data_version

        mgr._post_reload_ui_refresh()

        assert ("setUpdatesEnabled", (True,), {}) in widget.calls
        assert ("setEnabled", (True,), {}) in mw.reload_btn.calls
        assert mw._is_loading_new_data is False


class TestOtherDeferredCallbacksHonourDestroyedFlag:
    def test_deferred_cursor_refresh_all_returns_before_touching_widgets(self, manager):
        mw, mgr = manager
        widget = _Recorder()
        mw._is_being_destroyed = True

        mgr._deferred_cursor_refresh_all([widget], mw._data_version)

        assert widget.calls == []

    def test_post_load_actions_leaves_state_untouched(self, manager):
        mw, mgr = manager
        mw.loaded_path = "old.csv"
        mw._is_being_destroyed = True

        mgr._post_load_actions("new.csv")

        assert mw.loaded_path == "old.csv", "窗口销毁中不得改写已加载路径"


class TestSchedulingIdiomCheck:
    def test_every_single_shot_call_passes_a_context(self):
        """两参重载就是本缺陷的根源，本模块不允许再出现。"""
        tree = ast.parse(Path(flm_mod.__file__).read_text(encoding="utf-8"))
        offenders = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "singleShot"
            and len(node.args) < 3
        ]
        assert offenders == [], f"这些行的 singleShot 缺 context 参数: {offenders}"
