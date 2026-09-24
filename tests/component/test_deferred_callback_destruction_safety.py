"""延迟回调的销毁安全：窗口析构后打到已删控件的 RuntimeError 不得逃逸出槽。

缺陷形态（`src/ui/file_loader_manager.py`）：`_restore_cursor_state_after_reload`
末尾用三参 `QTimer.singleShot(50, self.mw, self._post_reload_ui_refresh)` 排延迟
回调。在 Nuitka 打包 exe 里，PySide6 post-load 补丁对三参形态的重载解析不稳定，
抛 `TypeError: ... is wrong (missing signature)` → 收尾永不执行 → 双锁不释放 →
3 秒兜底销毁全部曲线（见 tmp/rca-lazy-reload-curve-vanish.md）。

修复后统一走 `_schedule_delayed(msec, slot)`：QTimer 实例 parent 归 mw，
窗口销毁即取消；`timeout.connect(slot)` 走 Nuitka `patched_connect` 正确保活槽。

本文件钉住的防线：
- 模块内不允许出现任何 `QTimer.singleShot` 调用（AST 扫描）；
- `_schedule_delayed` 的 parent 所有权保活（gc 后回调仍触发）；
- 窗口销毁后回调不触发；
- 回调摸控件前先看 `mw._is_being_destroyed`；
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
    def test_no_static_single_shot_in_this_module(self):
        """模块内不允许出现任何 QTimer.singleShot 调用（一律走 _schedule_delayed）。

        原检查只禁两参形态；修复后三参形态在 Nuitka exe 里同样不稳定，
        故收紧为全面禁止。注释/文档字符串里的出现不算（AST 只看 Call 节点）。
        """
        tree = ast.parse(Path(flm_mod.__file__).read_text(encoding="utf-8"))
        offenders = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "singleShot"
        ]
        assert offenders == [], f"这些行仍在调用 QTimer.singleShot: {offenders}"


class TestScheduleDelayedIdiom:
    """_schedule_delayed 的 parent 所有权保活与窗口销毁取消。"""

    def test_callback_fires_without_python_reference(self, qtbot):
        """调度后丢弃 Python 引用 + gc.collect，回调仍能触发（parent 保活）。"""
        import gc

        mw = _FakeMainWindow()
        qtbot.addWidget(mw)
        mgr = FileLoaderManager(mw)

        fired = []
        # 不持有返回的 timer 引用
        mgr._schedule_delayed(30, lambda: fired.append("ok"))
        gc.collect()
        gc.collect()

        qtbot.wait(300)
        assert fired == ["ok"], "parent 所有权应保活定时器，gc 不得回收"

    def test_window_destruction_cancels_callback(self, qtbot):
        """窗口 deleteLater 后回调不触发（parent 销毁 → 定时器随之销毁）。"""
        mw = _FakeMainWindow()
        qtbot.addWidget(mw)
        mgr = FileLoaderManager(mw)

        fired = []
        mgr._schedule_delayed(50, lambda: fired.append("should_not_fire"))
        mw.deleteLater()

        qtbot.wait(400)
        assert fired == [], "窗口销毁后定时器应被取消，回调不得触发"
