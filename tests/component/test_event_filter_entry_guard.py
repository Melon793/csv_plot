"""`LayoutManager._handle_event_filter` 的入口判据：断链时判空即返，不许碰 `self.mw`。

`MainWindowBaseManager.mw` 是**会自抛**的弱引用 property（断链时 `RuntimeError`）。
这个函数跑在 QApplication 级事件过滤器上，一旦异常从 Qt 回调里逃出去，就会污染解释器
状态（后续任意一次 Qt→Python 调用报 `returned a result with an exception set`，实测可致
段错误）。所以入口必须先判空，且不允许"先摸对象再判窗口"。

本用例刻意不建真窗口：`LayoutManager.__new__` + 手放 `_mw_ref`，走真实类的代码路径但
不进入 Qt 事件回路（异常只在纯 Python 调用里抛出，被 pytest 正常记录）。
"""

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QLabel

from src.ui.layout_manager import LayoutManager


def _manager_with_dropped_window() -> LayoutManager:
    lm = LayoutManager.__new__(LayoutManager)
    lm._mw_ref = lambda: None  # 模拟主窗口已被回收
    return lm


def test_handler_returns_false_after_main_window_collected(qapp):
    lm = _manager_with_dropped_window()

    assert lm._handle_event_filter(QLabel("x"), QEvent(QEvent.Type.Show)) is False
