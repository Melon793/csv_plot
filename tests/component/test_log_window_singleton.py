"""P1-9：LogWindow 单例必须防「C++ 已销毁」

`LogWindow._instance` 的 parent 是主窗口：主窗口销毁会连带销毁对话框的 C++
对象，类属性却仍持有 Python 包装器，下一次 `get_instance(...).show()` 直接抛
RuntimeError。同项目 `VariableInfoDialog._live_instance()` 已解决同类问题。
"""

import inspect

import pytest
from PySide6.QtWidgets import QWidget
from shiboken6 import isValid

from src.core.logger import LogManager
from src.ui.dialogs import log_window as log_window_module
from src.ui.dialogs.log_window import LogWindow


def _force_delete(widget) -> None:
    """真正销毁 C++ 侧对象，保留 Python 包装器（线上残留引用的形态）"""
    from PySide6.QtCore import QCoreApplication, QEvent

    widget.deleteLater()
    QCoreApplication.sendPostedEvents(widget, QEvent.Type.DeferredDelete)
    QCoreApplication.processEvents()


@pytest.fixture(autouse=True)
def _reset_singleton(qapp):
    """qapp 参数顺带保证 QApplication 存在（建 QWidget 的前置条件）"""
    yield
    LogWindow._instance = None


def test_get_instance_rebuilds_after_owner_destroyed():
    owner = QWidget()
    first = LogWindow.get_instance(owner)
    assert LogWindow._instance is first

    _force_delete(owner)

    assert not isValid(first)
    assert LogWindow._live_instance() is None
    assert LogWindow._instance is None, "死实例引用应被回收"

    second_owner = QWidget()
    second = LogWindow.get_instance(second_owner)
    assert second is not first
    assert isValid(second)
    _force_delete(second_owner)


def test_get_instance_reuses_live_singleton():
    owner = QWidget()
    assert LogWindow.get_instance(owner) is LogWindow.get_instance(owner)
    _force_delete(owner)


def test_log_window_does_not_reach_into_log_manager_privates():
    """`_ui_handler` 改由 LogManager.ui_handler 只读暴露"""
    assert LogManager.get_instance().ui_handler is not None
    source = inspect.getsource(log_window_module)
    assert "_ui_handler" not in source
