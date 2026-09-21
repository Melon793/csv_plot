"""全局事件过滤器入口必须拒收"已不是 QObject"的被监视对象。

`main_window.py:460` 把 MainWindow 注册成了 **QApplication 级**事件过滤器，被监视集合是
全应用所有对象、无上限。对象销毁中途仍可能有事件投递进来，此时 `obj` 可能已被复用成
非 QObject（实测拿到 `QWidgetItem`），交给 `super().eventFilter()` 直接抛 TypeError：
在测试里表现为整个 session 的 INTERNALERROR，在生产里是日志里一坨无法归因的栈。
"""

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QLabel, QWidgetItem


def test_non_qobject_watched_object_is_rejected(main_window, qapp):
    """QWidgetItem 派生自 QLayoutItem 而非 QObject：入口就该拒收"""
    host = QLabel("host")
    item = QWidgetItem(host)

    assert main_window.eventFilter(item, QEvent(QEvent.Type.Show)) is False
