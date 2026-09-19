"""P1-8：legend 变量菜单的 `singleShot(0)` 回调缺存活守卫

`mouseReleaseEvent` 把 `_exec_var_menu` 排进 `singleShot(0)`，事件循环转回来之前
plot 可能已随关闭/重载销毁——`singleShot` 不会因此取消，回调摸到的只剩 Python
包装器（与 `2a86e94` 修的 X-link 防抖同类）。
"""

from PySide6.QtCore import QPoint

from src.ui.widgets.plot_ui_manager import LegendTextBrowser


def _drop(widget, qapp):
    """销毁 C++ 侧对象，保留 Python 包装器（正是线上残留引用的形态）"""
    from PySide6.QtCore import QCoreApplication, QEvent

    widget.deleteLater()
    QCoreApplication.sendPostedEvents(widget, QEvent.Type.DeferredDelete)
    qapp.processEvents()


def test_guard_helper_is_shared_not_duplicated():
    """`_widget_alive` 已上收到 core.config，layout_manager 只留别名"""
    from src.core.config import widget_alive
    from src.ui import layout_manager

    assert layout_manager._widget_alive is widget_alive


def test_legend_label_is_the_menu_owner(plot_factory):
    pw = plot_factory()
    assert isinstance(pw.legend_label, LegendTextBrowser)
    assert pw.legend_label._pw is pw


def test_exec_var_menu_on_destroyed_plot_does_not_raise(qapp):
    """销毁 plot 后回调仍被投递：必须静默跳过，而不是对已删除对象建 QMenu

    不用 `plot_factory`：它会在收尾时对已销毁的 pw 再调一次 `deleteLater`。
    """
    import pandas as pd

    from src.ui.widgets.plot_widget import DraggableGraphicsLayoutWidget

    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    pw = DraggableGraphicsLayoutWidget({}, df)
    legend = pw.legend_label
    pw.curves = {"a": object()}
    callback = legend._exec_var_menu

    _drop(pw, qapp)

    callback("a", QPoint(10, 10))
