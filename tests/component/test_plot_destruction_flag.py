"""P1-1：`_is_being_destroyed` 只在 `__init__` 置 False，全项目无处置真

8 处销毁期守卫（event_handler / cursor_manager×2 / plot_ui_manager×2 /
plot_data_manager / file_loader_manager）因此全部空转，销毁竞态只能靠各处
`except RuntimeError` 事后兜底。这里锁住置位入口以及它接上的消费点。
"""

import pandas as pd
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QWidget

from src.ui.widgets.plot_container import PlotContainerWidget
from src.ui.widgets.plot_widget import DraggableGraphicsLayoutWidget


def _drop(widget, qapp):
    """销毁 C++ 侧对象，保留 Python 包装器（正是线上残留引用的形态）"""
    widget.deleteLater()
    QCoreApplication.sendPostedEvents(widget, QEvent.Type.DeferredDelete)
    qapp.processEvents()


def _make_plot():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    return DraggableGraphicsLayoutWidget({}, df)


def test_mark_being_destroyed_flips_flag():
    """拆卸点的显式置位入口：deleteLater 到析构之间那段窗口靠它覆盖"""
    pw = _make_plot()
    assert pw._is_being_destroyed is False

    pw.mark_being_destroyed()

    assert pw._is_being_destroyed is True


def test_cpp_destruction_sets_flag_as_backstop(qapp):
    """没经过显式拆卸点的路径（主窗口退出时的子对象析构）也必须置位

    连接必须写成 lambda：连到绑定方法上时 PySide6 会在包装器失效时丢掉这条
    连接，实测不回调 —— 改成方法引用本用例即红。
    """
    pw = _make_plot()

    _drop(pw, qapp)

    assert pw._is_being_destroyed is True


def test_style_refresh_short_circuits_while_destroying(monkeypatch):
    """置位后 `update_plot_style` 不得再被调用（守卫真的接在消费点上）"""
    from src.ui.widgets import plot_ui_manager

    pw = _make_plot()
    called = []
    monkeypatch.setattr(
        plot_ui_manager.PlotUIManager,
        "update_plot_style",
        lambda *a, **k: called.append(1),
    )
    manager = pw._plot_ui_manager

    plot_ui_manager.PlotUIManager._run_style_refresh(manager, pw)
    assert called == [1], "对照组：未置位时应当刷新，否则这条断言没有意义"

    pw.mark_being_destroyed()
    plot_ui_manager.PlotUIManager._run_style_refresh(manager, pw)
    assert called == [1], "销毁期守卫没短路"


def test_layout_close_marks_contained_plots(qapp):
    """`_handle_close` 遍历的是 plot_widgets 里的容器，标记要透传到容器内 plot"""
    from src.ui.layout_manager import LayoutManager

    mw = QWidget()
    container = PlotContainerWidget(_make_plot())
    mw.plot_widgets = [container]
    mw._drop_event_filter_registered = False

    LayoutManager(mw)._handle_close()

    assert container.plot_widget._is_being_destroyed is True
