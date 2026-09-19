"""P1-7：交互防抖 QTimer 必须挂在 plot widget 生命周期下

`_interaction_timer` 超时调用 `pw._end_interaction()`。若定时器没有父对象，
pw 在防抖窗口内被销毁时定时器仍存活并触发，对已删除的 C++ 对象发起调用。
"""

import pandas as pd
from shiboken6 import isValid


def _drop(widget, qapp):
    """销毁 C++ 侧对象，保留 Python 包装器（正是线上残留引用的形态）。

    ``deleteLater() + processEvents()`` 删不掉：DeferredDelete 事件默认不在
    processEvents 的处理范围内，必须 ``sendPostedEvents`` 主动推一把。
    """
    from PySide6.QtCore import QCoreApplication, QEvent

    widget.deleteLater()
    QCoreApplication.sendPostedEvents(widget, QEvent.Type.DeferredDelete)
    qapp.processEvents()
    return widget


def test_interaction_timer_is_parented_to_plot_widget(plot_factory):
    pw = plot_factory()
    assert pw._interaction_timer.parent() is pw


def test_pending_interaction_timer_dies_with_plot_widget(qapp):
    from src.ui.widgets.plot_widget import DraggableGraphicsLayoutWidget

    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    pw = DraggableGraphicsLayoutWidget({}, df)
    timer = pw._interaction_timer
    timer.start(5000)
    assert timer.isActive()

    _drop(pw, qapp)

    assert not isValid(pw)
    assert not isValid(timer), "定时器未随 pw 销毁，超时后会访问已删除的 C++ 对象"
