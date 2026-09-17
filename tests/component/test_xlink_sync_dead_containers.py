"""延迟回调不得摸已销毁的控件（layout_manager._sync_linked_x_ranges 等）。

触发场景：`QTimer.singleShot(50, self._sync_linked_x_ranges)` 持有的是 Manager
的普通 Python 方法，关窗/关 tab 不会取消它；回调落地时 `plot_widgets` 里的
container 早已随窗口销毁——Python 侧属性照常可读，真正调到 `isVisible()` /
`geometry()` 才抛 `RuntimeError`，而它发生在 Qt 事件循环里，只会被
`sys.excepthook` 记成一条无人认领的错误（测试里更糟：随机归因给另一条用例）。

本文件用「先证明控件真的死了、再证明回调不炸」两步把守卫钉住：前提断言让
`pytest.raises(RuntimeError)` 充当陷阱存在的证据，守卫一旦被拆掉，随后的
调用断言就会以同样的 RuntimeError 失败，而不是「测了个寂寞」。
"""

from __future__ import annotations

import gc

import pytest

import pyqtgraph as pg
from PySide6.QtWidgets import QSplitter, QWidget

from src.ui.layout_manager import LayoutManager, _widget_alive


class _FakePlotWidget:
    """只暴露 layout_manager 会用到的两个名字（view_box / y_name）。"""

    def __init__(self, x_range=None):
        self.view_box = pg.ViewBox()
        self.y_name = "y"
        if x_range is not None:
            self.view_box.setXRange(*x_range, padding=0)


class _Container(QWidget):
    """真 QWidget：这样销毁之后 C++ 侧确实没了。"""

    def __init__(self, x_range=None):
        super().__init__()
        self.plot_widget = _FakePlotWidget(x_range)


class _Mw:
    """最小 MainWindow 替身（普通 Python 对象，可被 GC）。"""

    _pending_xlink_sync = True

    def __init__(self, containers=()):
        self.plot_widgets = list(containers)


class _MwWindow(QWidget):
    """真 QWidget 版替身：销毁它即销毁其下所有 container。"""

    _pending_xlink_sync = True
    _pending_splitter_adjustment = True
    _splitter_ready = True
    var_table_user_adjusted = False

    def __init__(self, containers=()):
        super().__init__()
        self.plot_widgets = list(containers)
        for c in containers:
            c.setParent(self)
        self.main_splitter = QSplitter(self)


def _drop(widget: QWidget, qapp) -> QWidget:
    """销毁 C++ 侧对象，保留 Python 包装器（正是线上残留引用的形态）。

    光靠 ``deleteLater() + processEvents()`` 删不掉：DeferredDelete 事件默认
    不在 processEvents 的处理范围里，必须 ``sendPostedEvents`` 主动推一把
    （真窗口关闭时是由 Qt 自己的退出流程推的）。
    """
    from PySide6.QtCore import QCoreApplication, QEvent

    widget.deleteLater()
    QCoreApplication.sendPostedEvents(widget, QEvent.Type.DeferredDelete)
    qapp.processEvents()
    return widget


def _manager(mw) -> LayoutManager:
    """建 Manager 并反向撑住它：LayoutManager 只被 mw 的闭包引用会立刻 GC。"""
    manager = LayoutManager(mw)
    mw._kept = manager
    return manager


# ---------------------------------------------------------------- 前提断言


def test_the_deleted_container_really_raises_on_a_widget_call(qapp):
    """前提：本文件模拟的是「属性读得到、摸控件就炸」那种残留引用。"""
    alive = _Container()
    dead = _drop(_Container(), qapp)

    assert _widget_alive(alive) is True
    assert _widget_alive(dead) is False
    assert hasattr(dead, "plot_widget")  # 弱守卫眼里的「一切正常」
    with pytest.raises(RuntimeError):
        dead.isVisible()
    with pytest.raises(RuntimeError):
        dead.geometry()


# ---------------------------------------------------------------- 守卫断言


def test_sync_skips_a_deleted_container_after_the_first_one(qapp):
    """首个存活、后面的死了：健康检查与范围同步两处都要跳过它。"""
    first = _Container(x_range=(0.0, 10.0))
    dead = _drop(_Container(x_range=(100.0, 200.0)), qapp)
    mw = _Mw([first, dead])
    manager = _manager(mw)

    manager._sync_linked_x_ranges()  # 不该抛

    assert mw._pending_xlink_sync is False
    assert first.plot_widget.view_box.viewRange()[0] == pytest.approx((0.0, 10.0))


def test_alive_container_after_a_dead_one_still_gets_synced(qapp):
    """夹在中间的死块只该跳过它自己，后面的存活块照常同步（守卫不得过杀）。

    只有 [活, 死, 活] 这种排布能区分 ``continue`` 与 ``return``：一旦有人把
    健康检查里的跳过写成提前返回，本用例就会因为第三块没被同步而失败。
    """
    first = _Container(x_range=(0.0, 10.0))
    dead = _drop(_Container(x_range=(100.0, 200.0)), qapp)
    third = _Container(x_range=(500.0, 600.0))
    third.plot_widget.view_box.setXLink(first.plot_widget.view_box)
    mw = _Mw([first, dead, third])
    manager = _manager(mw)

    manager._sync_linked_x_ranges()  # 不该抛

    assert third.plot_widget.view_box.viewRange()[0] == pytest.approx(
        (0.0, 10.0), abs=1e-6
    )


def test_sync_gives_up_when_the_first_container_is_gone(qapp):
    """首块都没了 = 整窗正在销毁：后面的范围同步无从谈起，直接返回。"""
    dead_first = _drop(_Container(x_range=(0.0, 10.0)), qapp)
    alive = _Container(x_range=(0.0, 5.0))
    mw = _Mw([dead_first, alive])
    manager = _manager(mw)

    manager._sync_linked_x_ranges()  # 不该抛

    assert mw._pending_xlink_sync is False
    assert alive.plot_widget.view_box.viewRange()[0] == pytest.approx((0.0, 5.0))


def test_sync_returns_quietly_when_mainwindow_wrapper_is_gone(qapp):
    """MainWindow 已被 GC：入口必须先经 ``_mw_ref()`` 判空。

    守卫若写成先摸 ``self.mw``（弱引用断开时那个 property 自己就抛
    RuntimeError），本用例就会以 ``LayoutManager: MainWindow has been
    garbage collected`` 失败。
    """
    mw = _Mw([_Container(x_range=(0.0, 10.0))])
    manager = LayoutManager(mw)
    mw._kept = manager
    del mw
    gc.collect()

    manager._sync_linked_x_ranges()  # 不该抛


def test_sync_returns_quietly_when_the_whole_window_is_deleted(qapp):
    """整窗连 container 一起销毁：残留列表无从同步，静默返回即可。"""
    mw = _MwWindow([_Container(x_range=(0.0, 10.0)), _Container(x_range=(0.0, 9.0))])
    manager = _manager(mw)
    _drop(mw, qapp)

    assert _widget_alive(mw) is False
    manager._sync_linked_x_ranges()  # 不该抛


# ---------------------------------------------------------------- 同类路径


def test_splitter_callbacks_skip_a_deleted_window(qapp):
    """两个 splitter 延迟回调同属「singleShot 落地时窗口已没了」这一类。

    ``hasattr(mw, "main_splitter")`` 在残留壳上恒真，真按下去才炸；
    ``_ensure_splitter_ready`` 少了这道守卫还会每 50ms 自我续排。
    """
    mw = _MwWindow()
    manager = _manager(mw)
    _drop(mw, qapp)

    with pytest.raises(RuntimeError):
        mw.main_splitter.sizes()  # 前提：陷阱真实存在

    manager._ensure_splitter_ready()  # 不该抛，也不该再续排
    manager._apply_fixed_splitter_width()  # 不该抛


# ---------------------------------------------------------------- 守卫本体


def test_widget_alive_takes_non_qt_standins():
    """替身（普通 Python 对象）不能被守卫误杀，否则组件测试全成空跑。"""
    assert _widget_alive(object()) is True
    assert _widget_alive(_Mw()) is True
    assert _widget_alive(None) is False
