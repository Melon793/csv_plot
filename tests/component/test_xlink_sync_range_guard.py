"""P1-19：XLink 程序化同步必须在 ``_is_syncing_range`` 保护下改范围。

`_sync_linked_x_ranges` 由 50ms 防抖的 ``QTimer.singleShot`` 触发，会对每个子图做
「临时 unlink → setXRange → 重新 link」；健康检查分支还会直接 `setXLink(master)`。
pyqtgraph 的 ``linkView`` 末尾会用**子图自己的** ``sigRangeChanged`` 回报新范围
（实测 unlink→setXRange→relink 各发一次），而 ``event_handler._on_range_changed``
正是挂在 ``view_box.sigRangeChanged`` 上：没有守卫时，一次纯程序化的窗口 resize
同步就能让子图误置 ``_is_interacting``，并在防抖窗口内把真正的用户交互源判成级联
结果（交互所有权反转）。

断言方式是「在信号落地的那一刻读标志位」，而不是事后看结果 —— 事后标志位必然已被
复原，看不出当时有没有保护。

宽度不等是制造「漂移」的手段：linked 子图按像素比例换算范围（master 0→10 在两倍
宽的子图上得到 0→23.7），这正是 ``_sync_linked_x_ranges`` 要修的目标形态；等宽时
relink 后范围与 master 完全相等，可用来单独验健康检查分支。
"""

from __future__ import annotations

import pyqtgraph as pg
import pytest
from PySide6.QtWidgets import QWidget

from src.ui import layout_manager
from src.ui.layout_manager import LayoutManager


class _FakePlotWidget:
    """带真实 ViewBox 的 pw 替身（裸 ViewBox 不参与级联，必须挂在 PlotWidget 上）。"""

    def __init__(self, name: str, x_range, size):
        self.view = pg.PlotWidget()
        self.view.resize(*size)
        # 必须真 show：隐藏时 ViewBox 拿不到几何，等宽假象会让「漂移」前提失效
        self.view.show()
        self.view_box = self.view.plotItem.vb
        self.y_name = name
        self.view_box.enableAutoRange(x=False)
        self.view_box.setXRange(*x_range, padding=0)
        self.emits: list[bool] = []
        self.view_box.sigRangeChanged.connect(self._record)

    def _record(self, _vb, _range):
        # 关键：读的是信号发出那一刻的标志位
        self.emits.append(getattr(self, "_is_syncing_range", False))


class _Container(QWidget):
    def __init__(self, name: str, x_range=(0.0, 10.0), size=(600, 300)):
        super().__init__()
        self.plot_widget = _FakePlotWidget(name, x_range, size)


class _Mw:
    _pending_xlink_sync = True

    def __init__(self, containers=()):
        self.plot_widgets = list(containers)


def _manager(mw) -> LayoutManager:
    manager = LayoutManager(mw)
    mw._kept = manager  # 反向撑住：LayoutManager 只被闭包引用会立刻 GC
    return manager


def _x_range(view_box):
    lo, hi = view_box.viewRange()[0]
    return float(lo), float(hi)


@pytest.fixture
def guarded_pws(monkeypatch):
    """记录每次进入守卫的 plot_widget，用于区分「跳过的块不该被无谓屏蔽」。"""
    entered: list = []
    real_guard = layout_manager._sync_guard

    def spy(*plot_widgets):
        entered.extend(plot_widgets)
        return real_guard(*plot_widgets)

    monkeypatch.setattr(layout_manager, "_sync_guard", spy)
    return entered


@pytest.fixture
def drifted_pair(qapp):
    """master 窄、child 宽：link 之后 child 必然偏离 master 范围。"""
    master = _Container("master", (0.0, 10.0), (400, 300))
    child = _Container("child", (0.0, 10.0), (900, 300))
    child.plot_widget.view_box.setXLink(master.plot_widget.view_box)
    assert _x_range(child.plot_widget.view_box) != _x_range(
        master.plot_widget.view_box
    ), "前提：等宽就不会漂移，本用例将退化为空跑"
    child.plot_widget.emits.clear()
    mw = _Mw([master, child])
    manager = _manager(mw)
    yield manager, master, child
    for container in (master, child):
        container.plot_widget.view.deleteLater()


@pytest.fixture
def equal_pair(qapp):
    """等宽：relink 后子图范围与 master 严格相等，可单独验健康检查分支。"""
    master = _Container("master", (0.0, 10.0), (600, 300))
    child = _Container("child", (100.0, 200.0), (600, 300))
    mw = _Mw([master, child])
    manager = _manager(mw)
    yield manager, master, child
    for container in (master, child):
        container.plot_widget.view.deleteLater()


class TestSyncLoopGuarded:
    def test_range_change_emissions_all_under_guard(self, drifted_pair, guarded_pws):
        manager, _master, child = drifted_pair

        manager._sync_linked_x_ranges()

        assert child.plot_widget.emits, "子图范围确实被改过（否则本用例是空跑）"
        assert all(child.plot_widget.emits)
        assert child.plot_widget._is_syncing_range is False

    def test_link_restore_emission_also_guarded(self, drifted_pair):
        # 重新 link 后子图又按像素比例回报一次，这最后一次同样不能漏
        manager, _master, child = drifted_pair

        manager._sync_linked_x_ranges()

        assert len(child.plot_widget.emits) >= 2, "setXRange 与 relink 应各回报一次"
        assert all(child.plot_widget.emits)

    def test_guard_scoped_to_the_plot_being_synced(self, drifted_pair, guarded_pws):
        manager, master, child = drifted_pair

        manager._sync_linked_x_ranges()

        assert guarded_pws == [child.plot_widget]
        assert getattr(master.plot_widget, "_is_syncing_range", False) is False


class _BarePw:
    """只用来验守卫本身：`object()` 没有 `__dict__`，塞不进标志位。"""


class TestSyncGuardItself:
    def test_restores_previous_value_instead_of_forcing_false(self):
        pw = _BarePw()
        pw._is_syncing_range = "outer"

        with layout_manager._sync_guard(pw):
            assert pw._is_syncing_range is True

        assert pw._is_syncing_range == "outer"

    def test_nested_guard_does_not_strip_outer_protection(self):
        pw = _BarePw()

        with layout_manager._sync_guard(pw):
            with layout_manager._sync_guard(pw):
                pass
            assert pw._is_syncing_range is True

        assert pw._is_syncing_range is False

    def test_exception_inside_guard_still_restores(self):
        pw = _BarePw()

        with pytest.raises(RuntimeError):
            with layout_manager._sync_guard(pw):
                raise RuntimeError("boom")

        assert pw._is_syncing_range is False


class TestHealthCheckGuarded:
    def test_relink_of_lost_xlink_is_guarded(self, equal_pair, guarded_pws):
        manager, master, child = equal_pair
        child.plot_widget.view_box.setXLink(None)
        child.show()  # 健康检查只看 isVisible() 的容器
        try:
            assert child.isVisible() is True
            child.plot_widget.emits.clear()
            guarded_pws.clear()

            manager._sync_linked_x_ranges()

            assert child.plot_widget.view_box.linkedView(0) is (
                master.plot_widget.view_box
            )
            assert all(child.plot_widget.emits), "relink 必须回报一次范围"
            assert child.plot_widget.emits == [True]
            assert guarded_pws == [child.plot_widget]
            assert child.plot_widget._is_syncing_range is False
        finally:
            child.hide()


class TestSkippedPlotsStayUnguarded:
    def test_matching_plot_is_never_entered_into_guard(self, equal_pair, guarded_pws):
        manager, master, child = equal_pair
        child.plot_widget.view_box.setXLink(None)
        child.plot_widget.view_box.setXRange(
            *_x_range(master.plot_widget.view_box), padding=0
        )
        child.plot_widget.emits.clear()
        guarded_pws.clear()

        manager._sync_linked_x_ranges()

        assert guarded_pws == []
        assert child.plot_widget.emits == []
