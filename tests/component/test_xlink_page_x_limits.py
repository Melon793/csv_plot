"""``apply_page_x_limits``：把子图 xLimits 统一到全局 oracle。

背景：pyqtgraph 的 ``linkedViewChanged`` 对子图调用 ``setXRange`` 时会先被
**子图自身 xLimits** 钳制。若某个子图的 limits 偏窄/偏移，XLink 看起来就
“失效”了（同宽度、整体平移，链路并未断开）。本组用例锁住修复后的行为。
"""

from __future__ import annotations

import pyqtgraph as pg
import pytest
from PySide6.QtWidgets import QWidget

from src.ui import layout_manager
from src.ui.layout_manager import apply_page_x_limits
from src.ui.widgets.plot_page import PlotPage

ORACLE = (-750.0, 15750.0, -750.0, 15750.0)

_CREATED = []


@pytest.fixture(autouse=True)
def _cleanup_widgets():
    yield
    while _CREATED:
        _CREATED.pop().deleteLater()


class _FakeAxis:
    def __init__(self, oracle=ORACLE, min_x_range=1.0):
        self._oracle = oracle
        self._min_x_range = min_x_range
        self.fail = False
        self.calls = 0
        self.pw = None

    def _global_x_limits(self):
        return self._oracle

    def _get_min_x_range_value(self):
        return self._min_x_range

    def _set_x_limits_with_min_range(self, x_min, x_max):
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom")
        self.pw.plot_item.setLimits(
            xMin=x_min, xMax=x_max, minXRange=self._min_x_range
        )


class _Pw:
    def __init__(self, limits=None, oracle=ORACLE, min_x_range=1.0):
        self.view = pg.PlotWidget()
        self.view.resize(400, 300)
        self.view.show()
        _CREATED.append(self.view)
        self.plot_item = self.view.plotItem
        self.view_box = self.plot_item.vb
        if limits is not None:
            self.view_box.setLimits(xMin=limits[0], xMax=limits[1])
        self.y_name = "p"
        self._is_syncing_range = False
        self._is_being_destroyed = False
        self._axis_manager = None
        if oracle is not None:
            self._axis_manager = _FakeAxis(oracle, min_x_range)
            self._axis_manager.pw = self


class _Container(QWidget):
    def __init__(self, pw):
        super().__init__()
        self.plot_widget = pw


def _grid(*containers):
    grid = type("G", (), {})()
    grid.plot_widgets = list(containers)
    return grid


def _x_limits(pw):
    return list(pw.view_box.state["limits"]["xLimits"])


def _pair(master_limits=(-750.0, 15750.0), child_limits=(-6000.0, 10000.0)):
    master = _Pw(master_limits)
    child = _Pw(child_limits)
    child.view_box.setXLink(master.view_box)
    return master, child


def test_narrow_child_limits_are_aligned_and_link_kept(qapp):
    master, child = _pair()

    changed = apply_page_x_limits(_grid(_Container(master), _Container(child)))

    assert changed == 1
    assert _x_limits(child) == [-750.0, 15750.0]
    assert child.view_box.linkedView(0) is master.view_box


def test_second_call_is_idempotent(qapp):
    master, child = _pair()
    grid = _grid(_Container(master), _Container(child))

    assert apply_page_x_limits(grid) == 1
    assert apply_page_x_limits(grid) == 0


def test_guard_covers_all_live_pws_when_any_change(monkeypatch, qapp):
    entered = []
    real_guard = layout_manager._sync_guard

    def spy(*plot_widgets):
        entered.extend(plot_widgets)
        return real_guard(*plot_widgets)

    monkeypatch.setattr(layout_manager, "_sync_guard", spy)
    master, child = _pair()

    apply_page_x_limits(_grid(_Container(master), _Container(child)))

    assert set(entered) == {master, child}


def test_master_sentinel_limits_follow_page_oracle(qapp):
    master = _Pw(None, oracle=None)  # 默认 ±1e307 哨兵，且自身无 oracle
    child = _Pw((-6000.0, 10000.0))
    child.view_box.setXLink(master.view_box)

    changed = apply_page_x_limits(_grid(_Container(master), _Container(child)))

    assert changed >= 1
    assert _x_limits(master) == [-750.0, 15750.0]
    assert _x_limits(child) == [-750.0, 15750.0]


def test_target_expands_with_min_x_range_and_stays_idempotent(qapp):
    # 数据域只有 10 宽，但 minXRange=50：写入时会被对称扩宽，比较必须同规则
    oracle = (0.0, 10.0, 0.0, 10.0)
    master = _Pw((0.0, 10.0), oracle=oracle, min_x_range=50.0)
    child = _Pw((-6000.0, 10000.0), oracle=oracle, min_x_range=50.0)
    child.view_box.setXLink(master.view_box)
    grid = _grid(_Container(master), _Container(child))

    assert apply_page_x_limits(grid) >= 1
    assert _x_limits(child) == [-20.0, 30.0]
    assert apply_page_x_limits(grid) == 0


def test_write_failure_restores_link_and_does_not_raise(qapp):
    master, child = _pair()
    child._axis_manager.fail = True

    apply_page_x_limits(_grid(_Container(master), _Container(child)))  # 不应抛出

    assert child.view_box.linkedView(0) is master.view_box


class _PageStub:
    pass


def test_pending_sync_defers_while_interacting(qapp):
    pw = _Pw((-750.0, 15750.0))
    pw._is_interacting = True
    page = _PageStub()
    page.epoch = 7
    page.widget = QWidget()
    page._pending_xlink_sync = True
    page.plot_widgets = [_Container(pw)]
    page._has_active_interaction = lambda: PlotPage._has_active_interaction(page)
    calls = []
    page.sync_xlink = lambda: calls.append("sync")
    page.schedule_xlink_sync = lambda: calls.append("schedule")

    assert PlotPage._has_active_interaction(page) is True
    PlotPage._run_pending_sync(page, 7)

    assert calls == ["schedule"], "交互中应只重排、不重写范围"
    assert page._pending_xlink_sync is False
