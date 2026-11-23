"""X 轴锁定 C2 component 测试：L1 菜单接线与 D14 AxisDialog X 档禁用。

菜单 triggered → request_toggle_x_lock → event_handler 槽 →
window().set_x_axis_locked 的整条信号链走真实实现，宿主只替身播报落点。
"""

import pandas as pd
import pytest

from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QWidget

from src.ui.dialogs.axis import AxisDialog
from src.ui.widgets.custom_viewbox import ZH_LOCK_X
from tests.component.conftest import FakeLayoutManager


class _FakeMenuEvent:
    def __init__(self, x=0.0, y=0.0):
        self._pos = QPointF(x, y)

    def scenePos(self):
        return self._pos


class _LockHost(QWidget):
    """记录 set_x_axis_locked 调用并真的落 plot 标志（模拟广播的最小闭环）"""

    def __init__(self, pw):
        super().__init__()
        self.layout_manager = FakeLayoutManager()
        self.calls = []
        self._pw = pw

    def set_x_axis_locked(self, locked, reason=""):
        self.calls.append((bool(locked), reason))
        self._pw._x_mouse_locked = bool(locked)


@pytest.fixture()
def menu_host(plot_factory, qapp):
    pw = plot_factory(pd.DataFrame({"a": [float(i) for i in range(10)]}))
    host = _LockHost(pw)
    pw.setParent(host)
    ctx = pw.plot_context
    ctx.is_cursor_enabled = lambda: False
    ctx.cursor_mode = "1 free cursor"
    ctx.cursor_values_hidden = False
    ctx.get_row_height = lambda row: 100
    ctx._plot_col_max_default = 1
    ctx.plot_widgets = []
    return pw, host


def _lock_action(vb):
    menu = vb.getMenu(_FakeMenuEvent())
    return next(a for a in menu.actions() if a.text() == ZH_LOCK_X)


def test_menu_trigger_toggles_via_signal_chain(menu_host):
    pw, host = menu_host
    act = _lock_action(pw.view_box)
    assert not act.isChecked()

    act.trigger()
    assert host.calls == [(True, "")]
    assert pw._x_mouse_locked is True

    # 再开一次菜单：checked 已随标志同步；再点=解锁
    act = _lock_action(pw.view_box)
    assert act.isChecked()
    act.trigger()
    assert host.calls == [(True, ""), (False, "")]
    assert pw._x_mouse_locked is False


def test_menu_trigger_without_host_setter_is_quiet(plot_factory):
    """替身宿主（FakeHost）没有 set_x_axis_locked：槽静默失效，不抛异常。"""
    pw = plot_factory()
    ctx = pw.plot_context
    ctx.is_cursor_enabled = lambda: False
    ctx.cursor_mode = "1 free cursor"
    ctx.cursor_values_hidden = False
    ctx.get_row_height = lambda row: 100
    ctx._plot_col_max_default = 1
    ctx.plot_widgets = []
    act = _lock_action(pw.view_box)
    act.trigger()  # 不得抛 AttributeError
    assert pw._x_mouse_locked is False


# ---------- D14：AxisDialog 锁定 X 档禁用、Y 档保留 ----------

@pytest.fixture()
def dialog_plot(plot_factory, qapp):
    pw = plot_factory(pd.DataFrame({"a": [float(i) for i in range(10)]}))
    pw.resize(600, 400)
    pw.window().show()
    pw.show()
    assert pw.plot_variable("a")
    pw.view_box.setXRange(2, 8, padding=0)
    pw.view_box.setYRange(-1, 11, padding=0)
    qapp.processEvents()
    return pw


def test_axis_dialog_x_locked_disables_inputs(dialog_plot, qapp):
    pw = dialog_plot
    pw._x_mouse_locked = True
    before_x = list(pw.view_box.viewRange()[0])

    dlg = AxisDialog(pw.axis_x, pw.view_box, "X", pw)
    try:
        assert not dlg.min_input.isEnabled()
        assert not dlg.max_input.isEnabled()
        # 行内小注存在
        from PySide6.QtWidgets import QLabel
        assert any(
            isinstance(w, QLabel) and w.text() == "X 轴已锁定"
            for w in dlg.findChildren(QLabel)
        )
        dlg.apply_changes()  # 确定=关闭，不写 X
        qapp.processEvents()
        assert list(pw.view_box.viewRange()[0]) == before_x
    finally:
        dlg.deleteLater()


def test_axis_dialog_y_still_editable_when_x_locked(dialog_plot, qapp):
    pw = dialog_plot
    pw._x_mouse_locked = True

    dlg = AxisDialog(pw.axis_y, pw.view_box, "Y", pw)
    try:
        assert dlg.min_input.isEnabled() and dlg.max_input.isEnabled()
        dlg.min_input.setText("0")
        dlg.max_input.setText("5")
        dlg.apply_changes()
        qapp.processEvents()
        y = pw.view_box.viewRange()[1]
        assert y[0] <= 0 and y[1] >= 5, "Y 档必须仍可写（D14 只禁 X）"
    finally:
        dlg.deleteLater()


def test_axis_dialog_x_unlocked_unchanged(dialog_plot, qapp):
    pw = dialog_plot
    dlg = AxisDialog(pw.axis_x, pw.view_box, "X", pw)
    try:
        assert dlg.min_input.isEnabled() and dlg.max_input.isEnabled()
        dlg.min_input.setText("3")
        dlg.max_input.setText("6")
        dlg.apply_changes()
        qapp.processEvents()
        x = pw.view_box.viewRange()[0]
        assert x[0] <= 3 and x[1] >= 6
    finally:
        dlg.deleteLater()