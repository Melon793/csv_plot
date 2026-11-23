"""X 轴锁定 component 测试：D14 AxisDialog X 档禁用（main 收口版）。

菜单/快捷键入口已按拍板撤销（L2 唯一手动入口=抽屉 chip，见
test_x_lock_drawer.py）；本文件只保留 D14 对话框语义。
"""

import pandas as pd
import pytest

from src.ui.dialogs.axis import AxisDialog


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
            isinstance(w, QLabel) and w.text() == "X 轴可见范围已锁定"
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