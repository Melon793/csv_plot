"""曲线长按抬框搬运 e2e：真实 300ms 计时、曲线来源 drop、源拒绝与复制。

offscreen 下 ``QDrag.exec`` 不进入嵌套循环，这里分两层验收：
- 真实计时器 + 副框：QTest.mousePress 后等待，验证 300ms 后抬起与松手无操作；
- drop 语义：手工登记 ``origin="curve"`` 注册表并构造 QDropEvent，验证
  单条曲线走 quiet 批量协议（不弹模态）、移动/复制与目标选择清空。
"""

import pytest

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QDragMoveEvent, QDropEvent
from PySide6.QtWidgets import QMessageBox
from PySide6.QtTest import QTest

from src.core.config import CURVE_LIFT_LONG_PRESS_MS
from src.ui.drag_drop import (
    build_legend_var_mimedata,
    clear_active_legend_drag,
    set_active_legend_drag,
)
from tests.fixtures.waits import settle

CTRL = Qt.KeyboardModifier.ControlModifier
ALT = Qt.KeyboardModifier.AltModifier

_keep_alive: list = []


@pytest.fixture()
def silent_dialogs(monkeypatch):
    calls = []

    def fake_question(parent, title, text, *args, **kwargs):
        calls.append((title, text))
        return QMessageBox.StandardButton.No

    def fake_information(parent, title, text, *args, **kwargs):
        calls.append((title, text))
        return QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QMessageBox, "question", staticmethod(fake_question))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(fake_information))
    return calls


def _curve_pos(pw, name, frac=0.5):
    ci = pw.curves[name]
    i = min(len(ci.x_data) - 1, int(len(ci.x_data) * frac))
    return pw.mapFromScene(
        pw.view_box.mapViewToScene(QPointF(float(ci.x_data[i]), float(ci.y_data[i])))
    )


def _drop_curve(dst, src, names, *, mods=Qt.KeyboardModifier.NoModifier):
    """登记曲线来源拖拽 + 构造 drop（曲线来源恒走 quiet 批量协议）。"""
    mime = build_legend_var_mimedata(list(names), src)
    _keep_alive.append(mime)
    set_active_legend_drag(src, list(names), batch=len(names) > 1, origin="curve")
    try:
        ev = QDropEvent(
            QPointF(10.0, 10.0),
            Qt.DropAction.CopyAction | Qt.DropAction.MoveAction,
            mime,
            Qt.MouseButton.LeftButton,
            mods,
        )
        dst.dropEvent(ev)
    finally:
        clear_active_legend_drag()


@pytest.mark.full_matrix
def test_real_long_press_requires_selection(
    two_plots, qtbot, qapp
):
    _mw, _page, left, _right = two_plots
    pos = _curve_pos(left, "speed")

    # 未选中：长按不抬框（与慢速拖动平移的按下-静止阶段无法区分）
    QTest.mousePress(left.viewport(), Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, pos)
    assert left._lift_timer.isActive()
    qtbot.wait(CURVE_LIFT_LONG_PRESS_MS + 300)
    qapp.processEvents()
    assert left._lift_active is False
    assert left._lift_card is None or not left._lift_card.isVisible()

    QTest.mouseRelease(left.viewport(), Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier, pos)
    qapp.processEvents()
    assert left._legend_selection == set(), "裸单击不得选中"

    # 已选中：真实 300ms 抬起副框，原地松手不搬运、不改选择
    left.legend_label._toggle_legend_selection("speed")
    QTest.mousePress(left.viewport(), Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, pos)
    qtbot.waitUntil(lambda: left._lift_active, timeout=CURVE_LIFT_LONG_PRESS_MS + 2000)
    settle()

    assert left._lift_names == ["speed"]
    assert left._lift_card is not None and left._lift_card.isVisible()
    assert left._legend_selection == {"speed"}

    QTest.mouseRelease(left.viewport(), Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier, pos)
    qapp.processEvents()

    assert left._lift_active is False
    assert left._legend_selection == {"speed"}, "原地松手不得改选择"
    assert left.curves["speed"].curve is not None, "松手不得搬运"


@pytest.mark.full_matrix
def test_ctrl_long_press_lifts_unselected_curve(two_plots, qtbot, qapp):
    """⌘+长按 = 抓取许可：未选中的曲线也能直接出副框，且不改选择集。"""
    _mw, _page, left, _right = two_plots
    pos = _curve_pos(left, "speed")
    assert left._legend_selection == set()

    QTest.mousePress(left.viewport(), Qt.MouseButton.LeftButton, CTRL, pos)
    qtbot.waitUntil(lambda: left._lift_active, timeout=CURVE_LIFT_LONG_PRESS_MS + 2000)
    settle()

    assert left._lift_names == ["speed"]
    assert left._legend_selection == set(), "预览阶段不得预选"

    QTest.mouseRelease(left.viewport(), Qt.MouseButton.LeftButton, CTRL, pos)
    qapp.processEvents()

    assert left._lift_active is False
    assert left._legend_selection == {"speed"}, "未拖动则回落为 ⌘ 单击选择"


@pytest.mark.full_matrix
def test_curve_origin_move_is_quiet_and_clears_target_selection(
    two_plots, silent_dialogs, qapp
):
    _mw, _page, left, right = two_plots
    right.legend_label._toggle_legend_selection("rpm")
    assert right._legend_selection == {"rpm"}

    _drop_curve(right, left, ["speed"])
    settle()

    assert "speed" in right.curves, "目标应新增被搬运曲线"
    assert "speed" not in left.curves, "无修饰键=移动"
    assert right._legend_selection == set(), "落点目标旧选择须清空"
    assert silent_dialogs == [], "曲线来源不得弹重复确认/提示"


@pytest.mark.full_matrix
def test_curve_origin_copy_with_alt_keeps_source(two_plots, silent_dialogs):
    _mw, _page, left, right = two_plots

    _drop_curve(right, left, ["speed"], mods=ALT)
    settle()

    assert "speed" in left.curves, "Alt=复制，源保留"
    assert "speed" in right.curves
    assert silent_dialogs == []


@pytest.mark.full_matrix
def test_drop_back_to_source_is_rejected(two_plots):
    _mw, _page, left, _right = two_plots
    mime = build_legend_var_mimedata(["speed"], left)
    _keep_alive.append(mime)
    set_active_legend_drag(left, ["speed"], batch=False, origin="curve")
    try:
        ev = QDragMoveEvent(
            (left.viewport().rect().center()),
            Qt.DropAction.CopyAction | Qt.DropAction.MoveAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        left.dragMoveEvent(ev)
        assert not ev.isAccepted(), "曲线来源拖回源 plot 必须拒绝"
    finally:
        clear_active_legend_drag()
