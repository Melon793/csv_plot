"""e2e：X 轴锁定交互流（C3）——菜单/快捷键两入口锁定后的真实事件序列。

冒烟级：合成滚轮/左拖/Shift 框选投给真实主窗口的子图，断言锁定态 X 冻结、
Y 仍可调，解锁后恢复。拖拽数学与菜单可见性的深断言在 component 层
（test_x_lock_core.py / test_x_lock_entries.py）。
"""

import pytest

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QDropEvent, QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication

from src.ui.drag_drop import build_var_mimedata

_keep_alive = []


def _plot_ready(mw, qtbot, qapp):
    pw = mw.plot_widgets[0].plot_widget
    mime = build_var_mimedata(["speed"])
    _keep_alive.append(mime)
    pw.dropEvent(QDropEvent(
        QPointF(10.0, 10.0),
        Qt.DropAction.CopyAction | Qt.DropAction.MoveAction,
        mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    ))
    qtbot.waitUntil(
        lambda: pw.view_box.sceneBoundingRect().height() > 100, timeout=5000
    )
    pw.view_box.setXRange(10, 40, padding=0)
    pw.view_box.setYRange(-5, 55, padding=0)
    qapp.processEvents()
    return pw


def _freeze_ranges(pw, qapp):
    """落一个稳定的已知范围：drop 后的排队刷新会重开 autoRange，
    捕获前必须显式关掉两轴 autoRange 再写范围，否则平移被快照回收。
    Y 取 limits 内的窗口（auto_range 留下的 yLimits 会钳制 setYRange，
    写超界值会被夹回数据域，平移顶到边界就测不出位移）。"""
    vb = pw.view_box
    vb.enableAutoRange(x=False, y=False)
    vb.setXRange(10, 40, padding=0)
    vb.setYRange(5, 35, padding=0)
    qapp.processEvents()


def _wheel(pw, delta):
    pos = QPointF(pw.width() // 2, pw.height() // 2)
    ev = QWheelEvent(
        pos, QPointF(pw.mapToGlobal(pos.toPoint())),
        QPoint(0, 0), QPoint(0, delta),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase, False,
    )
    pw.wheelEvent(ev)


def _drag(pw, start: QPoint, end: QPoint, button, modifiers=Qt.KeyboardModifier.NoModifier):
    """合成 press→move→release 投给 viewport（QGraphicsView 转场景）"""
    vp = pw.viewport()
    for etype, pos, buttons in (
        (QEvent.Type.MouseButtonPress, start, button),
        (QEvent.Type.MouseMove, end, button),
        (QEvent.Type.MouseButtonRelease, end, Qt.MouseButton.NoButton),
    ):
        ev = QMouseEvent(
            etype, QPointF(pos), QPointF(vp.mapToGlobal(pos)),
            button if etype != QEvent.Type.MouseMove else Qt.MouseButton.NoButton,
            buttons, modifiers,
        )
        QApplication.sendEvent(vp, ev)


class _FakeMenuEvent:
    def scenePos(self):
        return QPointF(0.0, 0.0)


def _menu_lock(pw):
    menu = pw.view_box.getMenu(_FakeMenuEvent())
    next(a for a in menu.actions() if a.text() == "锁定X轴").trigger()


def test_menu_lock_wheel_noop_shortcut_unlock_restores(loaded_window, qtbot, qapp):
    mw = loaded_window
    pw = _plot_ready(mw, qtbot, qapp)
    before = [list(a) for a in pw.view_box.viewRange()]

    _menu_lock(pw)  # 入口 1：菜单
    assert mw._x_axis_locked is True
    _wheel(pw, 120)
    _wheel(pw, -120)
    qapp.processEvents()
    assert [list(a) for a in pw.view_box.viewRange()] == before, "D1：滚轮 no-op"

    mw._on_x_lock_shortcut()  # 入口 2：快捷键解锁
    assert mw._x_axis_locked is False
    _wheel(pw, 120)
    qapp.processEvents()
    x = pw.view_box.viewRange()[0]
    assert x[1] - x[0] < before[0][1] - before[0][0], "解锁后滚轮应恢复缩放"


def test_shortcut_lock_left_drag_pans_y_only(loaded_window, qtbot, qapp):
    mw = loaded_window
    pw = _plot_ready(mw, qtbot, qapp)
    mw._on_x_lock_shortcut()
    _freeze_ranges(pw, qapp)
    x_before = list(pw.view_box.viewRange()[0])
    y_before = list(pw.view_box.viewRange()[1])

    center = QPoint(pw.width() // 2, pw.height() // 2)
    _drag(pw, center, QPoint(center.x(), center.y() - 80), Qt.MouseButton.LeftButton)
    qapp.processEvents()

    assert list(pw.view_box.viewRange()[0]) == x_before, "锁定后左拖不得动 X"
    assert list(pw.view_box.viewRange()[1]) != y_before, "Y 平移必须保留"


def test_shortcut_lock_box_zoom_y_only(loaded_window, qtbot, qapp):
    mw = loaded_window
    pw = _plot_ready(mw, qtbot, qapp)
    mw._on_x_lock_shortcut()
    x_before = list(pw.view_box.viewRange()[0])

    shift = Qt.KeyboardModifier.ShiftModifier
    pa, pb = QPoint(120, 120), QPoint(360, 320)
    vb = pw.view_box
    v1 = vb.mapSceneToView(pw.mapToScene(pa))
    v2 = vb.mapSceneToView(pw.mapToScene(pb))
    y0, y1 = sorted((v1.y(), v2.y()))
    dy = y1 - y0

    _drag(pw, pa, pb, Qt.MouseButton.LeftButton, shift)
    qapp.processEvents()

    assert list(pw.view_box.viewRange()[0]) == x_before, "D13：框选不得写 X"
    y = pw.view_box.viewRange()[1]
    assert y[0] == pytest.approx(y0 - 0.1 * dy, rel=1e-5)
    assert y[1] == pytest.approx(y1 + 0.1 * dy, rel=1e-5)


def test_unlock_restores_left_drag_x_pan(loaded_window, qtbot, qapp):
    mw = loaded_window
    pw = _plot_ready(mw, qtbot, qapp)
    mw._on_x_lock_shortcut()
    mw._on_x_lock_shortcut()  # 锁→解
    assert mw._x_axis_locked is False
    x_before = list(pw.view_box.viewRange()[0])

    center = QPoint(pw.width() // 2, pw.height() // 2)
    _drag(pw, center, QPoint(center.x() - 80, center.y()), Qt.MouseButton.LeftButton)
    qapp.processEvents()

    assert list(pw.view_box.viewRange()[0]) != x_before, "解锁后 X 平移应恢复"