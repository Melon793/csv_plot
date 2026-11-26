"""右键拖拽扫选 X 区间 component 测试（offscreen）。

覆盖 CustomViewBox.mouseDragEvent 的右键分支：
- 右键单击（无拖动）仍是弹菜单（不经过扫选会话）；
- 右键拖拽过判定线 → X 缩放到扫出的数据区间，Y 范围与 Y autoRange 都不动；
- 判定线固定为平台拖动距离（不随视宽放大）：窄扫掠照样缩放；
- 净位移不足判定线 → 范围不变，补弹右键菜单（判定只看净位移，不看路径）；
- 预览带只在拖拽期间存在、ignoreBounds、任何退出路径都摘干净，
  且未过判定线时是「虚线无填充」候选态、过线才变成 tint 实线；
- XLink 子图上扫选 → 整组一致；
- 反向拖动（右→左）结果相同；
- Esc 取消；X 轴锁定时不接管（沿用父类默认行为）。

事件按 pyqtgraph ``MouseDragEvent`` 的契约手工构造（move/press/last），
直接投给 ViewBox —— offscreen 下走真实 QGraphicsScene 需要额外 hover/grab 铺垫，
而本类只用到 ``scenePos`` / ``buttonDownScenePos`` / ``isStart`` / ``isFinish``。
"""

import pandas as pd
import pytest

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QGraphicsSceneMouseEvent

from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent, MouseDragEvent

from src.ui.widgets.custom_viewbox import SCAN_MIN_PX


# ---------- 事件构造辅助 ----------


def _qscene_event(kind, scene_pos, down_pos, buttons=Qt.MouseButton.RightButton):
    ev = QGraphicsSceneMouseEvent(kind)
    ev.setScenePos(QPointF(scene_pos))
    ev.setScreenPos(QPoint(int(scene_pos.x()), int(scene_pos.y())))
    ev.setButton(Qt.MouseButton.RightButton)
    ev.setButtons(buttons)
    ev.setButtonDownScenePos(Qt.MouseButton.RightButton, QPointF(down_pos))
    ev.setButtonDownScreenPos(
        Qt.MouseButton.RightButton, QPoint(int(down_pos.x()), int(down_pos.y()))
    )
    return ev


def _send(vb, drag, axis=None):
    """投产路径上 GraphicsScene 会先设好 currentItem（pos()/lastPos() 依赖它）。"""
    drag.currentItem = vb
    vb.mouseDragEvent(drag, axis=axis)


def _drag(vb, start_scene, end_scene, axis=None):
    """模拟一次完整右键拖拽：start → end → 松手。"""
    press = _qscene_event(QEvent.Type.GraphicsSceneMousePress, start_scene, start_scene)
    move = _qscene_event(QEvent.Type.GraphicsSceneMouseMove, end_scene, start_scene)
    _send(vb, MouseDragEvent(move, press, None, start=True, finish=False), axis=axis)
    _send(vb, MouseDragEvent(move, press, move, start=False, finish=False), axis=axis)
    release = _qscene_event(
        QEvent.Type.GraphicsSceneMouseRelease,
        end_scene,
        start_scene,
        buttons=Qt.MouseButton.NoButton,
    )
    _send(vb, MouseDragEvent(release, press, move, start=False, finish=True), axis=axis)


def _drag_start_only(vb, start_scene, mid_scene):
    """只发到 start（预览带已建、会话进行中）。"""
    press = _qscene_event(QEvent.Type.GraphicsSceneMousePress, start_scene, start_scene)
    move = _qscene_event(QEvent.Type.GraphicsSceneMouseMove, mid_scene, start_scene)
    _send(vb, MouseDragEvent(move, press, None, start=True, finish=False))


def _start_drag(vb, down_scene, pos_scene):
    _send(
        vb,
        MouseDragEvent(
            _qscene_event(QEvent.Type.GraphicsSceneMouseMove, pos_scene, down_scene),
            _qscene_event(QEvent.Type.GraphicsSceneMousePress, down_scene, down_scene),
            None,
            start=True,
            finish=False,
        ),
    )


def _continue_drag(vb, down_scene, from_scene, pos_scene):
    _send(
        vb,
        MouseDragEvent(
            _qscene_event(QEvent.Type.GraphicsSceneMouseMove, pos_scene, down_scene),
            _qscene_event(QEvent.Type.GraphicsSceneMousePress, down_scene, down_scene),
            _qscene_event(QEvent.Type.GraphicsSceneMouseMove, from_scene, down_scene),
            start=False,
            finish=False,
        ),
    )


def _finish_drag(vb, down_scene, from_scene, pos_scene):
    release = _qscene_event(
        QEvent.Type.GraphicsSceneMouseRelease,
        pos_scene,
        down_scene,
        buttons=Qt.MouseButton.NoButton,
    )
    _send(
        vb,
        MouseDragEvent(
            release,
            _qscene_event(QEvent.Type.GraphicsSceneMousePress, down_scene, down_scene),
            _qscene_event(QEvent.Type.GraphicsSceneMouseMove, from_scene, down_scene),
            start=False,
            finish=True,
        ),
    )


def _right_click(vb, scene_pos):
    """场景在「按下→松开且未拖动」时派发 MouseClickEvent。"""
    release = _qscene_event(
        QEvent.Type.GraphicsSceneMouseRelease,
        scene_pos,
        scene_pos,
        buttons=Qt.MouseButton.NoButton,
    )
    vb.mouseClickEvent(MouseClickEvent(release))


def _x_of(vb, pw, local: QPoint) -> float:
    return vb.mapSceneToView(pw.mapToScene(local)).x()


@pytest.fixture()
def scan_plot(plot_factory, qapp):
    """800x600 已显示的 plot，X 固定 0..100、Y 固定 -10..110。"""
    pw = plot_factory(pd.DataFrame({"a": [float(i) for i in range(100)]}))
    pw.resize(800, 600)
    pw.window().show()
    pw.show()
    qapp.processEvents()
    assert pw.plot_variable("a")
    # getMenu 会查这些成员（真实菜单用例直接走它，不放桩）
    ctx = pw.plot_context
    ctx.is_cursor_enabled = lambda: False
    ctx.cursor_mode = "1 free cursor"
    ctx.cursor_values_hidden = False
    ctx.get_row_height = lambda row: 100
    ctx._plot_col_max_default = 1
    ctx.plot_widgets = []
    pw.view_box.setXRange(0, 100, padding=0)
    pw.view_box.setYRange(-10, 110, padding=0)
    qapp.processEvents()
    return pw


# ---------- 单击 / 误触 ----------


def test_right_click_still_opens_menu(scan_plot, monkeypatch):
    """无拖动的右键单击：仍走 mouseClickEvent → 菜单，不建扫选会话。"""
    pw = scan_plot
    vb = pw.view_box
    opened = []
    monkeypatch.setattr(vb, "raiseContextMenu", lambda ev: opened.append(ev))

    start = pw.mapToScene(QPoint(300, 300))
    _right_click(vb, start)

    assert len(opened) == 1
    assert vb._scan_band is None
    assert vb._scan_active is False


def test_short_drag_keeps_range_and_opens_menu(scan_plot, monkeypatch):
    """拖动只有几个像素：按误触处理 —— 范围不变、补弹菜单、预览带撤掉。"""
    pw = scan_plot
    vb = pw.view_box
    opened = []
    monkeypatch.setattr(vb, "raiseContextMenu", lambda ev: opened.append(ev))
    x_before = list(vb.viewRange()[0])

    start = pw.mapToScene(QPoint(300, 300))
    # 6px：超过 GraphicsScene 的 moveDistance(5) 已算拖动，但不到最小有效宽度
    end = pw.mapToScene(QPoint(300 + SCAN_MIN_PX - 2, 300))
    _drag(vb, start, end)

    assert len(opened) == 1
    assert vb.viewRange()[0] == x_before
    assert vb._scan_band is None


def test_wobble_but_ends_past_line_still_zooms(scan_plot, monkeypatch):
    """判定只看净位移：中途来回摆、最终仍过线 → 照常缩放（保留此口径）。

    预览带的实线填充态与这里的提交条件同源，所以「看到实线就一定会缩」——
    不再有「实线带子却弹菜单」的视觉相悖。
    """
    pw = scan_plot
    vb = pw.view_box
    opened = []
    monkeypatch.setattr(vb, "raiseContextMenu", lambda ev: opened.append(ev))

    down = pw.mapToScene(QPoint(300, 300))
    p1 = pw.mapToScene(QPoint(316, 300))
    p2 = pw.mapToScene(QPoint(308, 300))
    p3 = pw.mapToScene(QPoint(313, 300))
    expected = sorted(
        [
            vb.mapSceneToView(down).x(),
            vb.mapSceneToView(p3).x(),
        ]
    )
    _start_drag(vb, down, p1)
    _continue_drag(vb, down, p1, p2)
    _continue_drag(vb, down, p2, p3)
    _finish_drag(vb, down, p3, p3)

    assert vb.viewRange()[0] == pytest.approx(expected, rel=1e-9, abs=1e-9)
    assert opened == []
    assert vb._scan_band is None


def test_short_drag_opens_real_menu(scan_plot, qapp):
    """不放桩走真实 raiseContextMenu：MouseDragEvent 也能正常弹出自建菜单。

    上面的误触用例只数了一次回调；这条证明 getMenu(MouseDragEvent) /
    addParentContextMenus / popup 整条真实链路可用（offscreen 下 popup 不阻塞）。
    """
    pw = scan_plot
    vb = pw.view_box
    start = pw.mapToScene(QPoint(300, 300))
    end = pw.mapToScene(QPoint(300 + SCAN_MIN_PX - 2, 300))

    _drag(vb, start, end)
    qapp.processEvents()

    assert vb._context_menu is not None and vb._context_menu.isVisible()
    vb._context_menu.hide()
    qapp.processEvents()
    assert vb.viewRange()[0] == pytest.approx([0, 100], abs=1e-9)


@pytest.mark.parametrize("axis", [0, 1])
def test_axis_drag_not_hijacked(scan_plot, axis):
    """轴区拖拽（axis 参数非 None）不进入扫选：不建预览带。"""
    pw = scan_plot
    vb = pw.view_box
    start = pw.mapToScene(QPoint(200, 300))
    end = pw.mapToScene(QPoint(500, 300))

    _drag(vb, start, end, axis=axis)

    assert vb._scan_band is None
    assert vb._scan_active is False


def test_x_locked_keeps_default_behavior(scan_plot):
    """X 轴锁定时不接管右键拖拽（与 D13「锁定时框选只写 Y」同口径）。"""
    pw = scan_plot
    vb = pw.view_box
    pw._x_mouse_locked = True
    vb.setMouseEnabled(x=False, y=True)  # 与 MainWindow.set_x_axis_locked 同态
    x_before = list(vb.viewRange()[0])
    y_before = list(vb.viewRange()[1])
    # 斜拖：父类的 scaleBy 用 screenPos 差分，纯水平拖不会产生 Y 缩放
    start = pw.mapToScene(QPoint(200, 300))
    end = pw.mapToScene(QPoint(500, 200))

    _drag(vb, start, end)

    assert vb._scan_band is None
    assert vb._scan_active is False
    assert vb.viewRange()[0] == x_before  # 锁 X：扫选不得写 X
    assert vb.viewRange()[1] != y_before  # 父类默认行为仍在（Y 缩放接管）


# ---------- 正常扫选 ----------


def test_drag_sets_x_range_exactly_and_keeps_y(scan_plot, monkeypatch):
    """扫选宽度达标：X 精确等于扫出的数据区间，Y 范围逐位不变。"""
    pw = scan_plot
    vb = pw.view_box
    opened = []
    monkeypatch.setattr(vb, "raiseContextMenu", lambda ev: opened.append(ev))

    start_local, end_local = QPoint(200, 300), QPoint(500, 300)
    start, end = pw.mapToScene(start_local), pw.mapToScene(end_local)
    x0, x1 = _x_of(vb, pw, start_local), _x_of(vb, pw, end_local)
    y_before = list(vb.viewRange()[1])

    _drag(vb, start, end)

    assert vb.viewRange()[0] == pytest.approx([x0, x1], rel=1e-9, abs=1e-9)
    assert vb.viewRange()[1] == y_before
    assert opened == []


def test_narrow_sweep_still_zooms(scan_plot, monkeypatch):
    """判定线不随视宽放大：14px 扫掠在旧「视宽 3%」规则下会被判点击，现在必须缩放。"""
    pw = scan_plot
    vb = pw.view_box
    opened = []
    monkeypatch.setattr(vb, "raiseContextMenu", lambda ev: opened.append(ev))
    assert 0.03 * vb.screenGeometry().width() > 14, "前提：旧规则下这段属于误触"

    start_local, end_local = QPoint(300, 300), QPoint(314, 300)
    start, end = pw.mapToScene(start_local), pw.mapToScene(end_local)
    expected = sorted([_x_of(vb, pw, start_local), _x_of(vb, pw, end_local)])

    _drag(vb, start, end)

    assert vb.viewRange()[0] == pytest.approx(expected, rel=1e-9, abs=1e-9)
    assert vb.viewRange()[0][1] - vb.viewRange()[0][0] < 99
    assert opened == []


def test_reverse_drag_same_result(scan_plot):
    """右→左拖动与左→右拖动得到同一区间。"""
    pw = scan_plot
    vb = pw.view_box
    left_local, right_local = QPoint(200, 300), QPoint(520, 300)
    left, right = pw.mapToScene(left_local), pw.mapToScene(right_local)
    expected = sorted([_x_of(vb, pw, left_local), _x_of(vb, pw, right_local)])

    _drag(vb, right, left)

    assert vb.viewRange()[0] == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_drag_keeps_y_autorange_enabled(scan_plot, qapp):
    """扫选只写 X：用户显式开启的 Y autoRange 不得被静默关掉。"""
    pw = scan_plot
    vb = pw.view_box
    pw.auto_y_in_x_range()
    qapp.processEvents()
    assert bool(vb.state["autoRange"][1]) is True

    start = pw.mapToScene(QPoint(200, 300))
    end = pw.mapToScene(QPoint(500, 300))
    _drag(vb, start, end)

    assert bool(vb.state["autoRange"][1]) is True
    assert vb.viewRange()[0][1] - vb.viewRange()[0][0] < 99


def test_cursor_crosshair_during_drag_and_restored(scan_plot, qapp):
    pw = scan_plot
    vb = pw.view_box
    viewport = pw.viewport()
    assert viewport.testAttribute(Qt.WidgetAttribute.WA_SetCursor) is False

    start = pw.mapToScene(QPoint(200, 300))
    mid = pw.mapToScene(QPoint(400, 300))
    _drag_start_only(vb, start, mid)

    assert viewport.cursor().shape() == Qt.CursorShape.CrossCursor
    assert viewport.testAttribute(Qt.WidgetAttribute.WA_SetCursor) is True

    # 收尾：补一个 finish，光标还原成「未显式设置」
    release = _qscene_event(
        QEvent.Type.GraphicsSceneMouseRelease,
        mid,
        start,
        buttons=Qt.MouseButton.NoButton,
    )
    _send(
        vb,
        MouseDragEvent(
            release,
            _qscene_event(QEvent.Type.GraphicsSceneMousePress, start, start),
            release,
            start=False,
            finish=True,
        ),
    )
    assert viewport.testAttribute(Qt.WidgetAttribute.WA_SetCursor) is False


# ---------- 预览带生命周期 ----------


def test_band_removed_after_scan(scan_plot):
    pw = scan_plot
    vb = pw.view_box
    start = pw.mapToScene(QPoint(200, 300))
    mid = pw.mapToScene(QPoint(400, 300))
    _drag_start_only(vb, start, mid)

    band = vb._scan_band
    assert band is not None
    assert band.scene() is not None

    end = pw.mapToScene(QPoint(500, 300))
    release = _qscene_event(
        QEvent.Type.GraphicsSceneMouseRelease,
        end,
        start,
        buttons=Qt.MouseButton.NoButton,
    )
    _send(
        vb,
        MouseDragEvent(
            release,
            _qscene_event(QEvent.Type.GraphicsSceneMousePress, start, start),
            release,
            start=False,
            finish=True,
        ),
    )

    assert vb._scan_band is None
    assert vb._scan_filter_view is None
    assert band.scene() is None
    assert band not in pw.plot_item.items
    assert band not in vb.addedItems


def test_band_two_states_before_and_after_commit_line(scan_plot):
    """未过判定线：只有 accent 虚线、不填充；过线：tint 填充 + 实线。"""
    pw = scan_plot
    vb = pw.view_box
    down = pw.mapToScene(QPoint(300, 300))
    small = pw.mapToScene(QPoint(305, 300))  # < 判定线(平台 10px)
    big = pw.mapToScene(QPoint(360, 300))

    _start_drag(vb, down, small)
    band = vb._scan_band
    assert band is not None
    assert band.brush.style() == Qt.BrushStyle.NoBrush
    assert all(line.pen.style() == Qt.PenStyle.DashLine for line in band.lines)

    _continue_drag(vb, down, small, big)
    assert band.brush.style() != Qt.BrushStyle.NoBrush
    assert all(line.pen.style() == Qt.PenStyle.SolidLine for line in band.lines)

    _finish_drag(vb, down, big, big)
    assert vb._scan_band is None


def test_band_ignores_bounds(scan_plot):
    """预览带必须 ignoreBounds：不参与 auto-range，也不进 addedItems。"""
    pw = scan_plot
    vb = pw.view_box
    start = pw.mapToScene(QPoint(200, 300))
    mid = pw.mapToScene(QPoint(400, 300))

    x_bounds_before = vb.childrenBounds()[0]
    _drag_start_only(vb, start, mid)

    band = vb._scan_band
    assert band is not None
    assert band not in vb.addedItems
    assert band in pw.plot_item.items  # 确实挂在 PlotItem 上（方便统一摘除）
    assert vb.childrenBounds()[0] == pytest.approx(x_bounds_before, abs=1e-9)


def test_exception_during_apply_still_cleans_up(scan_plot, monkeypatch):
    """任何退出路径（含异常）都要摘掉预览带、还原光标。"""
    pw = scan_plot
    vb = pw.view_box

    def boom(*_args):
        raise RuntimeError("apply failed")

    monkeypatch.setattr(pw, "set_xrange_with_link_handling", boom)
    start = pw.mapToScene(QPoint(200, 300))
    end = pw.mapToScene(QPoint(500, 300))

    with pytest.raises(RuntimeError):
        _drag(vb, start, end)

    assert vb._scan_band is None
    assert vb._scan_active is False
    assert pw.viewport().testAttribute(Qt.WidgetAttribute.WA_SetCursor) is False


# ---------- Esc 取消 ----------


def test_escape_cancels_scan(scan_plot, qapp, monkeypatch):
    pw = scan_plot
    vb = pw.view_box
    opened = []
    monkeypatch.setattr(vb, "raiseContextMenu", lambda ev: opened.append(ev))
    x_before = list(vb.viewRange()[0])

    start = pw.mapToScene(QPoint(200, 300))
    mid = pw.mapToScene(QPoint(400, 300))
    _drag_start_only(vb, start, mid)
    band = vb._scan_band
    assert band is not None

    qapp.sendEvent(
        pw, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
    )

    assert vb._scan_band is None
    assert band.scene() is None
    assert pw.viewport().testAttribute(Qt.WidgetAttribute.WA_SetCursor) is False

    # 取消后本次拖拽的 finish 不得再缩范围、也不得补弹菜单
    end = pw.mapToScene(QPoint(520, 300))
    release = _qscene_event(
        QEvent.Type.GraphicsSceneMouseRelease,
        end,
        start,
        buttons=Qt.MouseButton.NoButton,
    )
    _send(
        vb,
        MouseDragEvent(
            release,
            _qscene_event(QEvent.Type.GraphicsSceneMousePress, start, start),
            release,
            start=False,
            finish=True,
        ),
    )
    assert vb.viewRange()[0] == x_before
    assert opened == []

    # 取消不留下残留状态：紧接着的新扫选照常生效
    _drag(vb, start, pw.mapToScene(QPoint(520, 300)))
    assert vb.viewRange()[0] != x_before


# ---------- XLink ----------


def test_xlink_scan_updates_group(plot_factory, qapp):
    """在 XLink 子图上扫选：源视图与兄弟子图 X 范围都变成目标区间，Y 不动。"""
    df = pd.DataFrame({"a": [float(i) for i in range(100)]})
    master = plot_factory(df)
    child = plot_factory(df)
    for pw in (master, child):
        pw.resize(800, 300)
        pw.show()
        assert pw.plot_variable("a")
        pw.view_box.setXRange(0, 100, padding=0)
        pw.view_box.setYRange(-10, 110, padding=0)
    qapp.processEvents()
    child.view_box.setXLink(master.view_box)
    qapp.processEvents()

    child_y_before = list(child.view_box.viewRange()[1])
    master_y_before = list(master.view_box.viewRange()[1])

    start_local, end_local = QPoint(150, 150), QPoint(500, 150)
    start, end = child.mapToScene(start_local), child.mapToScene(end_local)
    vb = child.view_box
    expected = sorted([_x_of(vb, child, start_local), _x_of(vb, child, end_local)])

    _drag(vb, start, end)
    qapp.processEvents()

    assert master.view_box.viewRange()[0] == pytest.approx(
        expected, rel=1e-9, abs=1e-9
    )
    assert child.view_box.viewRange()[0] == pytest.approx(
        expected, rel=1e-9, abs=1e-9
    )
    assert child.view_box.viewRange()[1] == child_y_before
    assert master.view_box.viewRange()[1] == master_y_before