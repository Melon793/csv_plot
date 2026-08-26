"""ViewBox 交互 component 测试（offscreen）：滚轮缩放 / XLink 同步 / 框选缩放 / 双击中键清除。

模拟 help.md 4️⃣「交互操作」中的用户动作：
- 滚轮缩放 X 轴（以鼠标位置为中心，factor=FACTOR_SCROLL_ZOOM）
- Ctrl/Shift 修饰键下滚轮不触发 X 轴缩放
- XLink：一个子图缩放，联动子图 X 范围同步
- Shift+左键框选放大（含 10% margin、过小框忽略）
- 双击中键清除单个图表

事件一律基于控件自身坐标构造（不依赖绝对屏幕坐标），
QWheelEvent/QMouseEvent 手工构造后直接调用对应事件方法。
"""

import pandas as pd
import pytest

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent

from src.core.config import FACTOR_SCROLL_ZOOM


# ---------- 事件构造辅助 ----------

def _wheel_event(pw, local_pos: QPoint, delta_y: int, modifiers=Qt.KeyboardModifier.NoModifier) -> QWheelEvent:
    """构造一个落在 pw 指定局部坐标的滚轮事件（每齿 120）"""
    pos = QPointF(local_pos)
    global_pos = QPointF(pw.mapToGlobal(local_pos))
    return QWheelEvent(
        pos,
        global_pos,
        QPoint(0, 0),             # pixelDelta
        QPoint(0, delta_y),       # angleDelta（>0 向前=放大）
        Qt.MouseButton.NoButton,
        modifiers,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )


def _mouse_event(event_type, pw, local_pos: QPoint, button, buttons, modifiers) -> QMouseEvent:
    pos = QPointF(local_pos)
    global_pos = QPointF(pw.mapToGlobal(local_pos))
    return QMouseEvent(event_type, pos, global_pos, button, buttons, modifiers)


@pytest.fixture()
def shown_plot(plot_factory, qapp):
    """显示一个带 100 行数据的 plot（几何尺寸生效，坐标映射可用）

    注意：宿主窗口必须一同 show，否则子控件（如 rubberBand）
    的 isVisible() 恒为 False。
    """
    pw = plot_factory(pd.DataFrame({"a": [float(i) for i in range(100)]}))
    pw.resize(800, 600)
    pw.window().show()
    pw.show()
    qapp.processEvents()
    return pw


def _zoom_center_mapping(pw, local_pos: QPoint) -> float:
    """复刻 wheelEvent 的鼠标位置→view 坐标映射，返回鼠标处 X 值"""
    vb = pw.plot_item.getViewBox()
    scene_pos = pw.mapToScene(local_pos)
    return vb.mapSceneToView(scene_pos).x()


# ---------- 滚轮缩放 ----------

def test_wheel_zoom_in_centered_on_mouse(shown_plot, qapp):
    """滚轮向前：X 范围以鼠标为中心收缩 (1-FACTOR_SCROLL_ZOOM) 倍"""
    pw = shown_plot
    assert pw.plot_variable("a")
    vb = pw.view_box
    vb.setXRange(1, 100, padding=0)
    qapp.processEvents()

    pos = QPoint(400, 300)
    mouse_x = _zoom_center_mapping(pw, pos)
    left, right = vb.viewRange()[0]
    factor = max(0.000001, 1 - FACTOR_SCROLL_ZOOM)
    expected = (
        mouse_x - (mouse_x - left) * factor,
        mouse_x + (right - mouse_x) * factor,
    )

    pw.wheelEvent(_wheel_event(pw, pos, 120))

    actual = vb.viewRange()[0]
    assert actual[0] == pytest.approx(expected[0], rel=1e-6)
    assert actual[1] == pytest.approx(expected[1], rel=1e-6)
    # 新范围严格收缩
    assert actual[1] - actual[0] < right - left


def test_wheel_zoom_out_expands_range(shown_plot, qapp):
    """滚轮向后：X 范围以鼠标为中心放大 (1+FACTOR_SCROLL_ZOOM) 倍"""
    pw = shown_plot
    assert pw.plot_variable("a")
    vb = pw.view_box
    vb.setXRange(20, 80, padding=0)
    qapp.processEvents()
    width_before = vb.viewRange()[0][1] - vb.viewRange()[0][0]

    pw.wheelEvent(_wheel_event(pw, QPoint(400, 300), -120))

    width_after = vb.viewRange()[0][1] - vb.viewRange()[0][0]
    assert width_after == pytest.approx(width_before * (1 + FACTOR_SCROLL_ZOOM), rel=1e-6)


def test_wheel_with_modifier_delegates_to_pyqtgraph_default(shown_plot, qapp):
    """Ctrl/Shift + 滚轮：不走应用自定义的鼠标中心 X-only 缩放，
    委托给父类 → pyqtgraph ViewBox 默认行为（1.02^n 双轴缩放，
    受 mouseEnabled 掩码控制）"""
    pw = shown_plot
    assert pw.plot_variable("a")
    vb = pw.view_box
    vb.setXRange(10, 90, padding=0)
    vb.setYRange(-5, 105, padding=0)
    qapp.processEvents()
    before_x = vb.viewRange()[0]
    before_y = vb.viewRange()[1]

    pos = QPoint(400, 300)
    mouse_x = _zoom_center_mapping(pw, pos)
    factor = max(0.000001, 1 - FACTOR_SCROLL_ZOOM)
    custom_expected = (
        mouse_x - (mouse_x - before_x[0]) * factor,
        mouse_x + (before_x[1] - mouse_x) * factor,
    )

    for mods in (Qt.KeyboardModifier.ControlModifier, Qt.KeyboardModifier.ShiftModifier):
        vb.setXRange(*before_x, padding=0)
        qapp.processEvents()
        pw.wheelEvent(_wheel_event(pw, pos, 120, modifiers=mods))
        actual_x = vb.viewRange()[0]
        # 不属于应用自定义路径：X 范围与鼠标中心 factor 缩放期望不同，
        # 且 Y 可能被 pyqtgraph 默认缩放（自定义路径恒不动 Y）
        custom_applied = (
            actual_x[0] == pytest.approx(custom_expected[0], rel=1e-6)
            and actual_x[1] == pytest.approx(custom_expected[1], rel=1e-6)
            and vb.viewRange()[1] == list(before_y)
        )
        assert not custom_applied, f"修饰键 {mods} 下误走了自定义缩放路径"


def test_wheel_zoom_preserves_y_range(shown_plot, qapp):
    """滚轮仅缩放 X 轴：Y 范围不受影响（scaleBy factor=(f, 1)）"""
    pw = shown_plot
    assert pw.plot_variable("a")
    vb = pw.view_box
    vb.setXRange(1, 100, padding=0)
    vb.setYRange(-10, 120, padding=0)
    qapp.processEvents()
    y_before = vb.viewRange()[1]

    pw.wheelEvent(_wheel_event(pw, QPoint(400, 300), 120))

    assert vb.viewRange()[1] == list(y_before)


# ---------- XLink 同步 ----------

def test_xlink_sync_on_wheel_zoom(plot_factory, qapp):
    """XLink：pw1 滚轮缩放后，联动子图 pw2 的 X 范围同步一致"""
    df = pd.DataFrame({"a": [float(i) for i in range(100)]})
    pw1 = plot_factory(df)
    pw2 = plot_factory(df)
    for pw in (pw1, pw2):
        pw.resize(800, 300)
        pw.show()
        assert pw.plot_variable("a")
        pw.view_box.setXRange(1, 100, padding=0)
    qapp.processEvents()

    # 生产中等价于 layout_manager 建立 XLink：后续子图链接到首个子图
    pw2.view_box.setXLink(pw1.view_box)

    pw1.wheelEvent(_wheel_event(pw1, QPoint(400, 150), 120))
    qapp.processEvents()

    x1 = pw1.view_box.viewRange()[0]
    x2 = pw2.view_box.viewRange()[0]
    assert x2[0] == pytest.approx(x1[0], abs=1e-6)
    assert x2[1] == pytest.approx(x1[1], abs=1e-6)
    # 确实发生了缩放（范围比初始 1..100 窄）
    assert x1[1] - x1[0] < 99


# ---------- 框选缩放（Shift + 左键拖拽） ----------

def test_box_zoom_shift_drag_with_margin(shown_plot, qapp):
    """Shift+左键框选：释放后视图缩放到框区域（含 10% margin）"""
    pw = shown_plot
    assert pw.plot_variable("a")
    vb = pw.view_box
    vb.setXRange(1, 100, padding=0)
    vb.setYRange(-5, 105, padding=0)
    qapp.processEvents()

    pa, pb = QPoint(100, 100), QPoint(400, 350)
    shift = Qt.KeyboardModifier.ShiftModifier

    # 期望值必须在释放前计算：释放后 viewRange 已变，坐标映射随之改变
    v1 = vb.mapSceneToView(pw.mapToScene(pa))
    v2 = vb.mapSceneToView(pw.mapToScene(pb))

    pw.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, pw, pa,
                                    Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, shift))
    assert pw.rubberBand.isVisible()
    pw.mouseMoveEvent(_mouse_event(QEvent.Type.MouseMove, pw, pb,
                                   Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton, shift))
    pw.mouseReleaseEvent(_mouse_event(QEvent.Type.MouseButtonRelease, pw, pb,
                                      Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton, shift))
    qapp.processEvents()

    assert not pw.rubberBand.isVisible()

    x_min, x_max = sorted((v1.x(), v2.x()))
    y_min, y_max = sorted((v1.y(), v2.y()))
    dx, dy = x_max - x_min, y_max - y_min
    expected = (x_min - 0.1 * dx, x_max + 0.1 * dx, y_min - 0.1 * dy, y_max + 0.1 * dy)

    actual = vb.viewRange()
    assert actual[0][0] == pytest.approx(expected[0], rel=1e-6)
    assert actual[0][1] == pytest.approx(expected[1], rel=1e-6)
    assert actual[1][0] == pytest.approx(expected[2], rel=1e-6)
    assert actual[1][1] == pytest.approx(expected[3], rel=1e-6)


def test_box_zoom_too_small_is_ignored(shown_plot, qapp):
    """框选区域过小（宽或高 ≤10px）：视为误触，视图不变"""
    pw = shown_plot
    assert pw.plot_variable("a")
    vb = pw.view_box
    vb.setXRange(1, 100, padding=0)
    qapp.processEvents()
    before = [list(axis) for axis in vb.viewRange()]

    pa, pb = QPoint(100, 100), QPoint(105, 103)  # 5x3 px
    shift = Qt.KeyboardModifier.ShiftModifier
    pw.mousePressEvent(_mouse_event(QEvent.Type.MouseButtonPress, pw, pa,
                                    Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, shift))
    pw.mouseMoveEvent(_mouse_event(QEvent.Type.MouseMove, pw, pb,
                                   Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton, shift))
    pw.mouseReleaseEvent(_mouse_event(QEvent.Type.MouseButtonRelease, pw, pb,
                                      Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton, shift))
    qapp.processEvents()

    assert [list(axis) for axis in vb.viewRange()] == before


# ---------- 双击中键清除 ----------

def test_middle_double_click_clears_plot(shown_plot, qapp, monkeypatch):
    """双击中键：清空曲线并立即请求标记统计刷新"""
    pw = shown_plot
    assert pw.plot_variable("a")

    calls = []
    monkeypatch.setattr(
        pw.window().layout_manager,
        "request_mark_stats_refresh",
        lambda immediate=False: calls.append(immediate),
    )

    pw.mouseDoubleClickEvent(
        _mouse_event(QEvent.Type.MouseButtonDblClick, pw, QPoint(300, 300),
                     Qt.MouseButton.MiddleButton, Qt.MouseButton.MiddleButton,
                     Qt.KeyboardModifier.NoModifier)
    )

    assert len(pw.curves) == 0
    assert calls == [True]  # immediate=True
