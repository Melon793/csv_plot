"""ViewBox 交互 component 测试（offscreen）：滚轮缩放 / XLink 同步 / 框选缩放 / 双击中键清除。

模拟 help.md 4️⃣「交互操作」中的用户动作：
- 滚轮缩放 X 轴（以鼠标位置为中心，factor=FACTOR_SCROLL_ZOOM）
- Ctrl/Shift 修饰键下滚轮不触发 X 轴缩放
- XLink：一个子图缩放，联动子图 X 范围同步
- Shift+左键框选放大（含 10% margin、过小框忽略）
- 双击中键清除单个图表

事件基于控件自身坐标构造，QWheelEvent/QMouseEvent 手工构造后直接调用对应事件方法；
多列 / 同列不等宽两类宿主拓扑（``zoom_group``）例外——它们要靠子图的屏幕几何错开
才能测到 XLink 锚点问题，断言口径用「缩放前后同一像素处的数据值漂移（px）」。
"""

import pandas as pd
import pytest

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

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
    """滚轮仅缩放 X 轴：Y 范围不受影响"""
    pw = shown_plot
    assert pw.plot_variable("a")
    vb = pw.view_box
    vb.setXRange(1, 100, padding=0)
    vb.setYRange(-10, 120, padding=0)
    qapp.processEvents()
    y_before = vb.viewRange()[1]

    pw.wheelEvent(_wheel_event(pw, QPoint(400, 300), 120))

    assert vb.viewRange()[1] == list(y_before)


def test_wheel_zoom_keeps_y_autorange_enabled(shown_plot, qapp):
    """P0-1 防回归：Ctrl+Y 开启「Y 跟随可见段」后，滚轮缩放 X 轴
    不得静默关掉 Y 轴 autoRange（用户显式开启的交互模式必须存活）。"""
    pw = shown_plot
    assert pw.plot_variable("a")
    vb = pw.view_box
    vb.enableAutoRange(x=False)
    vb.setYRange(-10, 120, padding=0)
    pw.auto_y_in_x_range()
    qapp.processEvents()

    assert bool(vb.state["autoRange"][1]) is True

    pw.wheelEvent(_wheel_event(pw, QPoint(400, 300), 120))
    qapp.processEvents()

    assert bool(vb.state["autoRange"][1]) is True, "滚轮缩放静默废掉了 Y 轴 autoRange"
    # X 仍按 factor 缩放
    assert vb.viewRange()[0][1] - vb.viewRange()[0][0] < 99


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


def _two_linked_plots(plot_factory, qapp):
    """首图 + 一个 XLink 子图，两者 X 范围一致 (1, 100)"""
    df = pd.DataFrame({"a": [float(i) for i in range(100)]})
    master = plot_factory(df)
    child = plot_factory(df)
    for pw in (master, child):
        pw.resize(800, 300)
        pw.show()
        assert pw.plot_variable("a")
        pw.view_box.setXRange(1, 100, padding=0)
    qapp.processEvents()
    child.view_box.setXLink(master.view_box)
    qapp.processEvents()
    return master, child


def test_wheel_zoom_on_linked_child_shrinks_group(plot_factory, qapp):
    """在 XLink 子图（第 2~N 个）上滚轮：整组 X 范围必须真的被缩放。

    AxisManager 若走 unlink→setXRange→relink，relink 末尾 pyqtgraph 会用源的
    range 回调 linkedViewChanged → 子图 setXRange，把刚写入的新范围按源的旧
    范围静默覆盖，用户看到的就是「滚轮没反应」。
    """
    master, child = _two_linked_plots(plot_factory, qapp)
    before = child.view_box.viewRange()[0]
    assert (before[0], before[1]) == pytest.approx((1, 100), abs=1e-6)

    child.wheelEvent(_wheel_event(child, QPoint(400, 150), 120))
    qapp.processEvents()

    after = child.view_box.viewRange()[0]
    assert after[1] - after[0] < before[1] - before[0], "子图滚轮被 XLink relink 快照回收"
    # 整组一致：子图缩放后与首图同源
    assert after[0] == pytest.approx(master.view_box.viewRange()[0][0], abs=1e-6)
    assert after[1] == pytest.approx(master.view_box.viewRange()[0][1], abs=1e-6)


def test_wheel_zoom_equivalent_on_master_and_child(plot_factory, qapp):
    """共享 X 轴的语义：在子图上滚轮应与在首图上滚轮得到同一结果"""
    master_a, child_a = _two_linked_plots(plot_factory, qapp)
    master_b, _child_b = _two_linked_plots(plot_factory, qapp)
    pos = QPoint(400, 150)

    child_a.wheelEvent(_wheel_event(child_a, pos, 120))
    master_b.wheelEvent(_wheel_event(master_b, pos, 120))
    qapp.processEvents()

    got = child_a.view_box.viewRange()[0]
    expected = master_b.view_box.viewRange()[0]
    assert got[0] == pytest.approx(expected[0], abs=1e-6)
    assert got[1] == pytest.approx(expected[1], abs=1e-6)


def test_wheel_zoom_on_linked_child_keeps_y_autorange(plot_factory, qapp):
    """滚轮委托给 XLink 源后，子图自己的「Y 跟随可见段」仍不得被废掉。

    子图滚轮会经 XLink 级联收到 linkedViewChanged → setXRange，此路径只关 X 轴
    autoRange；一旦有人改回 rect 形式的 setRange，Y autoRange 会整组被关掉。
    """
    master, child = _two_linked_plots(plot_factory, qapp)
    child.auto_y_in_x_range()
    qapp.processEvents()
    assert bool(child.view_box.state["autoRange"][1]) is True

    child.wheelEvent(_wheel_event(child, QPoint(400, 150), 120))
    qapp.processEvents()

    assert bool(child.view_box.state["autoRange"][1]) is True
    assert child.view_box.viewRange()[0][1] - child.view_box.viewRange()[0][0] < 99


class _NoopLayoutManager:
    """宿主 layout_manager 替身：覆盖绘图/滚轮路径上会被查询的少数成员（均 no-op）"""

    def request_mark_stats_refresh(self, immediate: bool = False):
        pass

    def _hide_drag_indicator_for_plot(self, pw):
        pass

    def _show_drag_indicator_for_plot(self, pw, var_names, text=None):
        pass

    def _get_plot_container(self, pw):
        return None


class _ZoomHost(QWidget):
    """XLink 组宿主：把多张真实子图装进同一个布局，让它们的屏幕几何真的错开。

    ``plot_factory`` 给每个 plot 各自一个 FakeHost（独立顶层窗口），offscreen 下二者
    ``screenGeometry().left()`` 相同 → 跨列类缺陷在该拓扑里天然测不到，故本文件自建。
    """

    def __init__(self):
        super().__init__()
        self.layout_manager = _NoopLayoutManager()


@pytest.fixture()
def zoom_group(qapp, plot_factory):
    """按拓扑建一组 XLink 子图，收尾删宿主。

    arrange="row" → 水平并排（跨列；pyqtgraph 走「整段复制」，两侧范围数值相同）
    arrange="col" → 垂直堆叠（同列；``left_margin`` 把第 2 张右移，造出「像素对齐」分支）
    """
    hosts = []

    def _make(arrange: str, n: int, left_margin: int = 0):
        host = _ZoomHost()
        lay = QHBoxLayout(host) if arrange == "row" else QVBoxLayout(host)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        hosts.append(host)

        df = pd.DataFrame({"a": [float(i) for i in range(100)]})
        pws = []
        for i in range(n):
            pw = plot_factory(df)
            assert pw.plot_variable("a")
            pw.setParent(host)
            pws.append(pw)
            if arrange == "col" and i == 1 and left_margin:
                wrap = QWidget()
                wl = QHBoxLayout(wrap)
                wl.setContentsMargins(left_margin, 0, 0, 0)
                wl.setSpacing(0)
                wl.addWidget(pw)
                lay.addWidget(wrap)
            else:
                lay.addWidget(pw)

        host.resize(1600, 900)
        host.show()
        qapp.processEvents()

        # 与生产同序：先关掉各图 X autoRange 再建链。子视图 X autoRange 为 True 时
        # ``ViewBox.linkView``（0.14.0 :1054-1059）不回调，整组停在「两侧范围不等」的
        # 非一致态，后面的锚点断言就失去前提。
        for pw in pws:
            pw.view_box.enableAutoRange(x=False)
        pws[0].view_box.setXRange(1, 100, padding=0)
        for pw in pws[1:]:
            pw.view_box.setXLink(pws[0].view_box)
        qapp.processEvents()
        pws[0].view_box.setXRange(1, 100, padding=0)
        qapp.processEvents()
        return host, pws

    yield _make

    for host in hosts:
        host.deleteLater()
    qapp.processEvents()


def _anchor_drift_px(pw, local: QPoint, x_before: float) -> float:
    """滚轮缩放后，鼠标所在像素对应的数据值偏了多少 px（0 = 鼠标点被钉住）。

    口径必须是像素漂移，不能用「与源逐位一致的范围」：后者在不等宽/非一致态下天然不成立。
    """
    vb = pw.view_box
    x_after = vb.mapSceneToView(pw.mapToScene(local)).x()
    lo, hi = vb.viewRange()[0]
    return (x_after - x_before) * vb.screenGeometry().width() / (hi - lo)


def test_wheel_zoom_anchor_pixel_stable_across_columns(qapp, zoom_group):
    """跨列并排：在第 2 列滚轮，鼠标像素处的数据值不得漂移。

    回归的是「把锚点按屏幕像素映射到 XLink 源视图」的缺陷 —— 跨列时 pyqtgraph 走
    整段复制（两侧 viewRange 数值相同），那次映射把两列之间的水平距离折进了缩放中心，
    实测把锚点推到视野外（偏 1 个多视宽），画面每齿缩放都整体横滑。
    """
    _host, (master, col2) = zoom_group("row", 2)
    # 前提：两列屏幕 left 必须不同，否则退化成同列拓扑、测不到本缺陷
    gap = col2.view_box.screenGeometry().left() - master.view_box.screenGeometry().left()
    assert gap > 50, f"宿主拓扑未错开列：left gap={gap}"
    assert col2.view_box.viewRange()[0] == pytest.approx(
        master.view_box.viewRange()[0], abs=1e-6
    )

    local = QPoint(int(col2.width() * 0.30), int(col2.height() * 0.5))
    x_before = col2.view_box.mapSceneToView(col2.mapToScene(local)).x()

    col2.wheelEvent(_wheel_event(col2, local, 120))
    qapp.processEvents()

    assert abs(_anchor_drift_px(col2, local, x_before)) < 0.01
    lo, hi = col2.view_box.viewRange()[0]
    assert hi - lo < 99, "滚轮未缩放"


def test_wheel_zoom_anchor_pixel_stable_on_three_columns(qapp, zoom_group):
    """3 列并排：每一列的滚轮都钉住鼠标点（缺陷在旧实现里按列序线性放大）。"""
    _host, pws = zoom_group("row", 3)
    for pw in pws[1:]:
        local = QPoint(int(pw.width() * 0.35), int(pw.height() * 0.5))
        x_before = pw.view_box.mapSceneToView(pw.mapToScene(local)).x()

        pw.wheelEvent(_wheel_event(pw, local, 120))
        qapp.processEvents()

        assert abs(_anchor_drift_px(pw, local, x_before)) < 0.01, (
            f"列 {pws.index(pw) + 1} 滚轮锚点漂移"
        )


def test_wheel_zoom_anchor_pixel_stable_on_unequal_width_stack(qapp, zoom_group):
    """同列上下两张、第 2 张视窗被右移（pyqtgraph 像素对齐分支）：不回归。

    该分支下「按屏幕像素映射锚点」与「不映射」数学上给出同一结果（同一像素在两视图里
    本就是同一数据值），所以删掉换算不影响这一形态 —— 这条用例就是钉这个结论的。
    """
    _host, (master, child) = zoom_group("col", 2, left_margin=80)
    # 前提：真的落在像素对齐分支（两侧 viewRange 数值不同）
    assert child.view_box.viewRange()[0] != pytest.approx(
        master.view_box.viewRange()[0], abs=1e-3
    ), "宿主未造出不等宽的像素对齐态，用例失去意义"

    local = QPoint(int(child.width() * 0.30), int(child.height() * 0.5))
    x_before = child.view_box.mapSceneToView(child.mapToScene(local)).x()

    child.wheelEvent(_wheel_event(child, local, 120))
    qapp.processEvents()

    assert abs(_anchor_drift_px(child, local, x_before)) < 0.01
    lo, hi = child.view_box.viewRange()[0]
    assert hi - lo < (
        master.view_box.viewRange()[0][1] - master.view_box.viewRange()[0][0]
    ) + 1e-6, "整组未按同一比例缩放"



def test_set_xrange_from_linked_child_writes_link_source(plot_factory, qapp):
    """set_xrange_with_link_handling 在有 XLink 时必须写源视图，而不是子图自身"""
    master, child = _two_linked_plots(plot_factory, qapp)

    child.set_xrange_with_link_handling(20, 40, padding=0)
    qapp.processEvents()

    assert master.view_box.viewRange()[0][0] == pytest.approx(20, abs=1e-6)
    assert master.view_box.viewRange()[0][1] == pytest.approx(40, abs=1e-6)
    assert child.view_box.viewRange()[0] == pytest.approx(
        master.view_box.viewRange()[0], abs=1e-6
    )
    # 不得残留临时 unlink/重连造成的 link 丢失
    assert child.view_box.linkedView(0) is master.view_box


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
