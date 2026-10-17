"""标注「零干扰」对照测试（offscreen）。

这是本功能最重要的一份测试。用户要求"不影响代码原功能"，所以不能只验
"加标注后还能缩放"，而必须验**逐项完全一致**：同一串交互施加在"有标注"
和"无标注"两只 widget 上，每一步之后的视图范围必须逐位相同。

做法：``_make_pair`` 造两只除标注外完全相同的 widget，``_run_gesture_sequence``
对两者施加同一串滚轮缩放 / 拖拽平移 / Shift 框选缩放，然后整体比对快照。

空洞风险与对策：如果交互序列本身没生效，两边都"没变化"也会相等。因此
每次比对前先断言序列**确实改变了视图**（见 ``_assert_sequence_effective``）。

机制说明（P0 实测结论）：
- 自动范围隔离靠 ``dataBounds()`` 覆写返回 ``(None, None)``：图元仍进
  ViewBox.addedItems（所以 ViewBox 拆解时会正常回收它们），但
  ``childrenBounds()`` 会跳过它们。**不存在** ``setIgnoreBounds`` 这种 API。
- 交互隔离靠"非编辑态一律 NoButton + translatable=False + 手柄隐藏"。
  手柄必须隐藏：pyqtgraph 的 Handle 是 UIGraphicsItem，其 hoverEvent
  **无条件**调用 acceptDrags，不看 ROI 的 translatable。
"""

from __future__ import annotations

import pandas as pd

from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtWidgets import QGraphicsItem

from src.core.annotation_models import Z_ANNOTATION
from tests.component._viewbox_events import mouse_event, wheel_event

#: 参与对照的全部图元类型。折线（P3）必须一起进来：它的门控面比前五类
#: 多两层 —— 容器之外还有"线段"这种独立子图元，漏掉就等于漏掉一整条通路。
#: 标靶（P3）同理：它是唯一带**附属标签图元**的一类，而标签自己实现
#: mouseClickEvent/mouseDragEvent 并转发给父项，浏览态照吃按键 —— 从标签上
#: 起手拖拽会抢走 ViewBox 的平移，必须由本对照守住。
ANNOTATION_KINDS = ("text", "rect", "ellipse", "line", "arrow", "polyline", "target")


# ---------------------------------------------------------------------------
# 夹具与工具
# ---------------------------------------------------------------------------

def _make_pair(plot_factory, qapp, n: int = 100):
    """造两只除标注外完全相同的 widget：(baseline 无标注, decorated 加标注)"""
    df = pd.DataFrame({"a": [float(i) for i in range(n)]})
    pair = []
    for _ in range(2):
        pw = plot_factory(df)
        pw.resize(800, 600)
        pw.window().show()
        pw.show()
        assert pw.plot_variable("a"), "曲线未绘出，后续对比失去意义"
        pw.view_box.setXRange(1, n - 1, padding=0)
        pw.view_box.setYRange(-5, n + 5, padding=0)
        pair.append(pw)
    qapp.processEvents()
    return pair[0], pair[1]


def _decorate_widget(pw, *, far_outside: bool = False) -> int:
    """在 widget 的视图中央铺满五类标注，返回创建条数

    刻意让矩形/椭圆覆盖视图中心 —— 后面的拖拽/框选落点就在中心附近，
    这样"标注没抢走鼠标"才是有内容的断言。
    """
    vb = pw.view_box
    mgr = pw.annotation_manager
    x_range, y_range = vb.viewRange()
    vb.context_x = (x_range[0] + x_range[1]) / 2.0
    vb.context_y = (y_range[0] + y_range[1]) / 2.0

    created = 0
    for kind in ANNOTATION_KINDS:
        if mgr.create_at_context(kind) is not None:
            created += 1
    if far_outside:
        # 不隔离的话自动范围会被撑到 1e6 量级
        if mgr.create("text", [(1e6, 1e6)], text="远处") is not None:
            created += 1
        if mgr.create("rect", [(9e5, 9e5), (1.1e6, 1.1e6)]) is not None:
            created += 1
    return created


def _view_range(vb) -> tuple:
    """把 viewRange 展平成可精确比较的元组（避免 numpy 标量与 list 的类型噪声）"""
    x, y = vb.viewRange()
    return (float(x[0]), float(x[1]), float(y[0]), float(y[1]))


def _settle_autorange(pw, qapp, rounds: int = 10) -> tuple:
    """把 ``autoRange()`` 推到不动点后返回视图范围。

    ``ViewBox.autoRange`` 不是纯函数：它读当前 viewRect/targetRect 再写回，
    Y 又开着 ``autoVisibleOnly``（Y 跟着可见 X 段走），X↔Y 互相影响，要几轮
    才稳定。因此**不能跨 widget 比较**两次 autoRange 的结果 —— 两个实例的
    历史不同，收敛值也不同，断言会变成噪声。

    能成立的比较是**同一 widget 的前后对照**：把视图推稳、记录、只改动
    "有没有标注"这一件事、再推稳、再记录。唯一的变量就是标注。
    """
    for _ in range(rounds):
        pw.view_box.autoRange()
        qapp.processEvents()
        qapp.processEvents()
    return _view_range(pw.view_box)


def _run_gesture_sequence(pw, qapp, *, wheel_only: bool = False) -> list[tuple]:
    """对 widget 施加一串固定交互，返回每一步之后的视图范围快照

    ``wheel_only=True`` 只走滚轮缩放，跳过落在画布上的拖拽 —— 编辑态下拖拽
    会被标注图元合法地接走（那正是编辑模式的目的），只有滚轮才该保持一致。

    **两个手势之间必须泵一次事件循环**，否则对照会被一条 pyqtgraph 既有行为
    污染：``ViewBox.mouseDragEvent`` 的平移量用 ``childGroup.transform()``
    这个**缓存**矩阵把屏幕像素换算成数据单位，而该矩阵要等下一次 repaint 才
    刷新。改完范围立刻拖拽，拖拽用的是旧范围的比例尺 —— 实测表现为
    "X 平移量按滚轮前的跨度算、Y 正常（Y 没变所以缓存不脏）"。

    这与标注无关：不泵事件时**无标注那一侧**才会算错，有标注那侧因为场景
    重绘更频繁反而用上了新矩阵。真实事件循环里鼠标移动本身就会持续泵事件，
    所以泵一次才是对的模拟。
    """
    vb = pw.view_box
    snapshots: list[tuple] = []

    vb.setXRange(1, 99, padding=0)
    vb.setYRange(-5, 105, padding=0)
    qapp.processEvents()
    snapshots.append(_view_range(vb))                       # 0 初始

    pw.wheelEvent(wheel_event(pw, QPoint(400, 300), 120))
    qapp.processEvents()
    snapshots.append(_view_range(vb))                       # 1 滚轮放大

    pw.wheelEvent(wheel_event(pw, QPoint(250, 200), -120))
    qapp.processEvents()
    snapshots.append(_view_range(vb))                       # 2 滚轮缩小

    if wheel_only:
        return snapshots

    left, none = Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton
    shift = Qt.KeyboardModifier.ShiftModifier
    plain = Qt.KeyboardModifier.NoModifier

    # 左键拖拽 = 平移（起点刻意落在标注覆盖的视图中央）
    pw.mousePressEvent(mouse_event(QEvent.Type.MouseButtonPress, pw, QPoint(400, 300), left, left, plain))
    qapp.processEvents()
    pw.mouseMoveEvent(mouse_event(QEvent.Type.MouseMove, pw, QPoint(340, 260), none, left, plain))
    qapp.processEvents()
    pw.mouseReleaseEvent(mouse_event(QEvent.Type.MouseButtonRelease, pw, QPoint(340, 260), left, none, plain))
    qapp.processEvents()
    snapshots.append(_view_range(vb))                       # 3 平移

    # Shift + 左键拖拽 = 框选放大（同样从被标注覆盖的中央起框）
    pw.mousePressEvent(mouse_event(QEvent.Type.MouseButtonPress, pw, QPoint(400, 300), left, left, shift))
    qapp.processEvents()
    pw.mouseMoveEvent(mouse_event(QEvent.Type.MouseMove, pw, QPoint(150, 480), none, left, shift))
    qapp.processEvents()
    pw.mouseReleaseEvent(mouse_event(QEvent.Type.MouseButtonRelease, pw, QPoint(150, 480), left, none, shift))
    qapp.processEvents()
    snapshots.append(_view_range(vb))                       # 4 框选

    return snapshots


def _assert_sequence_effective(snapshots: list[tuple], where: str) -> None:
    """确认交互序列真的改变了视图，否则两边"都没动"的相等是空洞的"""
    labels = ("滚轮放大", "滚轮缩小", "拖拽平移", "框选缩放")
    for index, label in enumerate(labels[: len(snapshots) - 1]):
        assert snapshots[index] != snapshots[index + 1], (
            f"{where}：{label} 未生效（{snapshots[index]} -> {snapshots[index + 1]}），"
            "对照失去意义"
        )


# ---------------------------------------------------------------------------
# 机制层：自动范围隔离
# ---------------------------------------------------------------------------

class TestAutoRangeIsolation:
    def test_annotation_reports_no_data_bounds(self, plot_factory, qapp):
        """报告 (None, None) 才是隔离生效的直接原因"""
        pw, _ = _make_pair(plot_factory, qapp)
        assert _decorate_widget(pw) >= len(ANNOTATION_KINDS)

        for record in pw.annotation_manager._graphics.values():
            assert record.item.dataBounds(0) == (None, None), record.kind
            assert record.item.dataBounds(1) == (None, None), record.kind

    def test_arrow_head_also_reports_no_data_bounds(self, plot_factory, qapp):
        """箭头头部是独立图元，漏掉它 Y 轴会被撑到 (0,0) 以外"""
        pw, _ = _make_pair(plot_factory, qapp)
        _decorate_widget(pw)
        heads = [r.decor for r in pw.annotation_manager._graphics.values() if r.decor is not None]
        assert heads, "箭头头部未生成，断言无内容"
        for head in heads:
            assert head.dataBounds(0) == (None, None)
            assert head.dataBounds(1) == (None, None)

    def test_items_still_registered_on_viewbox_for_teardown(self, plot_factory, qapp):
        """隔离靠 dataBounds，不靠"不进 addedItems" —— 进表才能被 ViewBox 正常回收"""
        pw, _ = _make_pair(plot_factory, qapp)
        _decorate_widget(pw)
        vb = pw.view_box
        for record in pw.annotation_manager._graphics.values():
            assert record.item in vb.addedItems, record.kind

    def test_children_bounds_unchanged_by_annotations(self, plot_factory, qapp):
        """childrenBounds（自动范围的输入）在加标注前后逐位一致

        **必须是同一只 widget 的前后对照**。跨实例比较在本用例上是不成立的：
        childrenBounds 会跟着当前视图范围漂（Y 轴开着 autoVisibleOnly，X↔Y 互相
        影响），两个实例历史不同、收敛值就不同 —— 实测同一只 widget 前后 4/4 一致，
        换成跨实例 4 轮里 3 轮不一致（且与标注多少无关：只建一条矩形同样会漂）。
        见 README 陷阱 #14。
        """
        pw, _ = _make_pair(plot_factory, qapp)
        before = pw.view_box.childrenBounds()

        assert _decorate_widget(pw, far_outside=True) >= len(ANNOTATION_KINDS)
        after = pw.view_box.childrenBounds()

        assert after == before, f"标注污染了子项边界: {before} -> {after}"
        # 非空转守卫：边界确实覆盖到了数据，否则"两边都是同一个空值"也会相等
        assert before[0][1] > 90.0 and before[1][1] > 90.0, f"边界没覆盖数据: {before}"

    def test_settled_autorange_unchanged_by_annotations(self, plot_factory, qapp):
        """加满远处标注后，推稳的自动范围逐位不变（同一 widget 前后对照）"""
        pw, _ = _make_pair(plot_factory, qapp)
        clean = _settle_autorange(pw, qapp)

        assert _decorate_widget(pw, far_outside=True) >= len(ANNOTATION_KINDS)
        with_annotations = _settle_autorange(pw, qapp)

        assert with_annotations == clean
        # 非空转守卫：自动范围确实覆盖到了数据上界、且没被 1e6 处的标注撑爆
        assert clean[3] > 99.0, f"Y 未覆盖数据上界，断言无内容: {clean}"
        assert with_annotations[1] < 1000.0, f"X 被远处标注撑爆: {with_annotations}"

    def test_settled_autorange_restored_after_clear(self, plot_factory, qapp):
        """清空标注后推稳的自动范围与清空前一致（回收彻底，不留污染）"""
        pw, _ = _make_pair(plot_factory, qapp)
        _decorate_widget(pw, far_outside=True)
        with_annotations = _settle_autorange(pw, qapp)

        pw.annotation_manager.clear()
        qapp.processEvents()

        assert _settle_autorange(pw, qapp) == with_annotations


# ---------------------------------------------------------------------------
# 交互层：整串手势逐位一致
# ---------------------------------------------------------------------------

class TestInteractionParity:
    def test_gesture_sequence_bit_identical(self, plot_factory, qapp):
        """核心断言：同一串交互在有/无标注时给出逐位相同的结果"""
        baseline, decorated = _make_pair(plot_factory, qapp)
        assert _decorate_widget(decorated, far_outside=True) >= len(ANNOTATION_KINDS)

        expected = _run_gesture_sequence(baseline, qapp)
        _assert_sequence_effective(expected, "无标注基线")

        actual = _run_gesture_sequence(decorated, qapp)
        for step, (exp, got) in enumerate(zip(expected, actual)):
            assert got == exp, f"第 {step} 步视图范围不一致"

    def test_wheel_gestures_identical_in_edit_mode(self, plot_factory, qapp):
        """编辑态只比滚轮：拖拽会被标注图元合法地接走，那不是干扰

        编辑模式下 ROI ``translatable=True``，落在它身上的拖拽本就该移动标注
        而不是平移视图 —— 这是设计目标而非回归。滚轮与框选不落在图元命中
        路径上，必须与无标注时逐位一致。
        """
        baseline, decorated = _make_pair(plot_factory, qapp)
        _decorate_widget(decorated)
        decorated.annotation_manager.set_edit_mode(True)
        qapp.processEvents()

        exp = _run_gesture_sequence(baseline, qapp, wheel_only=True)
        _assert_sequence_effective(exp, "无标注基线")
        got = _run_gesture_sequence(decorated, qapp, wheel_only=True)
        for step, (e, g) in enumerate(zip(exp, got)):
            assert g == e, f"编辑态第 {step} 步不一致"

    def test_viewbox_keeps_mouse_priority_when_not_editing(self, plot_factory, qapp):
        """非编辑态：标注不参与鼠标路由，ViewBox 仍拿到拖拽

        直接读 pyqtgraph 的拖拽归属（``scene().dragItem``）而不是合成鼠标
        事件去拖图元：离屏环境下 ``GraphicsScene`` 的拖拽判定还叠着
        ``minDragTime=0.5s`` / ``_moveDistance`` 这些时间与距离启发式，
        合成事件驱动不稳；而"谁拿到拖拽"这个归属问题是确定的。
        """
        pw, _ = _make_pair(plot_factory, qapp)
        _decorate_widget(pw)
        manager = pw.annotation_manager
        manager.set_edit_mode(False)
        qapp.processEvents()

        view_before = _view_range(pw.view_box)
        scene = pw.plot_item.scene()
        scene.dragItem = None
        scene.dragButtons = []

        left, none = Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton
        plain = Qt.KeyboardModifier.NoModifier
        pw.mousePressEvent(mouse_event(QEvent.Type.MouseButtonPress, pw, QPoint(400, 300), left, left, plain))
        qapp.processEvents()
        pw.mouseMoveEvent(mouse_event(QEvent.Type.MouseMove, pw, QPoint(340, 260), none, left, plain))
        qapp.processEvents()

        assert scene.dragItem is pw.view_box, (
            f"非编辑态下拖拽被 {scene.dragItem} 接走（应仍归 ViewBox）"
        )
        assert _view_range(pw.view_box) != view_before, "拖拽没生效，归属断言可能恒真"

        pw.mouseReleaseEvent(mouse_event(QEvent.Type.MouseButtonRelease, pw, QPoint(340, 260), left, none, plain))
        qapp.processEvents()

    def test_x_range_padding_identical(self, plot_factory, qapp):
        """setXRange 的 padding 外扩量不受标注影响"""
        baseline, decorated = _make_pair(plot_factory, qapp)
        _decorate_widget(decorated)

        for pw in (baseline, decorated):
            pw.view_box.setXRange(10, 20)
        qapp.processEvents()
        assert _view_range(decorated.view_box) == _view_range(baseline.view_box)

    def test_y_autorange_in_x_range_identical(self, plot_factory, qapp):
        """Ctrl+Y「仅调节 Y 轴」的结果不受标注影响"""
        baseline, decorated = _make_pair(plot_factory, qapp)
        _decorate_widget(decorated, far_outside=True)

        for pw in (baseline, decorated):
            pw.view_box.setXRange(30, 60, padding=0)
            pw.auto_y_in_x_range()
        qapp.processEvents()

        assert _view_range(decorated.view_box) == _view_range(baseline.view_box)


# ---------------------------------------------------------------------------
# 图元层：非编辑态不产生任何鼠标事件
# ---------------------------------------------------------------------------

class _RecordingHover:
    """记录 hoverEvent 是否被接受（等价于 pyqtgraph 内部的 ev.acceptDrags/acceptClicks）"""

    def __init__(self):
        self.drag_accepted = False
        self.click_count = 0

    def isExit(self):
        return False

    def acceptDrags(self, button):
        self.drag_accepted = True
        return True

    def acceptClicks(self, button):
        self.click_count += 1
        return True


class TestGating:
    def test_all_items_reject_left_button_in_non_edit_mode(self, plot_factory, qapp):
        """非编辑态：所有标注图元都不接受任何鼠标键（Qt 命中测试的第一道闸）"""
        pw, _ = _make_pair(plot_factory, qapp)
        _decorate_widget(pw)
        manager = pw.annotation_manager
        manager.set_edit_mode(False)
        qapp.processEvents()

        for record in manager._graphics.values():
            buttons = record.item.acceptedMouseButtons()
            assert buttons == Qt.MouseButton.NoButton, f"{record.kind}: {buttons}"
            if record.decor is not None:
                assert record.decor.acceptedMouseButtons() == Qt.MouseButton.NoButton

    def test_roi_hover_does_not_accept_drags_in_non_edit_mode(self, plot_factory, qapp):
        pw, _ = _make_pair(plot_factory, qapp)
        _decorate_widget(pw)
        manager = pw.annotation_manager
        manager.set_edit_mode(False)

        # 取一条 ROI（rect / ellipse / line 都是 ROI）
        record = next(r for r in manager._graphics.values() if hasattr(r.item, "sigRegionChanged"))
        stub = _RecordingHover()
        record.item.hoverEvent(stub)
        assert stub.drag_accepted is False, "非编辑态 ROI 抢走了拖拽"
        assert stub.click_count == 0, "非编辑态 ROI 抢走了点击"

    def test_roi_hover_accepts_drags_in_edit_mode(self, plot_factory, qapp):
        """对照组：证明上一条不是恒假（编辑态确实会接受）"""
        pw, _ = _make_pair(plot_factory, qapp)
        _decorate_widget(pw)
        manager = pw.annotation_manager
        manager.set_edit_mode(True)

        record = next(r for r in manager._graphics.values() if hasattr(r.item, "sigRegionChanged"))
        stub = _RecordingHover()
        record.item.hoverEvent(stub)
        assert stub.drag_accepted is True, "编辑态 ROI 拖不动，上一条断言退化为空转"

    def test_roi_query_flags_forbid_drag_and_resize_in_non_edit_mode(self, plot_factory, qapp):
        """translatable/resizable 必须为 False：这是 hoverEvent 放行的前置条件"""
        pw, _ = _make_pair(plot_factory, qapp)
        _decorate_widget(pw)
        manager = pw.annotation_manager

        for edit in (False, True):
            manager.set_edit_mode(edit)
            for record in manager._graphics.values():
                if not hasattr(record.item, "translatable"):
                    continue
                assert record.item.translatable is edit, record.kind
                assert record.item.resizable is edit, record.kind
                # 旋转会丢角度信息（矩形只存两个对角点），恒关
                assert record.item.rotatable is False
                assert record.item.removable is False

    def test_text_item_ignores_mouse_in_non_edit_mode(self, plot_factory, qapp):
        pw, _ = _make_pair(plot_factory, qapp)
        _decorate_widget(pw)
        manager = pw.annotation_manager
        text_record = next(r for r in manager._graphics.values() if r.kind == "text")

        manager.set_edit_mode(False)
        assert text_record.item.acceptedMouseButtons() == Qt.MouseButton.NoButton
        assert not text_record.item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsSelectable

        manager.set_edit_mode(True)
        assert text_record.item.acceptedMouseButtons() == Qt.MouseButton.LeftButton
        assert text_record.item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsSelectable

    def test_handles_hidden_in_non_edit_mode(self, plot_factory, qapp):
        """手柄必须真的隐藏：Handle 的 hoverEvent 无条件 acceptDrags，不看 ROI 标志"""
        pw, _ = _make_pair(plot_factory, qapp)
        _decorate_widget(pw)
        manager = pw.annotation_manager
        manager.set_edit_mode(False)

        roi_records = [r for r in manager._graphics.values() if getattr(r.item, "handles", None)]
        assert roi_records, "没有 ROI 手柄，断言无内容"
        for record in roi_records:
            for handle in record.item.handles:
                assert handle["item"].isVisible() is False, record.kind

    def test_handles_visible_in_edit_mode(self, plot_factory, qapp):
        """对照组：编辑态手柄可见"""
        pw, _ = _make_pair(plot_factory, qapp)
        _decorate_widget(pw)
        manager = pw.annotation_manager
        manager.set_edit_mode(True)

        roi_records = [r for r in manager._graphics.values() if getattr(r.item, "handles", None)]
        assert roi_records
        for record in roi_records:
            for handle in record.item.handles:
                assert handle["item"].isVisible() is True, record.kind

    def test_polyline_segments_are_gated_separately(self, plot_factory, qapp):
        """折线的线段是**独立**子图元，容器关不住它（浏览态点段会插折点）

        并且按键必须是**赋值**而不是 |= ：``PolyLineROI.addSegment`` 会给两端手柄
        ``setDeletable(True)``，把手柄按键 OR 上 RightButton（右键能删折点）。
        订阅式地 OR 会越关越多，浏览态就会冒出一个"右键删我折点"的菜单。
        """
        pw, _ = _make_pair(plot_factory, qapp)
        _decorate_widget(pw)
        manager = pw.annotation_manager
        record = next(r for r in manager._graphics.values() if r.kind == "polyline")
        segments = record.item.segments
        assert segments, "折线没有线段，断言无内容"

        manager.set_edit_mode(False)
        for segment in segments:
            assert segment.acceptedMouseButtons() == Qt.MouseButton.NoButton
        for handle in record.item.handles:
            assert handle["item"].acceptedMouseButtons() == Qt.MouseButton.NoButton

        manager.set_edit_mode(True)
        for segment in segments:
            assert segment.acceptedMouseButtons() == Qt.MouseButton.LeftButton
        for handle in record.item.handles:
            # 只能有 LeftButton：RightButton 是 pyqtgraph 自己 OR 上去的折点删除菜单
            assert handle["item"].acceptedMouseButtons() == Qt.MouseButton.LeftButton

    def test_polyline_segment_gating_survives_point_insertion(self, plot_factory, qapp):
        """编辑态在段上点一下会插折点，而新段/新手柄是 pyqtgraph 现造的

        现造的段自带 LeftButton、现造的手柄还会被 OR 上 RightButton ——
        退出编辑态时全量重刷一遍门控，必须连这批新图元一起洗干净。
        """
        pw, _ = _make_pair(plot_factory, qapp)
        _decorate_widget(pw)
        manager = pw.annotation_manager
        record = next(r for r in manager._graphics.values() if r.kind == "polyline")

        manager.set_edit_mode(True)
        item = record.item
        before = len(item.handles)
        item.segmentClicked(
            item.segments[0],
            pos=(item.handles[0]["item"].pos() + item.handles[1]["item"].pos()) / 2,
        )
        qapp.processEvents()
        assert len(item.handles) == before + 1, "折点没插进去，本用例退化为空转"

        manager.set_edit_mode(False)
        qapp.processEvents()
        for segment in item.segments:
            assert segment.acceptedMouseButtons() == Qt.MouseButton.NoButton
        for handle in item.handles:
            assert handle["item"].acceptedMouseButtons() == Qt.MouseButton.NoButton
            assert handle["item"].isVisible() is False

    def test_arrow_head_never_accepts_mouse(self, plot_factory, qapp):
        """箭头头部是纯装饰，两种模式下都不参与交互"""
        pw, _ = _make_pair(plot_factory, qapp)
        _decorate_widget(pw)
        manager = pw.annotation_manager
        head = next(r.decor for r in manager._graphics.values() if r.decor is not None)

        for edit in (False, True):
            manager.set_edit_mode(edit)
            assert head.acceptedMouseButtons() == Qt.MouseButton.NoButton

    def test_annotation_z_sits_between_curves_and_cursor(self, plot_factory, qapp):
        """图层序：盖住曲线、不遮挡游标（既有"显示游标"视觉行为不变）"""
        pw, _ = _make_pair(plot_factory, qapp)
        _decorate_widget(pw)
        for record in pw.annotation_manager._graphics.values():
            assert record.item.zValue() == Z_ANNOTATION
            if record.decor is not None:
                assert record.decor.zValue() == Z_ANNOTATION

    def test_items_are_visible_in_non_edit_mode(self, plot_factory, qapp):
        """门控只关交互，不关显示 —— 退出编辑后标注必须照旧看得见（截图用）"""
        pw, _ = _make_pair(plot_factory, qapp)
        _decorate_widget(pw)
        manager = pw.annotation_manager
        manager.set_edit_mode(False)
        qapp.processEvents()

        for record in manager._graphics.values():
            assert record.item.isVisible() is True, record.kind
