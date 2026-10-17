"""折线标注 component 测试（offscreen）。

折线是唯一"容器之外还挂着一整套独立子图元"的标注类型：``PolyLineROI`` 的线段
（``_PolyLineSegment``）自带鼠标键、自带 z、点一下还能自己插折点。本文件把这几条
边界逐条钉住，顺序与踩坑顺序一致：

1. 建出来的拓扑（折点数 ↔ 手指数 / 段数）与折点数上限
2. 几何读写必须走父坐标 —— 整体平移不许丢（P1 的直线/箭头就栽在这上面）
3. 子图元的 z 必须跟随容器，否则选中压不住别人、描边还会被更低的标注盖住
4. 模板往返 / 资源回收
5. **浏览态线段不抢 ViewBox 的拖拽**（零干扰里最难被看见的一层）

门控（浏览态/编辑态按键与手柄可见性，含插点后新生的段与手柄）放在
``test_annotation_no_interference.py`` 的 ``TestGating`` 里，与本文件不重复。
"""

from __future__ import annotations

import math
import time

import pandas as pd
import pytest

import pyqtgraph as pg
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QPainter, QPixmap

from src.core.annotation_models import MAX_POLYLINE_POINTS, Z_ANNOTATION, Z_ANNOTATION_SELECTED
from src.ui.widgets.annotation_items import read_geometry, write_geometry
from tests.component._viewbox_events import mouse_event

#: 屏幕上"落在折线段上"的落点，与 test_annotation_no_interference 的中央区域一致
_LANDING = QPoint(400, 300)
_DRAG_TO = QPoint(340, 260)


@pytest.fixture()
def annotated(plot_factory, qapp):
    """已绘图并固定视图范围的 widget

    x/y 都取 0..100，绘图区约 790×567 px —— 两轴比例接近（各 ~7.9 px/单位），
    屏幕落点换算成数据坐标后可以直接当几何用，不必再纠偏。
    """
    df = pd.DataFrame({"a": [float(i) for i in range(100)]})
    pw = plot_factory(df)
    pw.resize(800, 600)
    pw.window().show()
    pw.show()
    assert pw.plot_variable("a"), "曲线未绘出，几何断言失去意义"
    pw.view_box.setXRange(0, 100, padding=0)
    pw.view_box.setYRange(0, 100, padding=0)
    qapp.processEvents()
    return pw


def _manager(pw):
    return pw.annotation_manager


def _item(pw, aid):
    return _manager(pw)._graphics[aid].item


def _make_polyline(pw, points) -> str:
    aid = _manager(pw).create("polyline", points)
    assert aid is not None, f"折线未创建: {points}"
    return aid


def _view_range(pw) -> tuple[float, float, float, float]:
    x, y = pw.view_box.viewRange()
    return (float(x[0]), float(x[1]), float(y[0]), float(y[1]))


def _landing_data_point(pw) -> tuple[float, float]:
    """把屏幕落点换算成数据坐标，供"让折线精确穿过落点"用"""
    view_point = pw.view_box.mapSceneToView(pw.mapToScene(_LANDING))
    return float(view_point.x()), float(view_point.y())


# ---------------------------------------------------------------------------
# 拓扑与上限
# ---------------------------------------------------------------------------

class TestTopology:
    def test_three_points_give_two_segments(self, annotated):
        aid = _make_polyline(annotated, [(0.0, 0.0), (5.0, 9.0), (10.0, 1.0)])
        item = _item(annotated, aid)
        assert isinstance(item, pg.PolyLineROI)
        assert len(item.handles) == 3
        assert len(item.segments) == 2

    def test_two_points_is_the_minimum(self, annotated):
        aid = _make_polyline(annotated, [(0.0, 0.0), (10.0, 10.0)])
        item = _item(annotated, aid)
        assert len(item.handles) == 2
        assert len(item.segments) == 1

    def test_single_point_is_rejected(self, annotated):
        """下限在模型层收口：1 个点画不出折线，不能留一条畸形标注"""
        assert _manager(annotated).create("polyline", [(0.0, 0.0)]) is None
        assert _manager(annotated).count() == 0

    def test_extra_points_are_truncated_to_the_cap(self, annotated):
        """折点数可以由用户点击无限追加，模板 points 更是外部输入 —— 必须有上限"""
        points = [(float(i), float(i % 7)) for i in range(MAX_POLYLINE_POINTS + 15)]
        aid = _make_polyline(annotated, points)
        item = _item(annotated, aid)
        assert len(item.handles) == MAX_POLYLINE_POINTS
        assert len(item.segments) == MAX_POLYLINE_POINTS - 1
        assert len(_manager(annotated).get(aid).points) == MAX_POLYLINE_POINTS

    def test_segment_click_inserts_a_point(self, annotated):
        """pyqtgraph 自带交互：点段插折点（编辑态下这是主要编辑手段）"""
        aid = _make_polyline(annotated, [(0.0, 0.0), (10.0, 10.0), (20.0, 0.0)])
        item = _item(annotated, aid)
        _manager(annotated).set_edit_mode(True)

        first, second = item.handles[0]["item"], item.handles[1]["item"]
        item.segmentClicked(item.segments[0], pos=(first.pos() + second.pos()) / 2)

        assert len(item.handles) == 4
        assert len(item.segments) == 3
        assert len(_manager(annotated).get(aid).points) == 4, "插的点没回写模型"

    def test_point_insertion_stops_at_the_cap(self, annotated):
        """上限必须在**加点之前**挡：事后由管理器拒绝回写会让画面多出一个
        模型里不存在的折点，两边分裂且再也对不上"""
        aid = _make_polyline(
            annotated, [(float(i), 0.0) for i in range(MAX_POLYLINE_POINTS)]
        )
        item = _item(annotated, aid)
        _manager(annotated).set_edit_mode(True)

        first, second = item.handles[0]["item"], item.handles[1]["item"]
        item.segmentClicked(item.segments[0], pos=(first.pos() + second.pos()) / 2)

        assert len(item.handles) == MAX_POLYLINE_POINTS
        assert len(_manager(annotated).get(aid).points) == MAX_POLYLINE_POINTS


# ---------------------------------------------------------------------------
# 几何读写：必须换算到父坐标（数据坐标）
# ---------------------------------------------------------------------------

class TestGeometryUsesParentCoordinates:
    def test_overall_translation_is_not_lost(self, annotated):
        """ROI 整体平移只改自己的位置，手柄一个都不动

        照抄"直接读手柄（局部）坐标"会把平移丢掉：屏幕上线条移了、模型里还是老
        坐标，存模板再套回来就跳回原位。P1 的直线/箭头就踩在这上面。
        """
        aid = _make_polyline(annotated, [(10.0, 10.0), (30.0, 50.0), (60.0, 20.0)])
        item = _item(annotated, aid)

        item.translate(7.0, -3.0)
        assert read_geometry(item) == pytest.approx([(17.0, 7.0), (37.0, 47.0), (67.0, 17.0)])
        # 顺手确认这条平移是真的走 signal 回了模型（不是只改了图元）
        assert _manager(annotated).get(aid).points == pytest.approx(
            [(17.0, 7.0), (37.0, 47.0), (67.0, 17.0)]
        )

    def test_write_then_read_is_the_identity(self, annotated):
        """连续两次往返必须收敛（差一点都会在存/取模板时放大）"""
        aid = _make_polyline(annotated, [(10.0, 10.0), (30.0, 50.0), (60.0, 20.0)])
        item = _item(annotated, aid)

        target = [(1.0, 2.0), (5.0, 8.0), (9.0, 3.0), (13.0, 6.0)]
        write_geometry(item, target)
        assert read_geometry(item) == pytest.approx(target)

        write_geometry(item, target)
        assert read_geometry(item) == pytest.approx(target)

    def test_write_after_translation_still_lands_on_target(self, annotated):
        """平移之后再写几何，必须落在目标上（写路径也得走父坐标的反变换）"""
        aid = _make_polyline(annotated, [(10.0, 10.0), (30.0, 50.0), (60.0, 20.0)])
        item = _item(annotated, aid)
        item.translate(40.0, -25.0)

        target = [(2.0, 3.0), (6.0, 7.0), (11.0, 4.0)]
        write_geometry(item, target)
        assert read_geometry(item) == pytest.approx(target)

    def test_dragging_a_handle_writes_back_to_the_model(self, annotated):
        """真实拖折点走的是 movePoint(..., finish=True)，回写必须跟上"""
        aid = _make_polyline(annotated, [(5.0, 5.0), (25.0, 45.0), (55.0, 15.0)])
        item = _item(annotated, aid)
        handle = item.handles[1]["item"]

        handle.movePoint(QPointF(handle.pos().x() + 3.0, handle.pos().y() + 4.0), finish=True)

        expected = read_geometry(item)
        assert expected[1] != pytest.approx((25.0, 45.0)), "拖拽没生效，断言可能恒真"
        assert _manager(annotated).get(aid).points == pytest.approx(expected)

    def test_under_minimum_readback_is_not_written_into_the_model(self, annotated, monkeypatch):
        """重建手柄的间隙会读到过少的点：写进去会让整条标注变非法

        非法标注会让整份模板加载失败，代价远大于"少写一次"。这里直接把回读
        结果替身成"1 个点"，钉住那道闸；真实的中间态（setPoints 拆建之间）
        走的是同一条早退分支，只是中途还会路过"点数合法"的形态，不好稳定复现。
        """
        from src.ui.widgets import annotation_manager as am_module

        manager = _manager(annotated)
        aid = _make_polyline(annotated, [(1.0, 1.0), (5.0, 9.0), (9.0, 2.0)])
        before = list(manager.get(aid).points)

        monkeypatch.setattr(am_module, "read_geometry", lambda _item: [(0.0, 0.0)])
        manager._on_item_geometry_changed(aid)
        assert manager.get(aid).points == before

    def test_rebuilding_points_ends_on_the_full_set(self, annotated):
        """setPoints 会先清空再重建，中途路过"点数不足"的形态是允许的 ——
        关键是最后落到完整的一组上，不能停在半途。"""
        aid = _make_polyline(annotated, [(1.0, 1.0), (5.0, 9.0), (9.0, 2.0)])
        item = _item(annotated, aid)
        target = [(0.0, 0.0), (4.0, 8.0), (8.0, 1.0), (12.0, 6.0)]

        item.clearPoints()
        assert read_geometry(item) == []
        write_geometry(item, target)

        assert _manager(annotated).get(aid).points == pytest.approx(target)
        assert len(item.handles) == 4 and len(item.segments) == 3


# ---------------------------------------------------------------------------
# 图层序：子图元要跟着容器走
# ---------------------------------------------------------------------------

class TestZOrder:
    def test_segments_follow_the_container_on_select(self, annotated):
        """线段的 z 只在建段那一刻取一次 self.zValue() + 1，之后容器改 z 它不跟

        不同步的话描边停在 11：被任何一条 z=50 的标注盖住，选中抬到 90 也压不住别人。
        """
        aid = _make_polyline(annotated, [(1.0, 1.0), (5.0, 9.0), (9.0, 2.0)])
        item = _item(annotated, aid)
        assert item.zValue() == Z_ANNOTATION
        assert [s.zValue() for s in item.segments] == [Z_ANNOTATION + 1] * 2

        _manager(annotated).select(aid)
        assert item.zValue() == Z_ANNOTATION_SELECTED
        assert [s.zValue() for s in item.segments] == [Z_ANNOTATION_SELECTED + 1] * 2

        _manager(annotated).select(None)
        assert [s.zValue() for s in item.segments] == [Z_ANNOTATION + 1] * 2

    def test_segments_rebuilt_by_write_geometry_keep_the_z(self, annotated):
        """setPoints 会把段全拆重建，新建的段必须落在同一层"""
        aid = _make_polyline(annotated, [(1.0, 1.0), (5.0, 9.0), (9.0, 2.0)])
        item = _item(annotated, aid)
        item.setZValue(Z_ANNOTATION_SELECTED)

        write_geometry(item, [(0.0, 0.0), (4.0, 8.0), (8.0, 1.0), (12.0, 6.0)])
        assert [s.zValue() for s in item.segments] == [Z_ANNOTATION_SELECTED + 1] * 3

    def test_segments_inserted_by_click_keep_the_z(self, annotated):
        aid = _make_polyline(annotated, [(1.0, 1.0), (5.0, 9.0), (9.0, 2.0)])
        item = _item(annotated, aid)
        _manager(annotated).select(aid)

        first, second = item.handles[0]["item"], item.handles[1]["item"]
        item.segmentClicked(item.segments[0], pos=(first.pos() + second.pos()) / 2)

        assert [s.zValue() for s in item.segments] == [Z_ANNOTATION_SELECTED + 1] * 3


# ---------------------------------------------------------------------------
# 填充（描边由线段子项负责，填充自己画）
# ---------------------------------------------------------------------------

class TestFill:
    def test_no_brush_by_default(self, annotated):
        aid = _make_polyline(annotated, [(1.0, 1.0), (5.0, 9.0), (9.0, 2.0)])
        assert _item(annotated, aid)._fill_brush is None

    def test_style_sets_a_translucent_brush(self, annotated):
        aid = _make_polyline(annotated, [(1.0, 1.0), (5.0, 9.0), (9.0, 2.0)])
        _manager(annotated).apply_style(aid, fill="#00FF00", fill_alpha=60)

        brush = _item(annotated, aid)._fill_brush
        assert brush is not None
        assert brush.color().name() == "#00ff00"
        assert brush.color().alpha() == 60

    def test_paint_survives_the_fill_path(self, annotated):
        """填充的 save/restore 必须成对：漏了 restore，后面的描边会被二次缩放成色块

        这里只做"能画完不炸"的冒烟；几何正确性在离屏环境里无法断言像素。
        """
        aid = _make_polyline(annotated, [(1.0, 1.0), (5.0, 9.0), (9.0, 2.0)])
        item = _item(annotated, aid)
        _manager(annotated).apply_style(aid, fill="#00FF00")

        pixmap = QPixmap(120, 120)
        pixmap.fill()
        painter = QPainter(pixmap)
        try:
            item.paint(painter, None, None)
        finally:
            painter.end()

    def test_fill_path_is_closed_even_though_stroke_is_open(self, annotated):
        """折线是"开口描边 + 闭合填充"：shape() 自己会把首尾连起来，填充直接用它"""
        aid = _make_polyline(annotated, [(1.0, 1.0), (5.0, 9.0), (9.0, 2.0)])
        item = _item(annotated, aid)
        assert item.closed is False

        path = item.shape()
        count = path.elementCount()
        assert count == 4, f"三个折点应得到 moveTo + 3 个 lineTo，实得 {count}"
        assert (path.elementAt(0).x, path.elementAt(0).y) == pytest.approx(
            (path.elementAt(count - 1).x, path.elementAt(count - 1).y)
        )


# ---------------------------------------------------------------------------
# 模板往返与资源回收
# ---------------------------------------------------------------------------

class TestRoundTripAndLifecycle:
    def test_dump_load_keeps_every_point(self, annotated):
        manager = _manager(annotated)
        first = _make_polyline(annotated, [(1.0, 1.0), (2.0, 5.0), (3.0, 2.0), (4.0, 6.0)])
        _make_polyline(annotated, [(7.0, 7.0), (9.0, 9.0)])
        assert manager.count() == 2
        payload = manager.dump()

        manager.clear()
        assert manager.load(payload) == 2
        assert manager.dump() == payload, "往返不收敛，模板会越存越偏"
        restored = manager.get(first)
        assert restored is not None and restored.kind == "polyline"
        assert restored.points == pytest.approx([(1.0, 1.0), (2.0, 5.0), (3.0, 2.0), (4.0, 6.0)])

    def test_over_cap_points_are_truncated_on_load(self, annotated):
        manager = _manager(annotated)
        points = [[float(i), float(i % 5)] for i in range(MAX_POLYLINE_POINTS + 30)]
        payload = [{"id": "abcdef01", "kind": "polyline", "points": points}]

        assert manager.load(payload) == 1
        assert len(manager.get("abcdef01").points) == MAX_POLYLINE_POINTS

    def test_load_into_a_fresh_widget_reproduces_the_dump(self, plot_factory, qapp, annotated):
        _make_polyline(annotated, [(1.0, 1.0), (3.0, 4.0), (6.0, 2.0)])
        _make_polyline(annotated, [(3.0, 3.0), (7.0, 7.0)])
        payload = _manager(annotated).dump()

        df = pd.DataFrame({"a": [float(i) for i in range(100)]})
        fresh = plot_factory(df)
        fresh.resize(800, 600)
        fresh.show()
        qapp.processEvents()

        assert fresh.annotation_manager.load(payload) == len(payload)
        assert fresh.annotation_manager.dump() == payload
        for record in fresh.annotation_manager._graphics.values():
            assert isinstance(record.item, pg.PolyLineROI)
            assert record.item.segments, "折线段没跟着重建，画面上只剩孤立的手柄"

    def test_create_remove_cycles_leave_no_scene_items(self, annotated):
        manager = _manager(annotated)
        scene = annotated.plot_item.scene()
        baseline = len(scene.items())

        for _ in range(5):
            aid = _make_polyline(annotated, [(1.0, 1.0), (3.0, 4.0), (6.0, 2.0)])
            assert manager.remove(aid) is True

        assert len(scene.items()) == baseline, "线段/手柄没跟着容器一起回收"

    def test_clear_removes_segments_too(self, annotated):
        manager = _manager(annotated)
        scene = annotated.plot_item.scene()
        baseline = len(scene.items())

        _make_polyline(annotated, [(1.0, 1.0), (3.0, 4.0), (6.0, 2.0)])
        assert len(scene.items()) > baseline
        manager.clear()
        assert len(scene.items()) == baseline


# ---------------------------------------------------------------------------
# 浏览态不抢 ViewBox 的拖拽（零干扰里最难被看见的一层）
# ---------------------------------------------------------------------------

def _rate_limit_gap() -> None:
    """GraphicsScene 有 ``mouseRateLimit``（默认 200/s = 5ms）：两次 move 挨得太近，
    后一次会被整个丢掉。真实事件循环里鼠标移动天然有间隔，这里补上。
    """
    time.sleep(0.02)


def _hover_then_drag(pw, qapp) -> object:
    """先 hover 再 press+move，返回这一串手势最终落到谁身上

    必须补 hover：GraphicsScene 在 hover 阶段收集 ``dragItems``，一旦有图元登记了
    左键，``sendDragEvent`` 走 init 分支会**直接用登记的图元当 dragItem，不再回退
    去问 ViewBox** —— 视图就平不动了。只发 press+move 测不到这条路径。
    """
    scene = pw.plot_item.scene()
    left, none = Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton
    plain = Qt.KeyboardModifier.NoModifier

    _rate_limit_gap()
    pw.mouseMoveEvent(mouse_event(QEvent.Type.MouseMove, pw, _LANDING, none, none, plain))
    qapp.processEvents()

    scene.dragItem = None
    scene.dragButtons = []
    _rate_limit_gap()
    pw.mousePressEvent(mouse_event(QEvent.Type.MouseButtonPress, pw, _LANDING, left, left, plain))
    qapp.processEvents()
    pw.mouseMoveEvent(mouse_event(QEvent.Type.MouseMove, pw, _DRAG_TO, none, left, plain))
    qapp.processEvents()

    dragged = scene.dragItem
    pw.mouseReleaseEvent(
        mouse_event(QEvent.Type.MouseButtonRelease, pw, _DRAG_TO, left, none, plain)
    )
    qapp.processEvents()
    return dragged


def _reset_view(pw, qapp) -> tuple[float, float, float, float]:
    pw.view_box.setXRange(0, 100, padding=0)
    pw.view_box.setYRange(0, 100, padding=0)
    qapp.processEvents()
    return _view_range(pw)


class TestPanParity:
    """折线的线段是唯一会"替 ViewBox 接走拖拽"的子图元，必须钉死它的边界"""

    def test_clean_landing_point(self, annotated, qapp):
        """前置条件：落点确实在段上、且手柄离得够开（否则下面的断言全是空转）"""
        dx, dy = _landing_data_point(annotated)
        aid = _make_polyline(
            annotated, [(dx - 25.0, dy + 20.0), (dx - 20.0, dy), (dx + 20.0, dy)]
        )
        item = _item(annotated, aid)
        landing_scene = annotated.mapToScene(_LANDING)
        segment = item.segments[1]

        assert segment.shape().contains(segment.mapFromScene(landing_scene)), (
            "折线段没有精确穿过落点，后面的对照测不到真东西"
        )
        # 手柄离落点足够远：否则命中半径会把折点也拉进候选，拖拽归属就不再是
        # "线段行不行"这一个变量了
        for handle in item.handles:
            offset = handle["item"].scenePos() - landing_scene
            assert math.hypot(offset.x(), offset.y()) > 50.0, "手柄离落点太近"

    def test_browse_mode_pan_is_bit_identical(self, annotated, qapp):
        """核心断言：折线穿过落点时，拖拽仍归 ViewBox，视图位移与无标注时逐位相同"""
        dx, dy = _landing_data_point(annotated)
        _make_polyline(annotated, [(dx - 25.0, dy + 20.0), (dx - 20.0, dy), (dx + 20.0, dy)])
        _manager(annotated).set_edit_mode(False)
        qapp.processEvents()

        decorated_before = _reset_view(annotated, qapp)
        decorated_drag = _hover_then_drag(annotated, qapp)
        decorated_after = _view_range(annotated)

        assert decorated_drag is annotated.view_box, (
            f"浏览态下拖拽被 {decorated_drag} 接走了（线段抢走平移 = 视图平不动）"
        )

        # 对照：同一只 widget、同一串手势，只是把折线清掉
        _manager(annotated).clear()
        qapp.processEvents()
        clean_before = _reset_view(annotated, qapp)
        clean_drag = _hover_then_drag(annotated, qapp)
        clean_after = _view_range(annotated)

        assert clean_before == decorated_before
        assert clean_after != clean_before, "拖拽本身没生效，对照失去意义"
        assert clean_drag is annotated.view_box
        assert decorated_after == clean_after, (
            f"折线改变了视图平移结果: 有标注 {decorated_after} / 无标注 {clean_after}"
        )

    def test_edit_mode_segment_takes_the_drag_by_design(self, annotated, qapp):
        """对照组：编辑态在线段上拖，本该移动折线而不是平移视图

        证明上一条不是恒真 —— 线段确实有能力抢走拖拽，只是浏览态被门控挡住了。
        """
        dx, dy = _landing_data_point(annotated)
        aid = _make_polyline(
            annotated, [(dx - 25.0, dy + 20.0), (dx - 20.0, dy), (dx + 20.0, dy)]
        )
        _manager(annotated).set_edit_mode(True)
        qapp.processEvents()

        before = _reset_view(annotated, qapp)
        dragged = _hover_then_drag(annotated, qapp)

        assert dragged is not annotated.view_box, "编辑态线段没接走拖拽，上一条断言可能恒真"
        assert isinstance(dragged, pg.LineSegmentROI)
        assert _view_range(annotated)[:2] == before[:2], "编辑态拖折线不该平移视图"

    def test_leaving_edit_mode_restores_pan(self, annotated, qapp):
        """退出编辑态必须把平移还给 ViewBox（门控全量重刷的收口）"""
        dx, dy = _landing_data_point(annotated)
        _make_polyline(annotated, [(dx - 25.0, dy + 20.0), (dx - 20.0, dy), (dx + 20.0, dy)])

        _manager(annotated).set_edit_mode(True)
        qapp.processEvents()
        _reset_view(annotated, qapp)
        assert _hover_then_drag(annotated, qapp) is not annotated.view_box

        _manager(annotated).set_edit_mode(False)
        qapp.processEvents()
        _reset_view(annotated, qapp)
        assert _hover_then_drag(annotated, qapp) is annotated.view_box, "退出编辑态后视图仍平不动"
