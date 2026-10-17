"""AnnotationManager component 测试（offscreen）。

覆盖管理器的完整公共接口：新建 / 删除 / 清空 / 装载 / 导出 / 编辑模式 /
显隐 / 选中 / 样式 / 资源回收 / 销毁期安全。

上游（数据模型）见 tests/unit/core/test_annotation_models.py；
"不干扰既有功能"的对照测试见 tests/component/test_annotation_no_interference.py。

**刻意不在本文件里清图后重绘**：裸 widget 上 ``clear_plot_item()`` 后立刻
重新绘图会触发项目既有的原生崩溃，与本功能无关，最小复现见
``tmp/repro_clear_replot.py``（该崩溃在完全绕过标注代码的对照实验里同样
出现）。
"""

from __future__ import annotations

import pandas as pd
import pytest

from PySide6.QtCore import QPoint, QPointF, Qt

from src.core.annotation_models import (
    DEFAULT_STROKE,
    FONT_SIZE_RANGE,
    MAX_ANNOTATIONS_PER_PLOT,
    STROKE_WIDTH_RANGE,
    Z_ANNOTATION,
    Z_ANNOTATION_SELECTED,
)
from src.ui.widgets import annotation_manager as am_module
from src.ui.widgets.annotation_items import read_geometry
from tests.component._viewbox_events import mouse_event

P1_KINDS = ("text", "rect", "ellipse", "line", "arrow")
#: 图形层能渲染的全部类型（折线在 P3 补齐）
ALL_KINDS = P1_KINDS + ("polyline",)


@pytest.fixture()
def annotated(plot_factory, qapp):
    """已绘图并固定视图范围的 widget，附带便利方法"""
    df = pd.DataFrame({"a": [float(i) for i in range(100)]})
    pw = plot_factory(df)
    pw.resize(800, 600)
    pw.window().show()
    pw.show()
    assert pw.plot_variable("a"), "曲线未绘出，几何断言失去意义"
    pw.view_box.setXRange(0, 100, padding=0)
    pw.view_box.setYRange(-10, 110, padding=0)
    qapp.processEvents()
    return pw


@pytest.fixture()
def change_counter(annotated):
    """记录 annotations_changed 的发射次数"""
    calls = []
    annotated.annotations_changed.connect(lambda: calls.append(1))
    return calls


def _scene_item_count(pw) -> int:
    return len(pw.plot_item.scene().items())


# ---------------------------------------------------------------------------
# 新建 / 删除 / 计数
# ---------------------------------------------------------------------------

class TestCreateAndRemove:
    def test_create_each_supported_kind(self, annotated):
        manager = annotated.annotation_manager
        for kind in ALL_KINDS:
            assert manager.create(kind, [(10.0, 10.0), (20.0, 20.0)]) is not None, kind
        assert manager.count() == len(ALL_KINDS)

    def test_create_returns_usable_id_looked_up_by_get(self, annotated):
        manager = annotated.annotation_manager
        aid = manager.create("rect", [(1.0, 1.0), (2.0, 2.0)])
        model = manager.get(aid)
        assert model is not None and model.id == aid and model.kind == "rect"

    @pytest.mark.parametrize("kind", ["hexagon", "", "arrowhead", "Text", None])
    def test_unknown_kind_is_rejected(self, annotated, kind):
        manager = annotated.annotation_manager
        assert manager.create(kind, [(0.0, 0.0), (1.0, 1.0)]) is None
        assert manager.count() == 0

    @pytest.mark.parametrize("points", [[], [(0.0, 0.0)], [("x", "y"), (1.0, 1.0)]])
    def test_invalid_geometry_is_rejected(self, annotated, points):
        manager = annotated.annotation_manager
        assert manager.create("rect", points) is None
        assert manager.count() == 0

    def test_polyline_is_rendered(self, annotated):
        """折线不再是"模型容许、图形丢弃"：建一条就该在场景里看得见

        P1 时期这里返回 None，因为图元还没实现。现在必须真画出来 ——
        返回 id 却画不出来会留下一条看不见也删不掉的标注。
        """
        import pyqtgraph as pg

        manager = annotated.annotation_manager
        aid = manager.create("polyline", [(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)])
        assert aid is not None
        assert manager.count() == 1

        item = manager._graphics[aid].item
        assert isinstance(item, pg.PolyLineROI)
        assert len(item.handles) == 3, "三个折点应有三只手柄"
        assert len(item.segments) == 2, "三个折点应有两段线段"
        assert item.isVisible() is True

    def test_remove_returns_false_for_unknown_id(self, annotated):
        assert annotated.annotation_manager.remove("nope") is False

    def test_remove_drops_from_count_and_scene(self, annotated):
        manager = annotated.annotation_manager
        baseline = _scene_item_count(annotated)
        aid = manager.create("rect", [(10.0, 10.0), (20.0, 20.0)])
        assert _scene_item_count(annotated) > baseline

        assert manager.remove(aid) is True
        assert manager.count() == 0
        assert _scene_item_count(annotated) == baseline

    def test_remove_clears_selection_if_it_was_selected(self, annotated):
        manager = annotated.annotation_manager
        aid = manager.create("rect", [(10.0, 10.0), (20.0, 20.0)])
        manager.select(aid)
        assert manager.selected_id == aid

        manager.remove(aid)
        assert manager.selected_id is None

    def test_clear_returns_removed_count_and_is_idempotent(self, annotated):
        manager = annotated.annotation_manager
        for kind in ALL_KINDS:
            manager.create(kind, [(10.0, 10.0), (20.0, 20.0)])
        assert manager.clear() == len(ALL_KINDS)
        assert manager.clear() == 0

    def test_clear_resets_selection(self, annotated):
        manager = annotated.annotation_manager
        aid = manager.create("rect", [(10.0, 10.0), (20.0, 20.0)])
        manager.select(aid)
        manager.clear()
        assert manager.selected_id is None

    def test_arrow_creates_a_separate_head_item(self, annotated):
        manager = annotated.annotation_manager
        arrow = manager.create("arrow", [(10.0, 10.0), (40.0, 40.0)])
        plain = manager.create("line", [(10.0, 10.0), (40.0, 40.0)])
        assert manager._graphics[arrow].decor is not None, "箭头缺头部图元"
        assert manager._graphics[plain].decor is None, "直线不该有头部图元"

    def test_count_cap_rejects_further_creates(self, annotated, monkeypatch):
        """上限按常量收口（真实值 200，这里替身成 3 以免测试变慢）"""
        assert MAX_ANNOTATIONS_PER_PLOT == 200
        monkeypatch.setattr(am_module, "MAX_ANNOTATIONS_PER_PLOT", 3)
        manager = annotated.annotation_manager

        for _ in range(3):
            assert manager.create("rect", [(1.0, 1.0), (2.0, 2.0)]) is not None
        assert manager.create("rect", [(1.0, 1.0), (2.0, 2.0)]) is None
        assert manager.count() == 3

    def test_cap_announces_to_window(self, annotated, monkeypatch):
        monkeypatch.setattr(am_module, "MAX_ANNOTATIONS_PER_PLOT", 1)
        manager = annotated.annotation_manager
        messages = []
        annotated.window()._broadcast = messages.append

        manager.create("rect", [(1.0, 1.0), (2.0, 2.0)])
        manager.create("rect", [(1.0, 1.0), (2.0, 2.0)])

        assert any("上限" in m for m in messages), f"未播报上限: {messages}"


# ---------------------------------------------------------------------------
# 右键落点处新建
# ---------------------------------------------------------------------------

class TestCreateAtContext:
    def test_uses_recorded_context_point(self, annotated):
        manager = annotated.annotation_manager
        view_box = annotated.view_box
        view_box.context_x, view_box.context_y = 25.0, 75.0

        aid = manager.create_at_context("text")
        (x, y), = manager.get(aid).points
        assert (x, y) == pytest.approx((25.0, 75.0), abs=1e-9)

    def test_shape_is_centred_on_context_point(self, annotated):
        manager = annotated.annotation_manager
        view_box = annotated.view_box
        view_box.context_x, view_box.context_y = 25.0, 75.0

        aid = manager.create_at_context("rect")
        (x0, y0), (x1, y1) = manager.get(aid).points
        assert (x0 + x1) / 2 == pytest.approx(25.0, abs=1e-6)
        assert (y0 + y1) / 2 == pytest.approx(75.0, abs=1e-6)
        # 边长取当前视图跨度的 1/4（半宽 = 跨度 / 8）
        x_span = view_box.viewRange()[0][1] - view_box.viewRange()[0][0]
        assert (x1 - x0) == pytest.approx(x_span / 4.0, rel=1e-6)

    def test_falls_back_to_view_centre_without_context_point(self, annotated):
        """没走过 getMenu（直接调接口 / 键盘触发）时用视图中心兜底"""
        manager = annotated.annotation_manager
        view_box = annotated.view_box
        for attr in ("context_x", "context_y"):
            if hasattr(view_box, attr):
                delattr(view_box, attr)

        aid = manager.create_at_context("text")
        (x, y), = manager.get(aid).points
        x_range, y_range = view_box.viewRange()
        assert x == pytest.approx((x_range[0] + x_range[1]) / 2, abs=1e-9)
        assert y == pytest.approx((y_range[0] + y_range[1]) / 2, abs=1e-9)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), None, "abc"])
    def test_non_finite_context_point_falls_back(self, annotated, bad):
        manager = annotated.annotation_manager
        view_box = annotated.view_box
        view_box.context_x, view_box.context_y = bad, bad

        aid = manager.create_at_context("text")
        (x, y), = manager.get(aid).points
        x_range, y_range = view_box.viewRange()
        assert x == pytest.approx((x_range[0] + x_range[1]) / 2, abs=1e-9)
        assert y == pytest.approx((y_range[0] + y_range[1]) / 2, abs=1e-9)

    def test_every_kind_created_at_context(self, annotated):
        manager = annotated.annotation_manager
        annotated.view_box.context_x = 50.0
        annotated.view_box.context_y = 50.0
        for kind in ALL_KINDS:
            assert manager.create_at_context(kind) is not None, kind

    def test_polyline_context_geometry_is_a_zigzag(self, annotated):
        """折线在落点处给的不是"两点直线"（那用户还得自己拉第三个点）

        形状无对错，但必须真的跨过落点两侧：否则用户以为点空了。
        """
        manager = annotated.annotation_manager
        annotated.view_box.context_x = 50.0
        annotated.view_box.context_y = 50.0

        points = manager.get(manager.create_at_context("polyline")).points
        assert len(points) == 4
        xs = [p[0] for p in points]
        assert min(xs) < 50.0 < max(xs), f"折线没跨过落点: {points}"
        ys = [p[1] for p in points]
        assert min(ys) < 50.0 < max(ys), f"折线没跨过落点: {points}"

    def test_returns_none_when_view_box_unavailable(self, annotated):
        """没有 ViewBox（widget 进入销毁期）时安静返回，不抛异常"""
        annotated._is_being_destroyed = True
        try:
            assert annotated.annotation_manager.create_at_context("rect") is None
        finally:
            annotated._is_being_destroyed = False


# ---------------------------------------------------------------------------
# 序列化：dump / load
# ---------------------------------------------------------------------------

class TestDumpAndLoad:
    def _fill(self, manager):
        manager.create("text", [(1.0, 2.0)], text="合格")
        manager.create("rect", [(0.0, 0.0), (5.0, 5.0)], fill="#00FF00")
        manager.create("arrow", [(0.0, 0.0), (5.0, 5.0)])
        return manager.dump()

    def test_dump_is_idempotent_and_order_stable(self, annotated):
        manager = annotated.annotation_manager
        first = self._fill(manager)
        assert manager.dump() == first
        assert [d["kind"] for d in first] == ["text", "rect", "arrow"]

    def test_load_replace_true_clears_first(self, annotated):
        manager = annotated.annotation_manager
        manager.create("text", [(99.0, 99.0)], text="旧")
        payload = self._fill(manager)
        manager.create("text", [(88.0, 88.0)], text="待清")

        assert manager.load(payload, replace=True) == len(payload)
        assert manager.count() == len(payload)

    def test_load_replace_false_appends(self, annotated):
        manager = annotated.annotation_manager
        payload = self._fill(manager)
        before = manager.count()

        assert manager.load(payload, replace=False) == len(payload)
        assert manager.count() == before + len(payload)

    def test_load_into_fresh_widget_reproduces_dump(self, plot_factory, qapp, annotated):
        payload = self._fill(annotated.annotation_manager)

        df = pd.DataFrame({"a": [float(i) for i in range(100)]})
        fresh = plot_factory(df)
        fresh.resize(800, 600)
        fresh.show()
        qapp.processEvents()

        assert fresh.annotation_manager.load(payload) == len(payload)
        assert fresh.annotation_manager.dump() == payload

    def test_load_skips_bad_entries_without_losing_good_ones(self, annotated):
        """模板是用户手工攒的资产：一条坏记录只丢它自己，不牵连整份"""
        manager = annotated.annotation_manager
        good = {"id": "aa11", "kind": "rect", "points": [[0.0, 0.0], [1.0, 1.0]]}
        payload = [
            good,
            {"kind": "hexagon", "points": [[0.0, 0.0]]},          # 未知类型
            {"kind": "rect", "points": [[0.0, "x"]]},             # 坐标非数
            {"kind": "rect"},                                     # 缺 points
            "not-a-dict",
            {"id": "bb22", "kind": "ellipse", "points": [[2.0, 2.0], [3.0, 3.0]]},
        ]
        assert manager.load(payload) == 2
        kinds = {m["kind"] for m in manager.dump()}
        assert kinds == {"rect", "ellipse"}

    def test_load_reassigns_duplicate_ids(self, annotated):
        """同一份配置贴到两个子图 / 同一份 data 里 id 重复：必须改派新 id

        否则图形表会用后一条覆盖前一条，用户看见"少了一条标注"。
        """
        manager = annotated.annotation_manager
        dup = {"id": "sameid00", "kind": "rect", "points": [[0.0, 0.0], [1.0, 1.0]]}
        assert manager.load([dup, dict(dup)]) == 2
        ids = [m["id"] for m in manager.dump()]
        assert len(set(ids)) == 2, f"id 冲突未处理: {ids}"

    def test_load_non_iterable_payload_is_noop(self, annotated):
        manager = annotated.annotation_manager
        assert manager.load(None) == 0
        assert manager.load([]) == 0
        assert manager.load("garbage") == 0

    def test_load_respects_cap(self, annotated, monkeypatch):
        monkeypatch.setattr(am_module, "MAX_ANNOTATIONS_PER_PLOT", 2)
        manager = annotated.annotation_manager
        payload = [
            {"id": f"id{i:06d}", "kind": "rect", "points": [[0.0, 0.0], [1.0, 1.0]]}
            for i in range(5)
        ]
        assert manager.load(payload) == 2
        assert manager.count() == 2


# ---------------------------------------------------------------------------
# 编辑模式 / 显隐 / 选中 / 样式
# ---------------------------------------------------------------------------

class TestEditMode:
    def test_defaults_to_off(self, annotated):
        assert annotated.annotation_manager.edit_mode is False

    def test_new_item_inherits_current_mode(self, annotated):
        manager = annotated.annotation_manager
        manager.set_edit_mode(True)

        aid = manager.create("rect", [(1.0, 1.0), (5.0, 5.0)])
        assert manager._graphics[aid].item._editable is True
        assert manager._graphics[aid].item.acceptedMouseButtons() == Qt.MouseButton.LeftButton

    def test_toggling_propagates_to_existing_items(self, annotated):
        manager = annotated.annotation_manager
        ids = [manager.create(k, [(1.0, 1.0), (5.0, 5.0)]) for k in ALL_KINDS]
        assert all(ids), "有类型没建出来，断言会漏掉它"

        manager.set_edit_mode(True)
        for aid in ids:
            assert manager._graphics[aid].item._editable is True

        manager.set_edit_mode(False)
        for aid in ids:
            assert manager._graphics[aid].item._editable is False
            assert manager._graphics[aid].item.acceptedMouseButtons() == Qt.MouseButton.NoButton

    def test_leaving_edit_mode_clears_selection(self, annotated):
        manager = annotated.annotation_manager
        aid = manager.create("rect", [(1.0, 1.0), (5.0, 5.0)])
        manager.set_edit_mode(True)
        manager.select(aid)

        manager.set_edit_mode(False)
        assert manager.selected_id is None

    def test_repeated_set_is_a_noop(self, annotated, change_counter):
        manager = annotated.annotation_manager
        manager.set_edit_mode(False)
        manager.set_edit_mode(False)
        assert change_counter == []


class TestVisibilityAndSelection:
    def test_set_visible_returns_false_for_unknown_id(self, annotated):
        assert annotated.annotation_manager.set_visible("nope", False) is False

    def test_set_visible_hides_item_and_model(self, annotated):
        manager = annotated.annotation_manager
        aid = manager.create("rect", [(1.0, 1.0), (5.0, 5.0)])

        assert manager.set_visible(aid, False) is True
        assert manager.get(aid).visible is False
        assert manager._graphics[aid].item.isVisible() is False

        assert manager.set_visible(aid, True) is True
        assert manager.get(aid).visible is True
        assert manager._graphics[aid].item.isVisible() is True

    def test_set_visible_also_toggles_arrow_head(self, annotated):
        manager = annotated.annotation_manager
        aid = manager.create("arrow", [(1.0, 1.0), (5.0, 5.0)])

        manager.set_visible(aid, False)
        assert manager._graphics[aid].decor.isVisible() is False

    def test_select_raises_z_and_deselect_restores(self, annotated):
        manager = annotated.annotation_manager
        aid = manager.create("rect", [(1.0, 1.0), (5.0, 5.0)])
        item = manager._graphics[aid].item
        assert item.zValue() == Z_ANNOTATION

        manager.select(aid)
        assert manager.selected_id == aid
        assert item.zValue() == Z_ANNOTATION_SELECTED

        manager.select(None)
        assert manager.selected_id is None
        assert item.zValue() == Z_ANNOTATION

    def test_select_unknown_id_is_treated_as_deselect(self, annotated):
        manager = annotated.annotation_manager
        aid = manager.create("rect", [(1.0, 1.0), (5.0, 5.0)])
        manager.select(aid)

        manager.select("nope")
        assert manager.selected_id is None
        assert manager._graphics[aid].item.zValue() == Z_ANNOTATION

    def test_selecting_another_moves_the_highlight(self, annotated):
        manager = annotated.annotation_manager
        a = manager.create("rect", [(1.0, 1.0), (5.0, 5.0)])
        b = manager.create("rect", [(20.0, 20.0), (25.0, 25.0)])

        manager.select(a)
        manager.select(b)
        assert manager._graphics[a].item.zValue() == Z_ANNOTATION
        assert manager._graphics[b].item.zValue() == Z_ANNOTATION_SELECTED


class TestApplyStyle:
    def test_unknown_id_returns_false(self, annotated):
        assert annotated.annotation_manager.apply_style("nope", stroke="#000000") is False

    def test_updates_stroke_and_width(self, annotated):
        manager = annotated.annotation_manager
        aid = manager.create("rect", [(1.0, 1.0), (5.0, 5.0)])

        assert manager.apply_style(aid, stroke="#123456", stroke_width=4) is True
        assert manager.get(aid).stroke == "#123456"
        assert manager.get(aid).stroke_width == 4

    def test_values_are_clamped_through_the_model(self, annotated):
        """改样式必须过一遍模型校验，不能直接 setattr"""
        manager = annotated.annotation_manager
        aid = manager.create("rect", [(1.0, 1.0), (5.0, 5.0)])

        manager.apply_style(aid, stroke_width=999, font_size=1, stroke="notacolor")
        model = manager.get(aid)
        assert model.stroke_width == STROKE_WIDTH_RANGE[1]
        assert model.font_size == FONT_SIZE_RANGE[0]
        assert model.stroke == DEFAULT_STROKE

    def test_unknown_keys_are_ignored(self, annotated):
        manager = annotated.annotation_manager
        aid = manager.create("rect", [(1.0, 1.0), (5.0, 5.0)])

        assert manager.apply_style(aid, bogus_field=42) is True
        assert not hasattr(manager.get(aid), "bogus_field")

    def test_geometry_and_kind_cannot_be_rewritten_via_style(self, annotated):
        """几何/类型不该能从 apply_style 溜进来

        本方法不写图元几何，改了模型就会留下"模型说在这、屏幕画在那"的分裂
        状态；kind 改了还会让已建好的图元类型对不上（矩形换椭圆不会变形状）。
        几何只有两条合法入口：用户在图上拖，和 load()。
        """
        manager = annotated.annotation_manager
        aid = manager.create("rect", [(1.0, 1.0), (5.0, 5.0)])
        before = list(manager.get(aid).points)

        manager.apply_style(aid, points=[(9.0, 9.0), (9.5, 9.5)])
        assert manager.get(aid).points == before
        assert manager.get(aid).kind == "rect"
        # 图元几何也必须还在原处（模型与画面不能分裂）—— 按应用回读的口径比对
        assert read_geometry(manager._graphics[aid].item) == before

    def test_text_update_reaches_the_item(self, annotated):
        manager = annotated.annotation_manager
        aid = manager.create("text", [(1.0, 1.0)], text="旧文案")

        manager.apply_style(aid, text="新文案")
        assert manager.get(aid).text == "新文案"
        assert manager._graphics[aid].item.toPlainText() == "新文案"

    def test_text_is_truncated_to_the_model_limit(self, annotated):
        from src.core.annotation_models import MAX_TEXT_LENGTH

        manager = annotated.annotation_manager
        aid = manager.create("text", [(1.0, 1.0)], text="x")

        manager.apply_style(aid, text="字" * (MAX_TEXT_LENGTH + 100))
        assert len(manager.get(aid).text) == MAX_TEXT_LENGTH

    def test_style_change_emits_once(self, annotated, change_counter):
        manager = annotated.annotation_manager
        aid = manager.create("rect", [(1.0, 1.0), (5.0, 5.0)])
        change_counter.clear()

        manager.apply_style(aid, stroke_width=3)
        assert len(change_counter) == 1


# ---------------------------------------------------------------------------
# 变更播报
# ---------------------------------------------------------------------------

class TestChangeSignal:
    def test_create_remove_clear_emit(self, annotated, change_counter):
        manager = annotated.annotation_manager
        aid = manager.create("rect", [(1.0, 1.0), (5.0, 5.0)])
        assert len(change_counter) == 1

        manager.remove(aid)
        assert len(change_counter) == 2

        manager.create("rect", [(1.0, 1.0), (5.0, 5.0)])
        manager.clear()
        assert len(change_counter) == 4

    def test_load_emits_exactly_once_regardless_of_size(self, annotated, change_counter):
        """批量装载只该播一次，否则列表对话框会被逐条重绘上百次"""
        manager = annotated.annotation_manager
        payload = [
            {"id": f"id{i:06d}", "kind": "rect", "points": [[0.0, 0.0], [1.0, 1.0]]}
            for i in range(30)
        ]
        manager.load(payload)
        assert len(change_counter) == 1

    def test_failed_create_does_not_emit(self, annotated, change_counter):
        manager = annotated.annotation_manager
        manager.create("hexagon", [(0.0, 0.0), (1.0, 1.0)])
        assert change_counter == []

    def test_set_visible_emits(self, annotated, change_counter):
        manager = annotated.annotation_manager
        aid = manager.create("rect", [(1.0, 1.0), (5.0, 5.0)])
        change_counter.clear()

        manager.set_visible(aid, False)
        assert len(change_counter) == 1


# ---------------------------------------------------------------------------
# 资源回收与销毁期安全
# ---------------------------------------------------------------------------

class TestLifecycleSafety:
    def test_create_remove_cycles_leave_no_leftovers(self, annotated):
        manager = annotated.annotation_manager
        baseline_items = _scene_item_count(annotated)
        baseline_added = len(annotated.view_box.addedItems)

        for _ in range(20):
            aid = manager.create("rect", [(10.0, 10.0), (20.0, 20.0)])
            manager.remove(aid)

        assert _scene_item_count(annotated) == baseline_items
        assert len(annotated.view_box.addedItems) == baseline_added

    def test_clear_removes_arrow_heads_too(self, annotated):
        manager = annotated.annotation_manager
        baseline_items = _scene_item_count(annotated)

        for _ in range(5):
            manager.create("arrow", [(1.0, 1.0), (5.0, 5.0)])
        assert _scene_item_count(annotated) > baseline_items

        manager.clear()
        assert _scene_item_count(annotated) == baseline_items, "箭头头部残留为幽灵图元"

    def test_clear_removes_items_from_viewbox_added_items(self, annotated):
        """必须走 view_box.removeItem：只调 scene().removeItem 会永久留在 addedItems"""
        manager = annotated.annotation_manager
        baseline_added = len(annotated.view_box.addedItems)

        manager.create("rect", [(1.0, 1.0), (5.0, 5.0)])
        manager.create("arrow", [(1.0, 1.0), (5.0, 5.0)])
        manager.clear()

        assert len(annotated.view_box.addedItems) == baseline_added

    def test_destroyed_widget_access_is_silent(self, annotated):
        """销毁期查/清都不该抛异常：收尾回调可能在 widget 已死之后才跑到"""
        manager = annotated.annotation_manager
        manager.create("rect", [(1.0, 1.0), (5.0, 5.0)])
        manager.set_edit_mode(True)

        annotated._is_being_destroyed = True
        try:
            assert manager._view_box() is None
            assert manager.create("rect", [(1.0, 1.0), (5.0, 5.0)]) is None
            assert manager.clear() == 1
            assert manager.count() == 0
            manager.set_edit_mode(False)
        finally:
            annotated._is_being_destroyed = False

    def test_graphics_store_lives_on_the_widget(self, annotated):
        """状态挂在 widget 上而不是 Manager 上：换布局时 widget 一起销毁重建"""
        manager = annotated.annotation_manager
        manager.create("rect", [(1.0, 1.0), (5.0, 5.0)])
        assert annotated._annotation_graphics is manager._graphics

    def test_second_manager_instance_sees_the_same_store(self, annotated):
        """同一 widget 的新 Manager 实例不该"丢"已存在的标注（换链/重建场景）"""
        manager = annotated.annotation_manager
        aid = manager.create("rect", [(1.0, 1.0), (5.0, 5.0)])

        from src.ui.widgets.annotation_manager import AnnotationManager

        other = AnnotationManager(annotated._event_handler)
        assert other.count() == 1
        assert other.get(aid) is not None


# ---------------------------------------------------------------------------
# 双击命中（hit_test）与属性对话框入口 —— 用户反馈：双击标注应直接进属性编辑
# ---------------------------------------------------------------------------

def _scene_point(pw, data_xy):
    """数据坐标 → 场景坐标（hit_test 的输入口径）"""
    return pw.view_box.mapViewToScene(QPointF(*data_xy))


def _widget_point(pw, data_xy):
    """数据坐标 → widget 局部像素（mouseDoubleClickEvent 的输入口径）"""
    return QPoint(pw.mapFromScene(_scene_point(pw, data_xy)))


class TestHitTest:
    """manager.hit_test：双击入口的判定核心，两段式（BSP 快路径 + 容差兜底）

    Qt 的几何命中是半开区间（``QRectF.contains`` 不含右/下边界），恰好压在
    矩形底边或角点上的点 BSP 与 shape 精筛**都会漏**；没有兜底的话双击
    矩形底边会落到"打开绘图变量编辑器"。
    """

    def test_text_rect_arrow_and_blank(self, annotated):
        manager = annotated.annotation_manager
        t1 = manager.create("text", [(30.0, 40.0)], text="甲")
        r1 = manager.create("rect", [(50.0, 50.0), (70.0, 70.0)])
        a1 = manager.create("arrow", [(10.0, 80.0), (20.0, 60.0)])

        assert manager.hit_test(_scene_point(annotated, (30.0, 40.0))) == t1
        assert manager.hit_test(_scene_point(annotated, (60.0, 60.0))) == r1
        # 箭杆中部（不是头部装饰）——decor 与本体的 id 相同
        assert manager.hit_test(_scene_point(annotated, (15.0, 70.0))) == a1
        assert manager.hit_test(_scene_point(annotated, (10.0, 10.0))) is None

    def test_bottom_edge_and_corner_are_not_lost(self, annotated):
        manager = annotated.annotation_manager
        r1 = manager.create("rect", [(50.0, 50.0), (70.0, 70.0)])

        assert manager.hit_test(_scene_point(annotated, (60.0, 50.0))) == r1, (
            "矩形底边中点命中失败（半开区间缺口未被兜底接住）"
        )
        assert manager.hit_test(_scene_point(annotated, (50.0, 50.0))) == r1

    def test_polyline_segment_and_target(self, annotated):
        manager = annotated.annotation_manager
        p1 = manager.create("polyline", [(10.0, 20.0), (30.0, 60.0), (60.0, 30.0)])
        t1 = manager.create("target", [(80.0, 50.0)], text="靶")

        assert manager.hit_test(_scene_point(annotated, (45.0, 45.0))) == p1
        assert manager.hit_test(_scene_point(annotated, (80.0, 50.0))) == t1

    def test_edit_mode_handle_spot_hits_the_shape(self, annotated):
        """编辑态手柄恰好压在矩形角点上，双击角点也要命中"""
        manager = annotated.annotation_manager
        r1 = manager.create("rect", [(50.0, 50.0), (70.0, 70.0)])
        manager.set_edit_mode(True)

        assert manager.hit_test(_scene_point(annotated, (50.0, 50.0))) == r1

    def test_unknown_scene_object_types_are_ignored(self, annotated):
        """曲线等非标注图元不参与命中（hit 只认 annotation_id）"""
        manager = annotated.annotation_manager
        assert manager.hit_test(_scene_point(annotated, (50.0, 50.0))) is None


class TestDoubleClickEntry:
    """widget 层双击分流：标注 → 属性对话框；空白 → 变量编辑器（原行为）"""

    def _dblclick(self, pw, data_xy):
        from PySide6.QtCore import QEvent

        event = mouse_event(
            QEvent.Type.MouseButtonDblClick, pw, _widget_point(pw, data_xy),
            Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        pw.mouseDoubleClickEvent(event)

    def test_double_click_on_annotation_opens_property_dialog(
        self, annotated, monkeypatch
    ):
        manager = annotated.annotation_manager
        r1 = manager.create("rect", [(50.0, 50.0), (70.0, 70.0)])
        opened, editors = [], []
        monkeypatch.setattr(
            manager, "open_property_dialog", lambda aid: opened.append(aid) or True
        )
        monkeypatch.setattr(annotated, "open_variable_editor", lambda: editors.append(1))

        self._dblclick(annotated, (60.0, 60.0))

        assert opened == [r1], "双击标注应打开该标注的属性对话框"
        assert editors == [], "双击标注不应再打开绘图变量编辑器"
        assert manager.selected_id == r1, "打开属性前应先选中（背后可见高亮）"

    def test_double_click_on_blank_still_opens_variable_editor(
        self, annotated, monkeypatch
    ):
        manager = annotated.annotation_manager
        opened, editors = [], []
        monkeypatch.setattr(
            manager, "open_property_dialog", lambda aid: opened.append(aid) or True
        )
        monkeypatch.setattr(annotated, "open_variable_editor", lambda: editors.append(1))

        self._dblclick(annotated, (10.0, 10.0))

        assert opened == []
        assert editors == [1], "双击空白处的原行为（变量编辑器）不能被改变"

    def test_double_click_works_outside_edit_mode(self, annotated, monkeypatch):
        """浏览态（图元被门控成 NoButton）双击标注同样进属性编辑"""
        manager = annotated.annotation_manager
        assert manager.edit_mode is False
        r1 = manager.create("rect", [(50.0, 50.0), (70.0, 70.0)])
        opened = []
        monkeypatch.setattr(
            manager, "open_property_dialog", lambda aid: opened.append(aid) or True
        )

        self._dblclick(annotated, (60.0, 60.0))
        assert opened == [r1]


# ---------------------------------------------------------------------------
# 文字外边框的 0 线宽 —— 用户反馈：希望文字可以无边框
# ---------------------------------------------------------------------------

class TestTextBorder:
    def test_zero_width_means_no_border(self, annotated):
        """线宽 0 = 无外边框：必须 NoPen —— Qt 里 width=0 的笔是 cosmetic 1px"""
        manager = annotated.annotation_manager
        aid = manager.create("text", [(20.0, 20.0)], text="无框")

        manager.apply_style(aid, stroke_width=0)

        item = manager._graphics[aid].item
        assert item.border.style() == Qt.PenStyle.NoPen
        assert manager.get(aid).stroke_width == 0

    def test_positive_width_keeps_the_border(self, annotated):
        manager = annotated.annotation_manager
        aid = manager.create("text", [(20.0, 20.0)], text="有框")

        manager.apply_style(aid, stroke_width=2)

        item = manager._graphics[aid].item
        assert item.border.style() == Qt.PenStyle.SolidLine
        assert item.border.width() == 2

    def test_zero_width_survives_dump_load_roundtrip(self, annotated):
        """粘贴 / 撤销 / 模板都走 dump → load，无边框语义不能在半路丢掉"""
        manager = annotated.annotation_manager
        aid = manager.create("text", [(20.0, 20.0)], text="无框")
        manager.apply_style(aid, stroke_width=0)

        assert manager.load(manager.dump(), replace=True) == 1

        item = manager._graphics[aid].item
        assert manager.get(aid).stroke_width == 0
        assert item.border.style() == Qt.PenStyle.NoPen

    def test_non_text_kinds_clamp_zero_back_to_one_pixel(self, annotated):
        """矩形/直线等宽 0 会隐形：模型保留 0，图元层钳回 1px"""
        manager = annotated.annotation_manager
        rid = manager.create("rect", [(10.0, 30.0), (20.0, 40.0)])

        manager.apply_style(rid, stroke_width=0)

        assert manager.get(rid).stroke_width == 0
        assert manager._graphics[rid].item.pen.width() == 1
