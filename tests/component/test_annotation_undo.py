"""标注撤销 / 重做 component 测试（offscreen）。

实现是**快照式**：``_history`` 的栈顶就是"当前状态"（不是操作日志），
``undo()`` 弹一格再整份写回（走 ``load(replace=True)``），``redo()`` 反向。
这样撤销路径与装载路径**完全同源** —— 装载路径有测试守着，撤销就跟着被守住。

四组用例，按"最容易出错"排序：

1. **手势合并**：一次拖动会发几十上百次 ``sigRegionChanged``，逐次入栈的后果是
   撤销栈被一个手势冲垮（按一次 Ctrl+Z 只退回一个像素）。这是全文件最要紧的一组。
2. 增 / 删 / 改 / 模板装载各走一步，撤销后逐项还原（含**屏幕上的图元**，
   不只是模型）。
3. 栈的边界：空步不入栈、新操作清空 redo、容量上限、``reset_history``。
4. 重入守卫：撤销自己不入栈，否则 Ctrl+Z 会来回弹。

合成鼠标事件必须留 ``mouseRateLimit`` 间隔（见 ``_gap``）：默认 200/s，
两次 move 挨得太近后一次会被**整条丢掉**，表现为"某类图元拖不动"的**假故障**
（排查时踩过）。
"""

from __future__ import annotations

import time

import pandas as pd
import pytest

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt

from src.core.annotation_models import MAX_UNDO_STEPS
from tests.component._viewbox_events import mouse_event

#: 视图范围固定成 0..100，便于把像素位移换算成稳定的数据位移
_SPAN = 100.0

#: 一次拖动的像素位移（往右上：数据 x 增大、y 增大）
_DRAG_PX = (60, -40)


@pytest.fixture()
def annotated(plot_factory, qapp):
    df = pd.DataFrame({"a": [float(i) for i in range(100)]})
    pw = plot_factory(df)
    pw.resize(800, 600)
    pw.window().show()
    pw.show()
    assert pw.plot_variable("a"), "曲线未绘出，几何断言失去意义"
    pw.view_box.setXRange(0, _SPAN, padding=0)
    pw.view_box.setYRange(0, _SPAN, padding=0)
    qapp.processEvents()
    return pw


def _manager(pw):
    return pw.annotation_manager


def _history_len(pw) -> int:
    return len(_manager(pw)._history)


def _scene_items(pw) -> int:
    """场景里的图元条数：撤销必须把图元也还原回去，不能只改模型"""
    return len(pw.plot_item.scene().items())


def _gap() -> None:
    """GraphicsScene.mouseRateLimit 默认 200/s = 5ms：不留间隔，后一条 move
    会被整个丢掉。丢光 move 的手势看起来就像"这个图元拖不动"。
    """
    time.sleep(0.03)


def _pixel_of(pw, point: tuple[float, float]) -> QPoint:
    return QPoint(pw.mapFromScene(pw.view_box.mapViewToScene(QPointF(*point))))


def _drag(pw, qapp, grab: tuple[float, float], delta_px=_DRAG_PX, *, steps: int = 4) -> list[str]:
    """在数据坐标 ``grab`` 处按下并拖动 ``delta_px`` 像素

    返回每一步"这一串手势落到了谁身上"（类名）—— 只用于排障；断言一律走
    历史步数与模型几何，不依赖 `scene.dragItem`（文字图元走自己那条路，它恒为 None）。
    """
    scene = pw.plot_item.scene()
    left, none = Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton
    plain = Qt.KeyboardModifier.NoModifier
    start = _pixel_of(pw, grab)
    end = QPoint(start.x() + delta_px[0], start.y() + delta_px[1])

    _gap()
    pw.mouseMoveEvent(mouse_event(QEvent.Type.MouseMove, pw, start, none, none, plain))
    qapp.processEvents()
    scene.dragItem = None
    scene.dragButtons = []

    _gap()
    pw.mousePressEvent(mouse_event(QEvent.Type.MouseButtonPress, pw, start, left, left, plain))
    qapp.processEvents()

    owners = []
    for i in range(1, steps + 1):
        _gap()
        mid = QPoint(
            start.x() + delta_px[0] * i // steps,
            start.y() + delta_px[1] * i // steps,
        )
        pw.mouseMoveEvent(mouse_event(QEvent.Type.MouseMove, pw, mid, none, left, plain))
        qapp.processEvents()
        owners.append(type(scene.dragItem).__name__)

    _gap()
    pw.mouseReleaseEvent(mouse_event(QEvent.Type.MouseButtonRelease, pw, end, left, none, plain))
    qapp.processEvents()
    return owners


def _drag_gesture(pw, qapp, aid: str, grab: tuple[float, float], *, steps: int = 3) -> int:
    """拖一次并返回这次手势往撤销栈里加了几步"""
    before_history = _history_len(pw)
    before_points = list(_manager(pw).get(aid).points)
    _drag(pw, qapp, grab, steps=steps)
    assert list(_manager(pw).get(aid).points) != pytest.approx(before_points), (
        "拖拽没生效，后面的合并断言会退化成空转"
    )
    return _history_len(pw) - before_history


# ---------------------------------------------------------------------------
# 手势合并：一次拖动只占一步
# ---------------------------------------------------------------------------

class TestGestureMerge:
    def test_dragging_a_rect_merges_into_one_step(self, annotated, qapp):
        aid = _manager(annotated).create("rect", [(20.0, 20.0), (30.0, 30.0)])
        assert aid is not None
        _manager(annotated).set_edit_mode(True)
        qapp.processEvents()

        added = _drag_gesture(annotated, qapp, aid, (25.0, 25.0))
        assert added == 1, (
            f"一次拖动记了 {added} 步 —— 撤销栈会被一个手势冲垮，"
            "用户按一次 Ctrl+Z 只退回一个像素"
        )

    def test_dragging_a_target_merges_into_one_step(self, annotated, qapp):
        """标靶没有手柄，拖动是它唯一的位置编辑手段，必须同样只占一步"""
        aid = _manager(annotated).create("target", [(25.0, 25.0)], text="目标")
        assert aid is not None
        _manager(annotated).set_edit_mode(True)
        qapp.processEvents()

        added = _drag_gesture(annotated, qapp, aid, (25.0, 25.0))
        assert added == 1, f"标靶一次拖动记了 {added} 步"

    def test_dragging_a_polyline_segment_merges_into_one_step(self, annotated, qapp):
        aid = _manager(annotated).create(
            "polyline", [(10.0, 10.0), (20.0, 40.0), (40.0, 10.0)]
        )
        assert aid is not None
        _manager(annotated).set_edit_mode(True)
        qapp.processEvents()

        added = _drag_gesture(annotated, qapp, aid, (25.0, 25.0))
        assert added == 1, f"折线一次拖动记了 {added} 步"

    def test_dragging_text_merges_into_one_step(self, annotated, qapp):
        """文字的"手势结束"不走信号，在 ``mouseReleaseEvent`` 里调注入的钩子"""
        aid = _manager(annotated).create("text", [(25.0, 25.0)], text="峰值")
        assert aid is not None
        _manager(annotated).set_edit_mode(True)
        qapp.processEvents()

        added = _drag_gesture(annotated, qapp, aid, (25.0, 25.0))
        assert added == 1, f"文字一次拖动记了 {added} 步"

    def test_more_intermediate_moves_still_merge_into_one_step(self, annotated, qapp):
        """把 move 拆成 8 步也只是那一步 —— 合并的判据是手势、不是事件条数"""
        aid = _manager(annotated).create("rect", [(20.0, 20.0), (30.0, 30.0)])
        assert aid is not None
        _manager(annotated).set_edit_mode(True)
        qapp.processEvents()

        added = _drag_gesture(annotated, qapp, aid, (25.0, 25.0), steps=8)
        assert added == 1, f"8 步 move 记了 {added} 步"

    def test_undo_after_a_drag_returns_to_the_pre_drag_geometry(self, annotated, qapp):
        """核心：撤销一步必须把手势**整段**退回起点，而不是退回倒数第二帧"""
        aid = _manager(annotated).create("rect", [(20.0, 20.0), (30.0, 30.0)])
        assert aid is not None
        _manager(annotated).set_edit_mode(True)
        qapp.processEvents()
        before = list(_manager(annotated).get(aid).points)

        _drag(annotated, qapp, (25.0, 25.0))
        assert _manager(annotated).get(aid).points != pytest.approx(before)

        assert _manager(annotated).undo() is True
        assert _manager(annotated).get(aid).points == pytest.approx(before), (
            "一次撤销停在手势中途 —— 拖动被拆成了多步"
        )

    def test_a_press_without_a_drag_records_nothing(self, annotated, qapp):
        """按下再松开（没动）不该吃掉一步撤销"""
        aid = _manager(annotated).create("rect", [(20.0, 20.0), (30.0, 30.0)])
        assert aid is not None
        _manager(annotated).set_edit_mode(True)
        qapp.processEvents()

        scene = annotated.plot_item.scene()
        left, none = Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton
        plain = Qt.KeyboardModifier.NoModifier
        spot = _pixel_of(annotated, (25.0, 25.0))

        before_history = _history_len(annotated)
        _gap()
        annotated.mouseMoveEvent(mouse_event(QEvent.Type.MouseMove, annotated, spot, none, none, plain))
        qapp.processEvents()
        scene.dragItem = None
        scene.dragButtons = []
        _gap()
        annotated.mousePressEvent(mouse_event(QEvent.Type.MouseButtonPress, annotated, spot, left, left, plain))
        qapp.processEvents()
        _gap()
        annotated.mouseReleaseEvent(mouse_event(QEvent.Type.MouseButtonRelease, annotated, spot, left, none, plain))
        qapp.processEvents()

        assert _history_len(annotated) == before_history, "空手势凭空加了一步"
        # 且栈没被搞乱：仍能正常撤销到创建之前
        assert _manager(annotated).undo() is True
        assert _manager(annotated).count() == 0

    def test_a_wedged_gesture_flag_does_not_block_later_steps(self, annotated, qapp):
        """``_in_gesture`` 靠鼠标事件配对；图元在手势中途被删掉就配不上对了

        显式操作一律视为手势结束（``_record_history`` 开头把它清掉）。
        没有这句的话，一次配不上对的手势会让**后续所有**变更都不入栈。
        """
        manager = _manager(annotated)
        manager._in_gesture = True                       # 模拟一个卡住的手势标记
        aid = manager.create("rect", [(20.0, 20.0), (30.0, 30.0)])
        assert aid is not None
        assert _history_len(annotated) == 2, "卡住的 _in_gesture 把后续变更全吞了"
        assert manager._in_gesture is False


# ---------------------------------------------------------------------------
# 各类变更各占一步，撤销后逐项还原
# ---------------------------------------------------------------------------

class TestEachMutationIsOneStep:
    def test_nothing_to_undo_at_the_start(self, annotated):
        manager = _manager(annotated)
        assert manager.can_undo is False
        assert manager.can_redo is False
        assert manager.undo() is False
        assert manager.redo() is False

    def test_create_is_one_undoable_step(self, annotated, qapp):
        manager = _manager(annotated)
        baseline_items = _scene_items(annotated)
        aid = manager.create("rect", [(20.0, 20.0), (30.0, 30.0)])
        assert aid is not None
        assert manager.can_undo is True

        assert manager.undo() is True
        assert manager.count() == 0
        assert _scene_items(annotated) == baseline_items, "撤销没把图元从场景里收回去"

        assert manager.redo() is True
        assert manager.count() == 1
        assert manager.get(aid) is not None

    def test_remove_can_be_undone(self, annotated, qapp):
        manager = _manager(annotated)
        aid = manager.create("rect", [(20.0, 20.0), (30.0, 30.0)])
        assert aid is not None
        before = manager.dump()

        assert manager.remove(aid) is True
        assert manager.count() == 0

        assert manager.undo() is True
        assert manager.dump() == before, "撤销删除没还原成原样"
        assert _scene_items(annotated) > 0

    def test_clear_can_be_undone(self, annotated, qapp):
        manager = _manager(annotated)
        for i in range(3):
            manager.create("rect", [(10.0 + 20 * i, 10.0), (18.0 + 20 * i, 18.0)])
        before = manager.dump()
        assert manager.count() == 3

        assert manager.clear() == 3
        assert manager.count() == 0

        assert manager.undo() is True
        assert manager.dump() == before, "撤销清空没把三条一起还回来"

    def test_visibility_can_be_undone(self, annotated, qapp):
        manager = _manager(annotated)
        aid = manager.create("rect", [(20.0, 20.0), (30.0, 30.0)])
        assert aid is not None

        assert manager.set_visible(aid, False) is True
        assert manager.get(aid).visible is False

        assert manager.undo() is True
        assert manager.get(aid).visible is True, "撤销显隐没生效"

        assert manager.redo() is True
        assert manager.get(aid).visible is False

    def test_style_change_can_be_undone(self, annotated, qapp):
        manager = _manager(annotated)
        aid = manager.create("rect", [(20.0, 20.0), (30.0, 30.0)])
        assert aid is not None
        original = manager.get(aid).stroke

        assert manager.apply_style(aid, stroke="#00A0FF", stroke_width=5) is True
        assert manager.get(aid).stroke == "#00A0FF"

        assert manager.undo() is True
        assert manager.get(aid).stroke == original
        assert manager.get(aid).stroke_width != 5

    def test_text_change_can_be_undone(self, annotated, qapp):
        manager = _manager(annotated)
        aid = manager.create("text", [(20.0, 20.0)], text="甲")
        assert aid is not None

        assert manager.apply_style(aid, text="乙") is True
        assert manager.get(aid).text == "乙"

        assert manager.undo() is True
        assert manager.get(aid).text == "甲"

    def test_three_creates_walk_back_three_steps(self, annotated, qapp):
        manager = _manager(annotated)
        for i in range(3):
            manager.create("rect", [(10.0 + 20 * i, 10.0), (18.0 + 20 * i, 18.0)])

        for expected in (2, 1, 0):
            assert manager.undo() is True
            assert manager.count() == expected
        assert manager.can_undo is False
        assert manager.undo() is False

    def test_a_mixed_session_round_trips_exactly(self, annotated, qapp):
        """一长串混合操作：全部撤销必须逐字节回到初始，全部重做必须回到终态"""
        manager = _manager(annotated)
        initial = manager.dump()          # 空

        a = manager.create("rect", [(10.0, 10.0), (20.0, 20.0)])
        b = manager.create("text", [(30.0, 30.0)], text="峰值")
        c = manager.create("target", [(40.0, 40.0)], text="目标")
        assert None not in (a, b, c)
        manager.remove(b)
        manager.apply_style(a, stroke="#00A0FF")
        manager.set_visible(c, False)
        manager.create("polyline", [(1.0, 1.0), (2.0, 5.0), (3.0, 2.0)])
        final = manager.dump()
        steps = _history_len(annotated) - 1

        for _ in range(steps):
            assert manager.undo() is True
        assert manager.dump() == initial, "撤销到底没有回到初始状态"
        assert manager.can_undo is False

        for _ in range(steps):
            assert manager.redo() is True
        assert manager.dump() == final, "重做到底没有回到终态"
        assert manager.can_redo is False


# ---------------------------------------------------------------------------
# 模板装载与回放
# ---------------------------------------------------------------------------

class TestTemplateAndReplay:
    def test_load_records_exactly_one_step(self, annotated, qapp):
        """回归：``clear(record=False)`` 曾被一个同名循环变量悄悄作废

        ``clear`` 里原来是 ``for record in records:``，循环结束后 ``record`` 变成
        最后一个图元（真值），于是 ``if record:`` 恒真 —— 整份替换会记**两步**：
        先是"空图"、再是新内容。用户撤销一次只得到空图，得连按两次。
        """
        manager = _manager(annotated)
        manager.create("rect", [(10.0, 10.0), (20.0, 20.0)])
        base = _history_len(annotated)

        manager.load([{"id": "abcdef01", "kind": "target", "points": [[50.0, 50.0]]}])
        assert _history_len(annotated) == base + 1, (
            f"整份替换记了 {_history_len(annotated) - base} 步（中间的空图不该入栈）"
        )

        assert manager.undo() is True
        assert manager.count() == 1, "撤销停在中间的空白状态上"

    def test_clear_with_record_false_leaves_the_history_alone(self, annotated, qapp):
        """直接钉 ``clear(record=False)`` 的语义：清画面，但不占历史"""
        manager = _manager(annotated)
        manager.create("rect", [(10.0, 10.0), (20.0, 20.0)])
        base = _history_len(annotated)

        assert manager.clear(record=False) == 1
        assert manager.count() == 0
        assert _history_len(annotated) == base, "record=False 的 clear 还是记了一步"

    def test_applying_a_template_is_one_step(self, annotated, qapp):
        manager = _manager(annotated)
        manager.create("rect", [(10.0, 10.0), (20.0, 20.0)])
        template = manager.dump()

        manager.clear()
        assert manager.load(template) == 1
        assert manager.count() == 1

        assert manager.undo() is True
        assert manager.count() == 0, "套用模板占了一步以上（中间那个'空'的状态不该单独入栈）"

    def test_undoing_a_template_restores_the_previous_set(self, annotated, qapp):
        """替换式装载的中间态是"空图"，它不该单独占一步 ——
        否则撤销一次只得到空图，用户得连按两次
        """
        manager = _manager(annotated)
        manager.create("rect", [(10.0, 10.0), (20.0, 20.0)])
        manager.create("text", [(30.0, 30.0)], text="原有")
        previous = manager.dump()

        template = [{"id": "abcdef01", "kind": "target", "points": [[50.0, 50.0]],
                     "text": "模板"}]
        assert manager.load(template) == 1
        assert manager.count() == 1

        assert manager.undo() is True
        assert manager.dump() == previous, "撤销套用模板没有回到套用前的整图"

    def test_replay_is_not_undoable(self, annotated, qapp):
        """换布局回放走 ``load(record=False)``（见 ``layout_manager.replay_annotations``）

        新子图的历史本是空的；不重设基准的话，第一次 Ctrl+Z 会把刚回放出来的
        标注整份抹掉 —— 用户看到的是"换个布局标注全没了"。
        """
        manager = _manager(annotated)
        entries = [{"id": "abcdef01", "kind": "target", "points": [[50.0, 50.0]],
                    "text": "回放"}]

        assert manager.load(entries, replace=True, record=False) == 1
        assert manager.count() == 1
        assert manager.can_undo is False, "回放留了一步可撤销的历史"
        assert manager.undo() is False
        assert manager.count() == 1, "撤销把回放出来的标注抹掉了"

    def test_replay_after_manual_edits_resets_the_baseline(self, annotated, qapp):
        manager = _manager(annotated)
        manager.create("rect", [(10.0, 10.0), (20.0, 20.0)])
        assert manager.can_undo is True

        manager.load(
            [{"id": "abcdef02", "kind": "rect", "points": [[1.0, 1.0], [5.0, 5.0]]}],
            replace=True,
            record=False,
        )
        assert manager.count() == 1
        assert manager.can_undo is False, "旧图的撤销栈被带到了新图"
        assert manager.undo() is False
        assert manager.count() == 1


# ---------------------------------------------------------------------------
# 栈的边界
# ---------------------------------------------------------------------------

class TestStackBoundaries:
    def test_repeating_the_same_state_does_not_add_a_step(self, annotated, qapp):
        """状态没变的操作不入栈，否则撤销栈会被一堆空步占满"""
        manager = _manager(annotated)
        aid = manager.create("rect", [(20.0, 20.0), (30.0, 30.0)])
        assert aid is not None
        base = _history_len(annotated)

        assert manager.remove("不存在的id") is False
        manager.set_visible(aid, True)                    # 本来就是 True
        manager.apply_style(aid, stroke=manager.get(aid).stroke)   # 同色
        manager.apply_style(aid, stroke_width=manager.get(aid).stroke_width)
        assert manager.load([], replace=False) == 0       # 空装载

        assert _history_len(annotated) == base, (
            f"空操作吃掉了 {_history_len(annotated) - base} 步撤销"
        )

    def test_selection_is_not_a_step(self, annotated, qapp):
        """选中只抬 Z 序、不进模型，不该占撤销步"""
        manager = _manager(annotated)
        aid = manager.create("rect", [(20.0, 20.0), (30.0, 30.0)])
        assert aid is not None
        base = _history_len(annotated)

        manager.select(aid)
        manager.select(None)
        manager.select(aid)

        assert _history_len(annotated) == base
        # 而且选中状态本身不该被撤销影响
        assert manager.selected_id == aid
        assert manager.undo() is True
        assert manager.count() == 0

    def test_new_change_after_undo_clears_the_redo_stack(self, annotated, qapp):
        """撤销之后又做了新操作：原来那条"未来"就不可达了，必须丢弃"""
        manager = _manager(annotated)
        manager.create("rect", [(10.0, 10.0), (20.0, 20.0)])
        manager.create("text", [(30.0, 30.0)], text="B")
        assert manager.redo() is False

        assert manager.undo() is True
        assert manager.can_redo is True

        manager.create("target", [(50.0, 50.0)], text="新分支")
        assert manager.can_redo is False, "redo 栈没被清空，会重做到一条不存在的分支上"

    def test_redo_stack_survives_multiple_undos(self, annotated, qapp):
        manager = _manager(annotated)
        for i in range(3):
            manager.create("rect", [(10.0 + 20 * i, 10.0), (18.0 + 20 * i, 18.0)])
        final = manager.dump()

        for _ in range(3):
            assert manager.undo() is True
        assert manager.count() == 0

        for _ in range(3):
            assert manager.redo() is True
        assert manager.dump() == final

    def test_history_is_capped(self, annotated, qapp, monkeypatch):
        """容量上限：快照式撤销每步存一份完整 dump，不能无界增长"""
        from src.ui.widgets import annotation_manager as am_module

        monkeypatch.setattr(am_module, "MAX_UNDO_STEPS", 5)
        manager = _manager(annotated)
        for i in range(20):
            manager.create("rect", [(float(i), 0.0), (float(i) + 1.0, 1.0)])

        assert _history_len(annotated) <= 5, (
            f"历史长到 {_history_len(annotated)} 步，上限没生效"
        )

    def test_undo_cannot_walk_past_the_retained_window(self, annotated, qapp, monkeypatch):
        from src.ui.widgets import annotation_manager as am_module

        monkeypatch.setattr(am_module, "MAX_UNDO_STEPS", 5)
        manager = _manager(annotated)
        for i in range(20):
            manager.create("rect", [(float(i), 0.0), (float(i) + 1.0, 1.0)])

        undos = 0
        while manager.undo():
            undos += 1
            assert undos < 50, "undo 没有终止（can_undo 与 pop 不同步）"

        assert undos == 4, f"上限 5 只该给出 4 次撤销，实得 {undos}"
        assert manager.count() == 16

    def test_default_cap_is_the_documented_number(self):
        """上限常量本身别悄悄漂走（50 步 ≈ 1MB 级，够用且不无界）"""
        assert MAX_UNDO_STEPS == 50

    def test_reset_history_makes_the_current_state_the_baseline(self, annotated, qapp):
        manager = _manager(annotated)
        for i in range(3):
            manager.create("rect", [(10.0 + 20 * i, 10.0), (18.0 + 20 * i, 18.0)])
        assert manager.can_undo is True

        manager.reset_history()
        assert manager.can_undo is False
        assert manager.can_redo is False
        assert manager.undo() is False
        assert manager.count() == 3, "reset_history 动到了画面"

    def test_reset_history_forgets_an_inflight_gesture(self, annotated, qapp):
        manager = _manager(annotated)
        manager._in_gesture = True
        manager.reset_history()
        assert manager._in_gesture is False


# ---------------------------------------------------------------------------
# 重入守卫：撤销自己不入栈
# ---------------------------------------------------------------------------

class TestReentrancy:
    def test_undo_does_not_record_itself(self, annotated, qapp):
        """``_restore`` 里若不加 ``_restoring`` 守卫，一次撤销会把"撤销"本身
        也记成一步，用户按 Ctrl+Z 会来回弹。
        """
        manager = _manager(annotated)
        manager.create("rect", [(10.0, 10.0), (20.0, 20.0)])
        assert _history_len(annotated) == 2

        assert manager.undo() is True
        assert _history_len(annotated) == 1
        assert manager.can_undo is False, "撤销把自己也记了一步，Ctrl+Z 会来回弹"

    def test_redo_does_not_record_itself(self, annotated, qapp):
        manager = _manager(annotated)
        manager.create("rect", [(10.0, 10.0), (20.0, 20.0)])
        manager.undo()

        assert manager.redo() is True
        assert _history_len(annotated) == 2
        assert manager.can_redo is False

    def test_alternating_undo_redo_is_stable(self, annotated, qapp):
        """反复撤销/重做不该让栈长起来（每次都必须回到同样的两端）"""
        manager = _manager(annotated)
        manager.create("rect", [(10.0, 10.0), (20.0, 20.0)])
        manager.create("text", [(30.0, 30.0)], text="B")
        with_items = manager.dump()
        history = _history_len(annotated)

        for _ in range(5):
            assert manager.undo() is True
            assert manager.undo() is True
            assert manager.count() == 0
            assert manager.redo() is True
            assert manager.redo() is True
            assert manager.dump() == with_items

        assert _history_len(annotated) == history, "反复撤销/重做把历史撑大了"

    def test_undo_rebuilds_the_scene_instead_of_patching_it(self, annotated, qapp):
        """``_restore`` 走 ``load(replace=True)``：撤销后场景里的图元数必须与
        "从没画过"时一致 —— 打补丁式的实现很容易漏掉子图元（折线线段、标靶标签）
        """
        manager = _manager(annotated)
        baseline = _scene_items(annotated)

        manager.create("polyline", [(10.0, 10.0), (20.0, 40.0), (40.0, 10.0)])
        manager.create("target", [(50.0, 50.0)], text="目标 {24℃}")
        manager.create("arrow", [(60.0, 20.0), (80.0, 60.0)])
        qapp.processEvents()
        assert _scene_items(annotated) > baseline

        for _ in range(3):
            assert manager.undo() is True
        qapp.processEvents()

        assert manager.count() == 0
        assert _scene_items(annotated) == baseline, (
            "撤销后场景里留下了孤儿图元（子图元没跟着回收）"
        )
