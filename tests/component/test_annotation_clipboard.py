"""标注复制 / 粘贴 component 测试（offscreen）。

主场景是**多分区对比图**：主驾那套标注照搬到副驾/后排。所以剪贴板的落点是
``pw.window()``（容器级）而不是单个子图 —— 本文件的第一组用例就守着这条：
两个挂在同一宿主下的子图必须共享同一份剪贴板。

覆盖点：
1. 复制源：选中一条 / 整图（``all_items``）/ 快捷键路径（无选中退化为整图）
2. 剪贴板是**共享且原地更新**的 list（别人可能正持着它）
3. 粘贴：换新 id、按视图跨度 4% 错开、跨子图、坏条目跳过、上限
4. 粘贴不吃别人的标注（源子图不受影响）

``paste()`` 只走 ``load(replace=False)``，所以"能不能贴进合法标注"这件事由
装载路径守着（``test_annotation_manager.py`` 已覆盖），本文件不重复。
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.core.annotation_models import MAX_ANNOTATIONS_PER_PLOT
from tests.component.conftest import FakeHost

#: 粘贴位移比例（与 annotation_manager.PASTE_OFFSET_RATIO 同源）
PASTE_OFFSET_RATIO = 0.04

#: 本文件固定用 0..100 的视图范围 —— 这样 4% 就是干净的 4.0 数据单位
_SPAN_LO = 0.0
_SPAN_HI = 100.0


class _RecordingHost(FakeHost):
    """带 ``_broadcast`` 的宿主：标注播报走 ``window()._broadcast``

    ``FakeHost`` 没有这个方法，``_announce`` 会整个 no-op；要断言"用户知不知道自己
    复制了什么"，就得把播报收下来。
    """

    def __init__(self):
        super().__init__()
        self.messages: list[str] = []

    def _broadcast(self, text: str) -> None:
        self.messages.append(text)


def _make_pair_under_one_host(plot_factory, qapp):
    """两只**挂在同一宿主下**的子图（= 同一个容器级剪贴板）

    ``plot_factory`` 默认给每只 widget 配一个独立 FakeHost，那样的两只子图
    ``window()`` 不同、剪贴板也不共享 —— 正好用来做"不共享"的反面对照。
    """
    df = pd.DataFrame({"a": [float(i) for i in range(100)]})
    host = _RecordingHost()
    pair = []
    for _ in range(2):
        pw = plot_factory(df)
        pw.setParent(host)          # 换到同一个宿主
        pw.resize(800, 600)
        host.show()
        pw.show()
        assert pw.plot_variable("a"), "曲线未绘出，视图跨度断言失去意义"
        pw.view_box.setXRange(_SPAN_LO, _SPAN_HI, padding=0)
        pw.view_box.setYRange(_SPAN_LO, _SPAN_HI, padding=0)
        pair.append(pw)
    qapp.processEvents()
    return host, pair[0], pair[1]


@pytest.fixture()
def shared(plot_factory, qapp):
    """同一宿主下的两只子图 A / B"""

    host, a, b = _make_pair_under_one_host(plot_factory, qapp)
    yield host, a, b


@pytest.fixture()
def solo(plot_factory, qapp):
    """单独一只子图（自带独立宿主）"""
    df = pd.DataFrame({"a": [float(i) for i in range(100)]})
    pw = plot_factory(df)
    pw.resize(800, 600)
    pw.window().show()
    pw.show()
    assert pw.plot_variable("a")
    pw.view_box.setXRange(_SPAN_LO, _SPAN_HI, padding=0)
    pw.view_box.setYRange(_SPAN_LO, _SPAN_HI, padding=0)
    qapp.processEvents()
    return pw


def _manager(pw):
    return pw.annotation_manager


def _expected_offset() -> tuple[float, float]:
    """容器里固定 0..100 视图跨度下的粘贴位移（屏幕上是"往右下"）"""
    dx = (_SPAN_HI - _SPAN_LO) * PASTE_OFFSET_RATIO
    dy = (_SPAN_LO - _SPAN_HI) * PASTE_OFFSET_RATIO   # 数据 y 轴向上 → 取负
    return (dx, dy)


def _seed(pw, n: int = 2) -> list[str]:
    """在子图上放 n 条可辨识的标注，返回它们的 id"""
    ids = []
    for i in range(n):
        x = 10.0 + 20.0 * i
        aid = _manager(pw).create("rect", [(x, 10.0), (x + 8.0, 18.0)])
        assert aid is not None, "标注未创建，用例前提不成立"
        ids.append(aid)
    return ids


def _points_of(pw, aid):
    return _manager(pw).get(aid).points


# ---------------------------------------------------------------------------
# 复制源
# ---------------------------------------------------------------------------

class TestCopySource:
    def test_copy_without_selection_copies_nothing(self, solo):
        """菜单里"复制"的语义是**复制选中项**：没选中就没得复制"""
        _seed(solo, 2)
        assert _manager(solo).copy() == 0
        assert _manager(solo).clipboard_count == 0

    def test_copy_takes_only_the_selected_item(self, solo):
        ids = _seed(solo, 3)
        _manager(solo).select(ids[1])

        assert _manager(solo).copy() == 1
        assert _manager(solo).clipboard_count == 1

        _manager(solo).clear()
        assert _manager(solo).paste() == 1                     # 只贴回那一条
        pasted = _manager(solo).items()
        assert len(pasted) == 1
        # 贴出来的是"选中那一条"的几何，不是第一/最后一条
        assert pasted[0].points == pytest.approx(
            [(30.0 + _expected_offset()[0], 10.0 + _expected_offset()[1]),
             (38.0 + _expected_offset()[0], 18.0 + _expected_offset()[1])]
        )

    def test_copy_all_takes_everything(self, solo):
        _seed(solo, 3)
        assert _manager(solo).copy(all_items=True) == 3
        assert _manager(solo).clipboard_count == 3

    def test_copy_all_with_no_annotations_is_a_noop(self, solo):
        assert _manager(solo).copy(all_items=True) == 0
        assert _manager(solo).clipboard_count == 0

    def test_shortcut_path_falls_back_to_copy_all(self, solo):
        """Ctrl+C 只有一键：没先点选时退化成整图复制，比什么都不发生有用"""
        _seed(solo, 3)
        assert _manager(solo).copy_selection_or_all() == 3

    def test_shortcut_path_prefers_the_selection(self, solo):
        ids = _seed(solo, 3)
        _manager(solo).select(ids[0])
        assert _manager(solo).copy_selection_or_all() == 1

    def test_shortcut_path_ignores_a_stale_selection(self, solo):
        """选中项被删掉后必须退化成整图复制

        用 3 条、删 1 条来构造：留在图上的是 **2** 条。若快捷键误判成"有选中"
        而走向单条复制，就会得到 0 条 —— 两条路径的结果数不同，用例才分得清。
        """
        ids = _seed(solo, 3)
        _manager(solo).select(ids[1])
        _manager(solo).remove(ids[1])
        assert _manager(solo).count() == 2

        assert _manager(solo).copy_selection_or_all() == 2, (
            "残留的选中 id 让快捷键一条都没复制出来"
        )

    def test_copy_survives_a_deleted_selection_gracefully(self, solo):
        ids = _seed(solo, 2)
        _manager(solo).select(ids[0])
        _manager(solo).remove(ids[0])
        assert _manager(solo).copy() == 0


# ---------------------------------------------------------------------------
# 剪贴板是容器级、且原地更新
# ---------------------------------------------------------------------------

class TestClipboardPlacement:
    def test_the_clipboard_lives_on_the_window(self, solo):
        _seed(solo, 1)
        _manager(solo).copy(all_items=True)
        assert getattr(solo.window(), "annotation_clipboard", None)

    def test_two_subplots_under_one_host_share_it(self, shared):
        _, a, b = shared
        _seed(a, 2)
        assert _manager(a).copy(all_items=True) == 2
        assert _manager(b).clipboard_count == 2, (
            "剪贴板没挂在容器上 —— 跨子图照搬标注就废了"
        )

    def test_independent_hosts_do_not_share(self, plot_factory, qapp):
        """反面对照：宿主不同 = 不同剪贴板（保证上一条不是恒真）"""
        df = pd.DataFrame({"a": [float(i) for i in range(100)]})
        one = plot_factory(df)
        other = plot_factory(df)
        for pw in (one, other):
            pw.resize(800, 600)
            pw.show()
            assert pw.plot_variable("a")
        qapp.processEvents()

        _seed(one, 2)
        assert _manager(one).copy(all_items=True) == 2
        assert _manager(other).clipboard_count == 0

    def test_the_list_is_updated_in_place(self, shared):
        """必须**原地改**而不是重新赋值：别的子图可能已经持着这个 list 对象"""
        _, a, b = shared
        _seed(a, 2)
        store = _manager(b)._clipboard()
        assert store is not None

        _manager(a).copy(all_items=True)
        assert len(store) == 2, "剪贴板被换成了新 list，持有旧引用的子图看不到更新"

        _manager(a).copy()          # 没选中 → 0 条，但也不能把旧内容清掉
        assert len(store) == 2

    def test_a_second_copy_replaces_the_previous_content(self, shared):
        _, a, b = shared
        _seed(a, 3)
        _manager(a).copy(all_items=True)
        _manager(a).select(_manager(a).items()[0].id)
        assert _manager(a).copy() == 1
        assert _manager(b).clipboard_count == 1, "第二次复制应覆盖而不是追加"


# ---------------------------------------------------------------------------
# 粘贴
# ---------------------------------------------------------------------------

class TestPaste:
    def test_paste_on_empty_clipboard_is_a_noop(self, solo):
        assert _manager(solo).paste() == 0
        assert _manager(solo).count() == 0

    def test_paste_offsets_by_four_percent_of_the_view_span(self, solo):
        _seed(solo, 1)
        source = _manager(solo).items()[0]
        source_points = list(source.points)
        _manager(solo).copy(all_items=True)

        assert _manager(solo).paste() == 1
        dx, dy = _expected_offset()
        assert dx == pytest.approx(4.0) and dy == pytest.approx(-4.0)

        pasted = [it for it in _manager(solo).items() if it.id != source.id]
        assert len(pasted) == 1
        assert pasted[0].points == pytest.approx(
            [(x + dx, y + dy) for x, y in source_points]
        ), "粘贴没有错开，会与原标注严丝合缝地重合"

    def test_paste_does_not_touch_the_source(self, solo):
        _seed(solo, 2)
        before = {it.id: list(it.points) for it in _manager(solo).items()}
        _manager(solo).copy(all_items=True)
        _manager(solo).paste()

        for aid, points in before.items():
            assert _points_of(solo, aid) == pytest.approx(points), (
                "粘贴改到了原有标注"
            )

    def test_pasted_items_get_fresh_ids(self, solo):
        ids = _seed(solo, 2)
        _manager(solo).copy(all_items=True)
        assert _manager(solo).paste() == 2

        after = {it.id for it in _manager(solo).items()}
        assert set(ids) <= after, "原标注不见了"
        assert len(after) == 4, "粘贴出来的 id 撞了原有的（图形表会互相覆盖）"

    def test_paste_keeps_kind_text_and_style(self, solo):
        aid = _manager(solo).create(
            "target", [(25.0, 30.0)], text="目标 {24℃}", stroke="#00A0FF", font_size=16
        )
        assert aid is not None
        _manager(solo).copy(all_items=True)
        _manager(solo).paste()

        pasted = [it for it in _manager(solo).items() if it.id != aid]
        assert len(pasted) == 1
        assert pasted[0].kind == "target"
        assert pasted[0].text == "目标 {24℃}"
        assert pasted[0].stroke == "#00A0FF"
        assert pasted[0].font_size == 16

    def test_paste_is_one_undo_step(self, solo):
        _seed(solo, 2)
        _manager(solo).copy(all_items=True)
        _manager(solo).paste()
        assert _manager(solo).count() == 4

        assert _manager(solo).undo() is True
        assert _manager(solo).count() == 2, "一次撤销应该把整批粘贴一起收回"

    def test_paste_skips_malformed_entries(self, solo):
        """剪贴板内容是外部可写的（模板/别的子图的 dump），坏条目跳过就行"""
        _seed(solo, 1)
        _manager(solo).copy(all_items=True)
        store = _manager(solo)._clipboard()
        store.extend([
            "不是字典",
            {"kind": "rect", "points": "不是点列"},
            {"kind": "rect", "points": []},
            {"kind": "不存在的类型", "points": [[1.0, 2.0], [3.0, 4.0]]},
        ])

        assert _manager(solo).paste() == 1, "坏条目应该被跳过，好条目照贴"

    def test_paste_into_another_subplot(self, shared):
        _, a, b = shared
        ids_a = _seed(a, 2)
        points_a = {aid: list(_points_of(a, aid)) for aid in ids_a}
        assert _manager(a).copy(all_items=True) == 2

        assert _manager(b).paste() == 2
        assert _manager(b).count() == 2
        assert _manager(a).count() == 2, "粘贴到 B 不该动到 A"

        dx, dy = _expected_offset()
        for it in _manager(b).items():
            assert it.id not in points_a, "跨子图粘贴也换新 id（id 相同会让列表/diff 难读）"
        b_points = sorted(tuple(it.points[0]) for it in _manager(b).items())
        expected = sorted(
            (x + dx, y + dy) for (x, y) in (p[0] for p in points_a.values())
        )
        assert b_points == pytest.approx(expected)

    def test_paste_twice_stacks(self, shared):
        _, a, b = shared
        _seed(a, 2)
        _manager(a).copy(all_items=True)
        assert _manager(b).paste() == 2
        assert _manager(b).paste() == 2
        assert _manager(b).count() == 4

    def test_paste_stops_at_the_annotation_cap(self, solo, monkeypatch):
        """上限在装载路径上收口，粘贴跟着一起被挡"""
        from src.ui.widgets import annotation_manager as am_module

        _seed(solo, 2)
        _manager(solo).copy(all_items=True)

        monkeypatch.setattr(am_module, "MAX_ANNOTATIONS_PER_PLOT", 3)
        assert _manager(solo).paste() == 1, "已有 2 条、上限 3 时应只贴进 1 条"
        assert _manager(solo).count() == 3

    def test_cap_constant_is_the_documented_one(self):
        """上限常量本身别悄悄漂走（200 条 ≈ 单条 200 字节，够用且不无界）"""
        assert MAX_ANNOTATIONS_PER_PLOT == 200


# ---------------------------------------------------------------------------
# 播报（用户必须知道复制/粘贴了什么）
# ---------------------------------------------------------------------------

class TestAnnouncements:
    def test_copy_all_announces_the_count(self, shared):
        host, a, _ = shared
        _seed(a, 3)
        host.messages.clear()
        _manager(a).copy(all_items=True)
        assert any("3" in m and "复制" in m for m in host.messages), host.messages

    def test_copy_selection_announces_the_count(self, shared):
        host, a, _ = shared
        ids = _seed(a, 3)
        _manager(a).select(ids[0])
        host.messages.clear()
        _manager(a).copy_selection_or_all()
        assert any("1" in m and "复制" in m for m in host.messages), host.messages

    def test_paste_announces_the_count(self, shared):
        host, a, b = shared
        _seed(a, 2)
        _manager(a).copy(all_items=True)
        host.messages.clear()
        _manager(b).paste()
        assert any("2" in m and "粘贴" in m for m in host.messages), host.messages

    def test_failed_paste_says_nothing(self, shared):
        host, _, b = shared
        host.messages.clear()
        assert _manager(b).paste() == 0
        assert not any("粘贴" in m for m in host.messages), host.messages
