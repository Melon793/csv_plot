"""整理布局提交器测试：置换应用、跨页重注入、XLink 重连、未建页补建。"""

from __future__ import annotations

import pandas as pd
import pytest

from src.ui.widgets.plot_arrange_commit import (
    _row_heights_from_measurement,
    apply_arrangement,
    build_state,
    snapshot_pages,
    validate_state,
)
from src.core.plot_arrange_model import insert_from_tray, kick_to_tray
from src.ui.widgets.plot_page import PlotPage
from tests.fixtures.waits import flush_deferred_deletes


class _FakeButton:
    def __init__(self, checked: bool = False):
        self._checked = checked

    def isChecked(self) -> bool:
        return self._checked


class _FakeLayoutManager:
    def request_mark_stats_refresh(self, *, immediate: bool = False):
        pass


class _FakeCursorSync:
    def collect_global_x_range(self):
        return (0.0, 1.0)

    def _sync_min_xrange(self):
        pass


class _FakeServices:
    """PlotPage.build_matrix / _new_plot_container 所需的最小服务集合。"""

    def __init__(self):
        self.units = {}
        self.data = pd.DataFrame({"a": [0.0, 1.0, 2.0]})
        self.time_channels_infos = {}
        self.cursor_btn = _FakeButton(False)
        self.cursor_values_hidden = False
        self.cursor_mode = "1 free cursor"
        self.pinned_x_values = []
        self.mark_region_btn = _FakeButton(False)
        self.layout_manager = _FakeLayoutManager()
        self.cursor_sync_manager = _FakeCursorSync()
        self.loader = None
        self.value_cache = {}
        self._enum_text_maps = {}
        self.factor = 1.0
        self.offset = 0.0
        self.saved_mark_range = None
        self._global_max_density = 0.0
        self._data_version = 0
        self._is_loading_new_data = False
        self._is_time_correction_active = False


def _set_curves(container, names):
    container.plot_widget.curves = {name: object() for name in names}


@pytest.fixture()
def pages(qapp, qtbot):
    """页A 2x2（3 条曲线，1 空位）+ 页B 1x2（1 条曲线）+ 页C 未建 1x2。"""
    page_a = PlotPage(
        parent=None, title="页1", rows_max=4, cols_max=3,
        rows_current=2, cols_current=2,
    )
    page_a.services = _FakeServices()
    page_a.build_matrix(2, 2)
    _set_curves(page_a.plot_widgets[0], ["a", "b"])
    _set_curves(page_a.plot_widgets[1], ["c"])
    _set_curves(page_a.plot_widgets[2], ["d"])

    page_b = PlotPage(
        parent=None, title="页2", rows_max=4, cols_max=3,
        rows_current=1, cols_current=2,
    )
    page_b.services = _FakeServices()
    page_b.build_matrix(1, 2)
    _set_curves(page_b.plot_widgets[0], ["e"])

    page_c = PlotPage(
        parent=None, title="页3", rows_max=4, cols_max=3,
        rows_current=1, cols_current=2,
    )
    page_c.services = _FakeServices()
    # 页C 刻意不建矩阵：验证提交时的惰性补建

    for page in (page_a, page_b, page_c):
        qtbot.addWidget(page.widget)

    yield page_a, page_b, page_c

    for page in (page_a, page_b, page_c):
        page.dispose()
    flush_deferred_deletes()


def _token_of(state, page_index, seat):
    return state.pages[page_index].cells[seat]


def test_snapshot_builds_tokens_for_curve_containers_only(pages):
    page_a, page_b, page_c = pages
    snapshot = snapshot_pages([page_a, page_b, page_c])
    state = build_state(snapshot)
    assert len(snapshot.tokens) == 4  # 3 + 1，空位/未建页不算
    assert state.pages[0].rows == 2 and state.pages[0].cols == 2
    assert state.pages[2].built is False
    assert sum(1 for c in state.pages[0].cells if c) == 3


def test_snapshot_uses_layout_positions_not_list_order(pages):
    """格位以 QGridLayout 真实坐标为准：列表下标与布局坐标脱节也不串位。"""
    page_a, _page_b, _page_c = pages
    # 人为制造「列表顺序 ≠ 布局坐标」：把 0 号容器挪到新的一行
    container = page_a.plot_widgets[0]
    page_a.plot_layout.addWidget(container, 2, 0)
    snapshot = snapshot_pages([page_a])
    state = build_state(snapshot)
    # token id 与格位解耦（按容器编号，不随行列漂移）
    token = state.pages[0].cells[4]  # 2 列网格下 (row=2, col=0) → seat 4
    assert token is not None
    assert snapshot.tokens[token] is container
    assert snapshot.seat_of_token[token] == (2, 0)
    assert state.pages[0].rows == 3


def test_hidden_row_height_estimates_follow_factors():
    """隐藏行按本行行高因子估算（同口径换算），不再一律取可见中位数。"""
    heights = _row_heights_from_measurement(
        [200.0, 400.0, 200.0, 0.0, 0.0], [50, 100, 50, 25, 100]
    )
    assert heights == [200.0, 400.0, 200.0, 100.0, 400.0]

    # 页面未布局（全 0）：退回百分比作相对权重
    assert _row_heights_from_measurement(
        [0.0, 0.0, 0.0], [50, 100, 200]
    ) == [50.0, 100.0, 200.0]


def test_snapshot_measured_hidden_rows_keep_factor_ratio(pages):
    """真实布局下：隐藏行估算与可见行同口径（25% 行 = 100% 行的 1/4）。"""
    from tests.fixtures.waits import settle

    page_a, _page_b, _page_c = pages
    page = PlotPage(
        parent=None, title="页1", rows_max=5, cols_max=3,
        rows_current=3, cols_current=1,
    )
    page.services = _FakeServices()
    page.build_matrix(5, 3)
    page.apply_visibility(3, 1)
    for row, pct in {0: 50, 1: 100, 2: 50, 3: 25, 4: 100}.items():
        page.set_row_height(row, pct)
    page.widget.resize(900, 800)
    page.widget.show()
    settle()

    snapshot = snapshot_pages([page])
    heights = snapshot.row_heights[0]
    unit = heights[1] / 100  # 100% 行
    assert unit > 0
    assert abs(heights[3] / unit - 25) < 5
    assert abs(heights[4] / unit - 100) < 5

    page.dispose()
    flush_deferred_deletes()


def test_snapshot_carries_row_height_percentages(pages):
    page_a, _page_b, _page_c = pages
    page_a.row_height_factors = {0: 50, 1: 200}
    snapshot = snapshot_pages([page_a])
    assert snapshot.row_pcts[0] == [50, 200]


def test_apply_in_page_swap_keeps_curves_and_relinks_x(pages):
    page_a, _page_b, _page_c = pages
    snapshot = snapshot_pages([page_a])
    state = build_state(snapshot)
    token_at_0 = state.pages[0].cells[0]
    token_at_1 = state.pages[0].cells[1]
    container_at_0 = snapshot.tokens[token_at_0]
    container_at_1 = snapshot.tokens[token_at_1]

    # 交换 seat0 与 seat1（等价拖 1-1 到 1-2）
    from src.core.plot_arrange_model import move_within

    # 提交前给 master 一个可辨认的 X 范围；spy 证明恢复逻辑真被调用
    page_a.master_viewbox().setXRange(3.0, 9.0, padding=0)
    applied_ranges = []
    original_apply = page_a.apply_x_range
    page_a.apply_x_range = lambda x_min, x_max: applied_ranges.append(
        (x_min, x_max)
    ) or original_apply(x_min, x_max)
    result = move_within(state, 0, [token_at_0], 1)
    assert result["ok"] is True
    apply_arrangement(snapshot, state)
    page_a.apply_x_range = original_apply
    assert applied_ranges, "提交后未走 apply_x_range 恢复 X 范围"
    assert abs(applied_ranges[0][0] - 3.0) < 1e-6
    assert abs(applied_ranges[0][1] - 9.0) < 1e-6

    assert page_a.plot_widgets[0] is container_at_1
    assert page_a.plot_widgets[1] is container_at_0
    assert set(page_a.plot_widgets[1].plot_widget.curves) == {"a", "b"}
    # 非 master 子图必须重新 link 到首可见 master
    master_vb = page_a.plot_widgets[0].plot_widget.view_box
    linked = page_a.plot_widgets[1].plot_widget.view_box.linkedView(0)
    assert linked is master_vb
    x_min, x_max = master_vb.viewRange()[0]
    assert abs(x_min - 3.0) < 1e-6 and abs(x_max - 9.0) < 1e-6, (
        f"提交后 X 范围漂移：{x_min}..{x_max}"
    )


def test_cross_page_move_reinjects_page_and_context(pages):
    page_a, page_b, _page_c = pages
    snapshot = snapshot_pages([page_a, page_b])
    state = build_state(snapshot)

    token_a0 = _token_of(state, 0, 0)
    token_b0 = _token_of(state, 1, 0)
    container_a0 = snapshot.tokens[token_a0]
    container_b0 = snapshot.tokens[token_b0]

    # 页A 的图 1 顶替页B 图 1；页B 原图回填页A 的空位（seat3）
    kick_to_tray(state, 0, [token_a0])
    insert_from_tray(state, 1, [token_a0], 0)
    assert state.tray == [token_b0]
    insert_from_tray(state, 0, [token_b0], 3)
    assert state.tray == []

    apply_arrangement(snapshot, state)

    moved = container_a0.plot_widget
    assert page_b.plot_widgets[0] is container_a0
    assert moved.page is page_b
    assert moved.plot_context._services is page_b.services
    back = container_b0.plot_widget
    assert page_a.plot_widgets[3] is container_b0
    assert back.page is page_a
    assert back.plot_context._services is page_a.services
    # 搬入页B 首格后它就是该页 master：自身无 XLink，页B 其余格必须链到它；
    # 页A 的兄弟格只能链页A 新 master，绝不能残留跨页 link
    assert page_b.master_container().plot_widget is moved
    assert moved.view_box.linkedView(0) is None
    sibling_b = page_b.plot_widgets[1].plot_widget
    assert sibling_b.view_box.linkedView(0) is moved.view_box
    for container in page_a.plot_widgets:
        linked = container.plot_widget.view_box.linkedView(0)
        assert linked is None or linked is page_a.master_viewbox()


def test_unbuilt_page_is_built_on_commit(pages):
    page_a, _page_b, page_c = pages
    snapshot = snapshot_pages([page_a, page_c])
    state = build_state(snapshot)
    assert len(page_c.plot_widgets) == 0

    token = _token_of(state, 0, 0)
    kick_to_tray(state, 0, [token])
    insert_from_tray(state, 1, [token], 1)
    apply_arrangement(snapshot, state)

    assert len(page_c.plot_widgets) == 2
    assert page_c.plot_widgets[1] is snapshot.tokens[token]
    assert page_c.plot_widgets[1].plot_widget.page is page_c
    # 未建页的结构字段由提交补齐（不再走 build_matrix）
    assert page_c._built_rows == 1 and page_c._built_cols == 2


def test_plot_moved_into_hidden_seat_stays_hidden(pages):
    """拖入隐藏格后必须保持不可见，否则 QGridLayout 会撑开隐藏行列。"""
    from src.core.plot_arrange_model import move_within

    page_a, _page_b, _page_c = pages
    page_a.rows_current = 1  # 2x2 矩阵：第二行变成隐藏区
    page_a.plot_layout.setRowStretch(1, 0)
    page_a.plot_widgets[2].setVisible(False)
    page_a.plot_widgets[3].setVisible(False)

    snapshot = snapshot_pages([page_a])
    state = build_state(snapshot)
    token = state.pages[0].cells[0]
    move_within(state, 0, [token], 2)  # 拖进隐藏行（seat2）
    summary = apply_arrangement(snapshot, state)

    assert page_a.plot_widgets[2] is snapshot.tokens[token]
    assert page_a.plot_widgets[2].isHidden() is True
    assert page_a.plot_widgets[1].isHidden() is False
    assert summary["hidden"] >= 1


def test_prepare_failure_leaves_pages_untouched(pages, monkeypatch):
    """预备阶段失败（新建空容器抛错）时，真实页面必须零改动。"""
    from src.core.plot_arrange_model import insert_from_tray, kick_to_tray

    page_a, page_b, _page_c = pages
    _set_curves(page_a.plot_widgets[3], ["f"])  # 页A 四个格全有图：无空容器可复用
    snapshot = snapshot_pages([page_a, page_b])
    state = build_state(snapshot)
    token = state.pages[0].cells[0]
    kick_to_tray(state, 0, [token])          # 页A 产生空位
    insert_from_tray(state, 1, [token], 1)   # 放进页B 空位，清空等候区
    assert state.tray == []

    before_a = list(page_a.plot_widgets)
    before_b = list(page_b.plot_widgets)
    before_links = [
        c.plot_widget.view_box.linkedView(0) for c in before_a + before_b
    ]

    def _boom(*_args, **_kwargs):
        raise RuntimeError("simulated allocation failure")

    monkeypatch.setattr(page_a, "_new_plot_container", _boom)
    import pytest as _pytest

    with _pytest.raises(RuntimeError):
        apply_arrangement(snapshot, state)

    assert list(page_a.plot_widgets) == before_a
    assert list(page_b.plot_widgets) == before_b
    assert page_a.plot_layout.count() == len(before_a)
    assert [
        c.plot_widget.view_box.linkedView(0) for c in page_a.plot_widgets + page_b.plot_widgets
    ] == before_links


def test_model_growth_builds_extra_cells_on_apply(pages):
    """对话框扩可见区后：提交把矩阵长大、可见行列与 stretch 一并落地。"""
    page_a, _page_b, _page_c = pages
    snapshot = snapshot_pages([page_a])
    state = build_state(snapshot)
    ps = state.pages[0]
    old_positions = {
        token: divmod(seat, ps.cols)
        for seat, token in enumerate(ps.cells)
        if token
    }

    ps.rows, ps.cols = 3, 3
    ps.vis_rows, ps.vis_cols = 3, 3
    ps.cells = [None] * 9
    for token, (row, col) in old_positions.items():
        ps.cells[row * 3 + col] = token

    apply_arrangement(snapshot, state)

    assert (page_a._built_rows, page_a._built_cols) == (3, 3)
    assert (page_a.rows_current, page_a.cols_current) == (3, 3)
    assert len(page_a.plot_widgets) == 9
    for container in page_a.plot_widgets:
        assert container.isHidden() is False
    for token, (row, col) in old_positions.items():
        assert page_a.plot_widgets[row * 3 + col] is snapshot.tokens[token]
    # 曲线一条不少
    assert sum(
        1 for c in page_a.plot_widgets if getattr(c.plot_widget, "curves", {})
    ) == len(old_positions)


def test_stale_snapshot_is_rejected(pages):
    page_a, _page_b, _page_c = pages
    snapshot = snapshot_pages([page_a])
    state = build_state(snapshot)
    # 对话框打开期间曲线发生变化 → 指纹失效
    _set_curves(page_a.plot_widgets[0], ["a", "b", "zzz"])
    assert validate_state(snapshot, state) is False

    import pytest as _pytest

    with _pytest.raises(ValueError):
        apply_arrangement(snapshot, state)


def test_page_keeps_container_count_after_noop_rearrange(pages):
    page_a, page_b, _page_c = pages
    snapshot = snapshot_pages([page_a, page_b])
    state = build_state(snapshot)
    token_b0 = _token_of(state, 1, 0)

    kick_to_tray(state, 1, [token_b0])
    insert_from_tray(state, 1, [token_b0], 0)  # 放回原位
    # 仅验证空位复用路径：页A 3 个空位的页面搬走后不再新建容器
    apply_arrangement(snapshot, state)
    assert len(page_a.plot_widgets) == 4
    assert all(
        _container_names_nonempty(c) for c in page_a.plot_widgets[:3]
    )


def _container_names_nonempty(container):
    pw = container.plot_widget
    return bool(getattr(pw, "curves", {}))