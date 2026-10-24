"""PlotPage 网格步长与增量补格（设计文档 §5.3 / 决策 D-C）。

新建页只建 ``rows_current × cols_current``，扁平列表的列步长必须用实际
构建列数 ``grid_cols``；扩大布局时只补缺失格，已有子图对象必须原样保留。
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.ui.widgets.plot_page import PlotPage


class _FakeButton:
    def __init__(self, checked: bool = False):
        self._checked = checked

    def isChecked(self) -> bool:
        return self._checked


class _FakeLayoutManager:
    def __init__(self):
        self.toggle_calls = []
        self.stats_calls = []

    def toggle_mark_region(self, enabled: bool):
        self.toggle_calls.append(enabled)

    def request_mark_stats_refresh(self, *, immediate: bool = False):
        self.stats_calls.append(immediate)


class _FakeCursorSync:
    def __init__(self):
        self.sync_calls = 0

    def _sync_min_xrange(self):
        self.sync_calls += 1

    def collect_global_x_range(self):
        return (0.0, 1.0)


class _FakeServices:
    """PlotPage.build_matrix 所需的最小服务集合。"""

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


@pytest.fixture()
def page(qapp, qtbot):
    fake = _FakeServices()
    page = PlotPage(
        parent=None,
        title="图 1",
        rows_max=4,
        cols_max=3,
        rows_current=1,
        cols_current=1,
    )
    page.services = fake
    qtbot.addWidget(page.widget)
    yield page
    page.dispose()


def test_partial_grid_uses_actual_column_stride(page):
    """只建 2×1 时列步长是 1，不能按容量 3 反算行列。"""
    page.build_matrix(2, 1)
    assert page.grid_cols == 1
    assert len(page.plot_widgets) == 2
    assert page.container_at(1, 0) is page.plot_widgets[1]
    assert page.container_at(0, 1) is None
    assert page.row_of(page.plot_widgets[1].plot_widget) == 1

    page.apply_visibility(2, 1)
    assert [page.cell_visible(i) for i in range(2)] == [True, True]
    # 容量第 2/3 列没有容器，不能残留 stretch（否则右侧多出空白）
    assert [
        page.plot_layout.columnStretch(c) for c in range(page.cols_max)
    ] == [1, 0, 0]


def test_same_size_reapply_leaves_no_stretch_for_phantom_column(page):
    # 新页实际只建 3×1，重复应用 3×1 也不能给容量列发 stretch
    page.build_matrix(3, 1)
    page.apply_visibility(3, 1)
    assert page.grid_cols == 1
    assert [
        page.plot_layout.columnStretch(c) for c in range(page.cols_max)
    ] == [1, 0, 0]


def test_shrink_to_two_columns_leaves_no_stretch_for_phantom_column(page):
    page.build_matrix(3, 1)
    page.apply_visibility(3, 2)
    assert page.grid_cols == 2
    assert [
        page.plot_layout.columnStretch(c) for c in range(page.cols_max)
    ] == [1, 1, 0]


def test_shrink_to_one_column_keeps_only_first_column(page):
    page.build_matrix(3, 2)
    page.apply_visibility(3, 1)
    assert [
        page.plot_layout.columnStretch(c) for c in range(page.cols_max)
    ] == [1, 0, 0]


def test_three_columns_keep_all_column_stretches(page):
    page.build_matrix(3, 1)
    page.apply_visibility(3, 3)
    assert [
        page.plot_layout.columnStretch(c) for c in range(page.cols_max)
    ] == [1, 1, 1]


def test_ensure_matrix_first_build_uses_current_rows_cols(page):
    page.rows_current, page.cols_current = 2, 2
    created = page.ensure_matrix()
    assert created is True
    assert page.grid_cols == 2
    assert len(page.plot_widgets) == 4
    # 第二次无变化
    assert page.ensure_matrix() is False


def test_grow_keeps_existing_containers_and_remaps_cells(page):
    page.build_matrix(2, 1)
    old_first, old_second = page.plot_widgets
    first_pw = old_first.plot_widget

    created = page.ensure_matrix(2, 2)
    assert created is True
    assert page.grid_cols == 2
    assert len(page.plot_widgets) == 4
    # 原两格按 (r,c) 保留，不被重建
    assert page.container_at(0, 0) is old_first
    assert page.container_at(1, 0) is old_second
    assert page.plot_widgets[0].plot_widget is first_pw
    # 新格补在 (0,1)/(1,1)
    assert page.container_at(0, 1) is not None
    assert page.container_at(1, 1) is not None
    assert page.row_of(page.plot_widgets[2].plot_widget) == 1


def test_apply_visibility_grows_partial_page(page):
    """新页继承 1×1，用户把布局改成 2×2：只补格，不重建。"""
    page.ensure_matrix()
    assert len(page.plot_widgets) == 1
    first = page.plot_widgets[0]

    page.apply_visibility(2, 2)
    assert page.grid_cols == 2
    assert len(page.plot_widgets) == 4
    assert page.plot_widgets[0] is first
    assert page.rows_current == 2 and page.cols_current == 2
    assert all(page.cell_visible(i) for i in range(4))
    assert page.container_at(1, 1) is page.plot_widgets[3]
