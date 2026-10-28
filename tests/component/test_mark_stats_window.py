"""MarkStatsWindow 重构后的组件测试。

覆盖设计文档 §7 计划：注册表校验 / 数值格式化 / Model 角色语义（NaN、inf、
SortRole 数值排序）/ 旧签名兼容 / 双路刷新 / 列配置持久化与排序列重映射 /
筛选控件。

本文件不启动真实 MainWindow：窗口直接以 parent=None 构造（几何恢复与
状态栏区间读取都防御了 None parent），数据用手造 MarkStatEntry 喂。
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QAction

from src.core.data_types import MarkStatEntry
from src.ui.mark_stats import (
    DEFAULT_COLUMN_KEYS,
    STAT_COLUMNS,
    CONTEXT_COLUMNS,
    SORT_ROLE,
    MarkStatsProxyModel,
    MarkStatsTableModel,
    MarkStatsWindow,
    StatRow,
    _format_stat_value,
)

# _show_header_menu 的 pos 参数仅用于 mapToGlobal，未显示窗口下取原点即可
_QT_POINT_ZERO = QPoint(0, 0)


def _entry(**over) -> MarkStatEntry:
    """一份各字段都合法的基准条目，用例按需覆盖单字段。"""
    base = dict(
        x1=1.0,
        x2=2.0,
        y1=10.0,
        y2=20.0,
        dx=1.0,
        dy=10.0,
        slope=10.0,
        label="a (km/h)",
        y_avg=15.0,
        y_max=20.0,
        y_min=10.0,
        var_name="a",
        unit="km/h",
        x_at_y_max=2.0,
        x_at_y_min=1.0,
        y_pp=10.0,
        count=5,
    )
    base.update(over)
    return MarkStatEntry(**base)


@pytest.fixture()
def stats_window(qapp, app_settings):
    """隔离的统计窗口：每个用例前把列配置/表头状态归零，防跨用例 ini 残留。"""
    app_settings.set_mark_stats_columns(list(DEFAULT_COLUMN_KEYS))
    app_settings.set_mark_stats_header_state("")
    window = MarkStatsWindow()
    yield window
    window.hide()  # 触发 hideEvent 保存，随后归零，不污染后续用例
    window.deleteLater()
    app_settings.set_mark_stats_columns(list(DEFAULT_COLUMN_KEYS))
    app_settings.set_mark_stats_header_state("")


class TestStatColumnRegistry:
    """§5.2 注册表：单一事实来源的结构不变量。"""

    def test_default_keys_are_registry_subset(self):
        assert DEFAULT_COLUMN_KEYS
        assert set(DEFAULT_COLUMN_KEYS) <= set(STAT_COLUMNS)

    def test_getters_readable_on_full_entry(self):
        e = _entry()
        for key, col in STAT_COLUMNS.items():
            value = col.getter(e)
            assert value is not None, f"{key} 取值为 None"

    def test_context_columns_are_fixed_four(self):
        assert [c.key for c in CONTEXT_COLUMNS] == ["page", "plot", "curve", "unit"]


class TestFormatStatValue:
    """§5.4 自适应精度：NaN/inf 特判 + 4 位有效数字。"""

    @pytest.mark.parametrize(
        "value,expected",
        [
            (1842.0, "1842"),
            (13.82, "13.82"),
            (0.0882, "0.0882"),
            (0.0, "0"),
            (-0.05, "-0.05"),
            (100.0, "100"),
            (7, "7"),
            (np.float64(13.82), "13.82"),
        ],
    )
    def test_adaptive_precision(self, value, expected):
        assert _format_stat_value(value) == expected

    def test_special_values(self):
        assert _format_stat_value(float("nan")) == "-"
        assert _format_stat_value(float("inf")) == "inf"
        assert _format_stat_value(float("-inf")) == "-inf"


class TestTableModelRoles:
    """§5.4 Model 角色语义：DisplayRole 格式化 + SortRole 数值。"""

    def _make_model(self, keys: list[str], rows: list[StatRow]):
        model = MarkStatsTableModel()
        model.set_column_keys(keys)
        model.reset_rows(rows)
        return model

    def test_display_roles_for_nan_inf_and_count(self):
        model = self._make_model(
            ["y_avg", "slope", "count"],
            [
                StatRow(
                    "p", 0,
                    _entry(y_avg=float("nan"), slope=float("inf"), count=123),
                )
            ],
        )
        nan_col = len(CONTEXT_COLUMNS) + 0
        inf_col = len(CONTEXT_COLUMNS) + 1
        cnt_col = len(CONTEXT_COLUMNS) + 2
        assert model.data(model.index(0, nan_col), Qt.ItemDataRole.DisplayRole) == "-"
        assert model.data(model.index(0, inf_col), Qt.ItemDataRole.DisplayRole) == "inf"
        assert (
            model.data(model.index(0, cnt_col), Qt.ItemDataRole.DisplayRole) == "123"
        )

    def test_sort_role_nan_pins_to_minus_inf_and_inf_passthrough(self):
        model = self._make_model(
            ["y_avg", "slope"],
            [StatRow("p", 0, _entry(y_avg=float("nan"), slope=float("inf")))],
        )
        nan_sort = model.data(
            model.index(0, len(CONTEXT_COLUMNS)), SORT_ROLE
        )
        assert nan_sort == float("-inf")
        inf_sort = model.data(
            model.index(0, len(CONTEXT_COLUMNS) + 1), SORT_ROLE
        )
        assert math.isinf(inf_sort) and inf_sort > 0

    def test_sort_role_numeric_ordering_via_proxy(self):
        rows = [
            StatRow("p", 0, _entry(var_name="a", label="a", y_avg=30.0)),
            StatRow("p", 0, _entry(var_name="b", label="b", y_avg=10.0)),
            StatRow("p", 1, _entry(var_name="c", label="c", y_avg=20.0)),
        ]
        model = self._make_model(["y_avg"], rows)
        proxy = MarkStatsProxyModel()
        proxy.setSourceModel(model)
        proxy.sort(len(CONTEXT_COLUMNS), Qt.SortOrder.AscendingOrder)
        names = [
            proxy.data(proxy.index(r, 2), Qt.ItemDataRole.DisplayRole)
            for r in range(proxy.rowCount())
        ]
        assert names == ["b", "c", "a"]  # 10 → 20 → 30，字符串排序会得到错误顺序

    def test_context_columns_display(self):
        model = self._make_model(
            ["dy"], [StatRow("page X", 3, _entry(var_name="speed", unit="km/h"))]
        )
        assert model.data(model.index(0, 0), Qt.ItemDataRole.DisplayRole) == "page X"
        assert model.data(model.index(0, 1), Qt.ItemDataRole.DisplayRole) == "4"  # +1 显示
        assert model.data(model.index(0, 2), Qt.ItemDataRole.DisplayRole) == "speed"
        assert model.data(model.index(0, 3), Qt.ItemDataRole.DisplayRole) == "km/h"
        # 中文表头（痛点 3）
        assert (
            model.headerData(2, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole)
            == "曲线"
        )


class TestUpdateStats:
    """§5.8 入口兼容与双路刷新。"""

    def test_old_style_list_normalized_to_single_page(self, stats_window):
        stats_window.update_stats([[_entry()], None])
        rows = stats_window.model.rows()
        assert len(rows) == 1
        assert rows[0].page == "图 1"
        assert rows[0].plot_idx == 0
        assert rows[0].entry.var_name == "a"
        assert stats_window._empty_plots == 1  # 空白子图只计数不生成行

    def test_new_style_tuples_grouped_by_page(self, stats_window):
        stats_window.update_stats(
            [
                ("page 1", 0, [_entry()]),
                ("page 2", 1, [_entry(var_name="b", label="b")]),
            ]
        )
        rows = stats_window.model.rows()
        assert [(r.page, r.plot_idx) for r in rows] == [
            ("page 1", 0),
            ("page 2", 1),
        ]

    def test_same_row_keys_refresh_inplace_without_reset(self, stats_window):
        resets: list[bool] = []
        changes: list[bool] = []
        stats_window.model.modelReset.connect(lambda: resets.append(True))
        stats_window.model.dataChanged.connect(lambda *a: changes.append(True))

        stats_window.update_stats([("p", 0, [_entry()])])
        assert len(resets) == 1

        stats_window.update_stats([("p", 0, [_entry(y_avg=99.0)])])
        assert len(resets) == 1, "行 key 集未变，不得走 reset"
        assert changes, "原位更新必须发 dataChanged"
        assert stats_window.model.rows()[0].entry.y_avg == 99.0

    def test_row_set_change_triggers_reset(self, stats_window):
        resets: list[bool] = []
        stats_window.model.modelReset.connect(lambda: resets.append(True))
        stats_window.update_stats([("p", 0, [_entry()])])
        stats_window.update_stats(
            [("p", 0, [_entry()]), ("p", 1, [_entry(var_name="b", label="b")])]
        )
        assert len(resets) == 2

    def test_clear_stats_empties_everything(self, stats_window):
        stats_window.update_stats([("p", 0, [_entry()])])
        stats_window.clear_stats()
        assert stats_window.model.rows() == []
        assert stats_window.proxy.rowCount() == 0


class TestColumnConfig:
    """§5.3/§5.6 列客制化：持久化 + 排序列按 key 重映射。"""

    def test_apply_column_keys_persisted_to_settings(
        self, stats_window, app_settings
    ):
        stats_window._apply_column_keys(["y_pp", "dy"])
        assert stats_window.model.column_keys() == ["y_pp", "dy"]
        assert app_settings.get_mark_stats_columns(
            DEFAULT_COLUMN_KEYS, STAT_COLUMNS.keys()
        ) == ["y_pp", "dy"]

    def test_column_config_survives_window_rebuild(self, qapp, app_settings):
        """模拟重启：新窗口实例从设置读回列配置。"""
        from tests.fixtures.waits import flush_deferred_deletes

        app_settings.set_mark_stats_columns(list(DEFAULT_COLUMN_KEYS))
        app_settings.set_mark_stats_header_state("")
        w1 = MarkStatsWindow()
        w1._apply_column_keys(["y_pp", "dy"])
        w1.hide()
        w1.deleteLater()
        flush_deferred_deletes()

        w2 = MarkStatsWindow()
        try:
            assert w2.model.column_keys() == ["y_pp", "dy"]
        finally:
            w2.hide()
            w2.deleteLater()
            flush_deferred_deletes()
        app_settings.set_mark_stats_columns(list(DEFAULT_COLUMN_KEYS))
        app_settings.set_mark_stats_header_state("")

    def test_sort_column_remaps_by_key_after_column_change(self, stats_window):
        stats_window.update_stats(
            [("p", 0, [_entry()]), ("p", 1, [_entry(var_name="b", label="b")])]
        )
        # dy 是默认列集第一项 → 列 4；按它排序后改列集，排序必须跟着 dy 走
        stats_window.proxy.sort(4, Qt.SortOrder.DescendingOrder)
        assert stats_window.proxy.sortColumn() == 4

        stats_window._apply_column_keys(["slope", "dy"])  # dy 挪到列 5
        assert stats_window.proxy.sortColumn() == 5
        assert stats_window.proxy.sortOrder() == Qt.SortOrder.DescendingOrder

        stats_window._apply_column_keys(["slope"])  # dy 被删 → 排序清除
        assert stats_window.proxy.sortColumn() == -1


class _HeaderMenuStub:
    """替换 mark_stats 模块内的 QMenu 符号：免模态 exec，捕获菜单内容。

    PySide6 的 QMenu.exec 是 C++ 方法，monkeypatch 类属性拦不住实例调用
    （真实模态 exec 会阻塞到超时）；stub 化模块符号后整条
    _show_header_menu 路径可以纯同步驱动。
    """

    # 测试前设置要"点击"的菜单项标题；None = 模拟弹出后不选择
    click_title: str | None = None
    last: "_HeaderMenuStub | None" = None

    def __init__(self, parent=None):
        self._actions: list = []
        self._parent = parent
        _HeaderMenuStub.last = self

    def addAction(self, text):
        act = QAction(text)
        self._actions.append(act)
        return act

    def addSeparator(self):
        act = QAction()
        act.setSeparator(True)
        self._actions.append(act)
        return act

    def actions(self):
        return list(self._actions)

    def exec(self, _pos):
        for act in self._actions:
            if act.text() == _HeaderMenuStub.click_title:
                return act
        return None


class TestHeaderMenu:
    """§5.6 表头右键菜单：勾选载荷与端到端生效。

    回归背景：曾因漏 ``act.setData(key)`` 导致 ``chosen.data()`` 恒为
    None，任何勾选都静默失效（key 变成 "None" 字面量被注册表过滤）。
    """

    @pytest.fixture()
    def header_menu(self, stats_window, monkeypatch):
        """注入菜单 stub 并在用例结束后恢复。"""
        import src.ui.mark_stats as mark_stats_module

        monkeypatch.setattr(mark_stats_module, "QMenu", _HeaderMenuStub)
        _HeaderMenuStub.click_title = None
        _HeaderMenuStub.last = None
        yield _HeaderMenuStub
        _HeaderMenuStub.click_title = None
        _HeaderMenuStub.last = None

    def _open_menu(self, stats_window, click: str | None = None):
        _HeaderMenuStub.click_title = click
        stats_window._show_header_menu(_QT_POINT_ZERO)
        menu = _HeaderMenuStub.last
        assert menu is not None
        return menu

    def test_every_checkable_action_carries_valid_key(
        self, stats_window, header_menu
    ):
        """勾选项必须携带注册表合法 key（漏 setData 的直接回归测试）。"""
        menu = self._open_menu(stats_window)
        checkable = [a for a in menu.actions() if a.isCheckable()]
        assert len(checkable) == len(STAT_COLUMNS)
        for act in checkable:
            data = act.data()
            assert isinstance(data, str), f"'{act.text()}' 未携带 key 载荷"
            assert data in STAT_COLUMNS

    def test_checking_metric_appends_column(self, stats_window, header_menu):
        """勾选「样本数 n」→ count 列追加到末尾。"""
        self._open_menu(stats_window, click="样本数 n")

        keys = stats_window.model.column_keys()
        assert keys[-1] == "count"
        assert stats_window.model.columnCount() == len(CONTEXT_COLUMNS) + len(keys)
        # 表头确实多出「样本数 n」列
        header_texts = [
            stats_window.model.headerData(
                c, Qt.Orientation.Horizontal, Qt.ItemDataRole.DisplayRole
            )
            for c in range(stats_window.model.columnCount())
        ]
        assert "样本数 n" in header_texts

    def test_unchecking_metric_removes_column(
        self, stats_window, header_menu, app_settings
    ):
        """取消勾选「均值 y_avg」→ y_avg 列移除并持久化。"""
        self._open_menu(stats_window, click="均值 y_avg")

        keys = stats_window.model.column_keys()
        assert "y_avg" not in keys
        assert app_settings.get_mark_stats_columns(
            DEFAULT_COLUMN_KEYS, STAT_COLUMNS.keys()
        ) == keys

    def test_last_remaining_metric_cannot_be_unchecked(
        self, stats_window, header_menu
    ):
        """仅剩 1 列指标时该项禁用取消。"""
        stats_window._apply_column_keys(["dy"])
        menu = self._open_menu(stats_window)
        dy_act = next(a for a in menu.actions() if a.text() == "变化量 dy")
        assert dy_act.isChecked() is True
        assert dy_act.isEnabled() is False


class TestFilters:
    """§5.5 筛选：曲线名搜索 / 页筛选 / 仅当前页 / 单页退化。"""

    def _two_pages_two_curves(self, window):
        window.update_stats(
            [
                ("page 1", 0, [_entry(var_name="alpha", label="alpha")]),
                ("page 1", 0, [_entry(var_name="beta", label="beta")]),
                ("page 2", 0, [_entry(var_name="alpha2", label="alpha2")]),
            ],
            active_page="page 1",
        )

    def test_search_filters_curve_name_case_insensitive(self, stats_window):
        self._two_pages_two_curves(stats_window)
        stats_window.current_page_only.setChecked(False)  # 关闭页过滤，全页搜索
        stats_window.search_input.setText("ALP")
        assert stats_window.proxy.rowCount() == 2  # alpha + alpha2

    def test_current_page_only_follows_active_page(self, stats_window):
        self._two_pages_two_curves(stats_window)
        assert stats_window.current_page_only.isChecked()
        pages = {
            stats_window.proxy.data(
                stats_window.proxy.index(r, 0), Qt.ItemDataRole.DisplayRole
            )
            for r in range(stats_window.proxy.rowCount())
        }
        assert pages == {"page 1"}

        stats_window.update_stats(
            [
                ("page 1", 0, [_entry(var_name="alpha", label="alpha")]),
                ("page 2", 0, [_entry(var_name="alpha2", label="alpha2")]),
            ],
            active_page="page 2",
        )
        pages = {
            stats_window.proxy.data(
                stats_window.proxy.index(r, 0), Qt.ItemDataRole.DisplayRole
            )
            for r in range(stats_window.proxy.rowCount())
        }
        assert pages == {"page 2"}

    def test_page_combo_selects_specific_page(self, stats_window):
        self._two_pages_two_curves(stats_window)
        stats_window.current_page_only.setChecked(False)
        idx = stats_window.page_combo.findData("page 2")
        stats_window.page_combo.setCurrentIndex(idx)
        assert stats_window.proxy.rowCount() == 1

    def test_single_page_hides_page_controls(self, stats_window):
        stats_window.update_stats([("图 1", 0, [_entry()])])
        # 窗口未 show()，isVisible() 恒 False；用 isVisibleTo(窗口) 只看控件自身
        assert stats_window.page_combo.isVisibleTo(stats_window) is False
        assert stats_window.current_page_only.isVisibleTo(stats_window) is False
        # 单页退化不做过滤：行原样可见
        assert stats_window.proxy.rowCount() == 1


class TestExport:
    """§5.7 导出：所见即所得（筛选后可见行 + 视觉列序）。"""

    def test_export_csv_and_markdown(self, stats_window):
        stats_window.update_stats(
            [("p", 0, [_entry(var_name="alpha", label="alpha")]),
             ("p", 1, [_entry(var_name="beta", label="beta", y_avg=float("nan"))])]
        )
        stats_window.search_input.setText("alpha")
        csv_text = stats_window._export_text(as_csv=True)
        lines = csv_text.splitlines()
        assert lines[0].split("\t")[:4] == ["页", "子图", "曲线", "单位"]
        assert len(lines) == 2  # 表头 + 1 行（beta 被筛掉）
        assert "alpha" in lines[1]

        md_text = stats_window._export_text(as_csv=False)
        assert md_text.splitlines()[0].startswith("| 页 | 子图 | 曲线 | 单位 |")
        assert "alpha" in md_text.splitlines()[2]

    def test_status_bar_counts_and_range(self, stats_window):
        from PySide6.QtWidgets import QWidget

        holder = QWidget()  # 状态栏区间来源读取 parent.saved_mark_range
        holder.saved_mark_range = (1.5, 9.5)
        stats_window.setParent(holder)
        try:
            stats_window.update_stats(
                [("p", 0, [_entry()]), ("p", 1, None)]
            )
            text = stats_window.status_label.text()
            assert "1 页" in text and "2 子图" in text
            assert "1 条曲线" in text
            assert "标记区间 [1.5, 9.5]" in text
            assert "1 个空子图未统计" in text
        finally:
            stats_window.setParent(None)
            holder.deleteLater()
