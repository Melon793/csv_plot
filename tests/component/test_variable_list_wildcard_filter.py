"""左栏变量/单位过滤的通配符支持（* / ?）component 回归测试。

过滤语义保持不变：空格分隔多关键词、任一命中即保留（OR）；
仅当关键词含 `*` 或 `?` 时启用通配，普通关键词仍走原子串匹配。
"""

from __future__ import annotations

import pytest

from PySide6.QtCore import Qt

from src.ui.variable_list import MyTableWidget

NAMES = [
    "demo__rl_w_msg",
    "demo__fl_w_msg",
    "demo__rl_r_msg",
    "demo__diag_msg_1",
    "engine_rpm",
    "temp(ENG01).oil",
    "mass_flow",
]
UNITS = {
    "demo__rl_w_msg": "bool",
    "demo__fl_w_msg": "bool",
    "demo__rl_r_msg": "bool",
    "demo__diag_msg_1": "bool",
    "engine_rpm": "rpm",
    "temp(ENG01).oil": "degC",
    "mass_flow": "kg*m/s^2",
}


@pytest.fixture()
def table(qapp):
    widget = MyTableWidget(None)
    widget.populate(list(NAMES), dict(UNITS), {n: 1 for n in NAMES})
    yield widget
    widget.deleteLater()


def _visible_names(table):
    rows = [
        (row, table.item(row, 0).data(Qt.ItemDataRole.UserRole + 1))
        for row in range(table.rowCount())
    ]
    return [name for row, name in rows if not table.isRowHidden(row)]


def test_star_wildcard_narrows(table):
    assert table.hide_non_matching(["demo__*_msg"], [], UNITS) is True
    assert set(_visible_names(table)) == {
        "demo__rl_w_msg",
        "demo__fl_w_msg",
        "demo__rl_r_msg",
        "demo__diag_msg_1",
    }


def test_question_wildcard_matches_single_char(table):
    table.hide_non_matching(["demo__r?_w_msg"], [], UNITS)
    assert _visible_names(table) == ["demo__rl_w_msg"]


def test_plain_keyword_unchanged(table):
    table.hide_non_matching(["engine_rpm"], [], UNITS)
    assert _visible_names(table) == ["engine_rpm"]


def test_or_semantics_preserved_with_wildcards(table):
    table.hide_non_matching(["engine_*", "temp*"], [], UNITS)
    assert set(_visible_names(table)) == {"engine_rpm", "temp(ENG01).oil"}


def test_literal_parens_with_wildcard(table):
    table.hide_non_matching(["temp(ENG01).*"], [], UNITS)
    assert _visible_names(table) == ["temp(ENG01).oil"]


def test_unit_filter_supports_wildcard(table):
    table.hide_non_matching([], ["r?m"], UNITS)
    assert _visible_names(table) == ["engine_rpm"]


def test_text_star_in_unit_is_literal(table):
    """单位值自身含 `*`（如 kg*m/s^2）时，通配符仍能正常匹配它"""
    table.hide_non_matching([], ["k*g"], UNITS)
    assert _visible_names(table) == ["mass_flow"]


def test_no_match_hides_all(table):
    table.hide_non_matching(["zzz_*_zzz"], [], UNITS)
    assert _visible_names(table) == []


def test_reapply_after_sort_keeps_pattern(table):
    table.hide_non_matching(["demo__*_msg"], [], UNITS)
    table.sortItems(0, Qt.SortOrder.DescendingOrder)
    assert set(_visible_names(table)) == {
        "demo__rl_w_msg",
        "demo__fl_w_msg",
        "demo__rl_r_msg",
        "demo__diag_msg_1",
    }
