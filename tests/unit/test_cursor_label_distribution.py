"""列内分配策略的单元护栏。

现场诉求：容量有余时，后续标签应继续留在第一列（5+1），
而不是平均切 3+3；列数被宽度/滞回截断时退回均分，避免末列 4+6。
"""

from __future__ import annotations

from src.ui.widgets.cursor_manager import CursorManager


def _sizes(labels: list, cols: int, cap: int) -> list[int]:
    groups = CursorManager._distribute_labels(labels, cols, cap)
    assert sum(len(g) for g in groups) == len(labels)
    assert [x for g in groups for x in g] == labels, "不得丢标签/改顺序"
    return [len(g) for g in groups]


def test_fill_to_capacity():
    labels = list(range(6))
    assert _sizes(labels, 2, 5) == [5, 1]
    assert _sizes(labels, 2, 4) == [4, 2]


def test_multi_column_fill():
    labels = list(range(10))
    assert _sizes(labels, 3, 4) == [4, 4, 2]
    assert _sizes(labels, 2, 10) == [9, 1]


def test_single_column_when_capacity_enough():
    assert _sizes(list(range(5)), 1, 5) == [5]
    # 强制两列时仍优先填满第一列
    assert _sizes(list(range(5)), 2, 9) == [4, 1]


def test_capacity_one():
    assert _sizes(list(range(4)), 4, 1) == [1, 1, 1, 1]


def test_fallback_to_even_when_capacity_exceeded():
    """列数被截断（n > cap*cols）→ 退回均分，不得 4+6"""
    labels = list(range(10))
    assert _sizes(labels, 2, 4) == [5, 5]
    assert _sizes(list(range(8)), 2, 3) == [4, 4]