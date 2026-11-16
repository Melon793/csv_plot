"""列容量精确口径的单元护栏。

n 个标签实际需要 n*h + (n-1)*gap；旧公式 n*(h+gap) 会多要 1 个 gap，
临界行高误判容量 n-1、白白多拆一列（现场“刚好放下 6 个却跳两列”）。
"""

from __future__ import annotations

import pytest

from src.ui.widgets.cursor_manager import CursorManager


@pytest.mark.parametrize("h,gap", [(23.0, 2.0), (23.0, 4.0), (17.0, 2.0)])
def test_exact_fit_boundary(h, gap):
    # n=1 例外：极矮兜底要求 available ≥ h+gap，见 test_single_label_threshold
    for n in range(2, 9):
        need = n * h + (n - 1) * gap
        assert CursorManager._column_capacity(need, h, gap) == n
        # 恰好少 0.6px → 放不下 n 个
        assert CursorManager._column_capacity(need - 0.6, h, gap) == n - 1
        # 多 0.6px → 仍是 n（未到 n+1）
        assert CursorManager._column_capacity(need + 0.6, h, gap) == n


def test_tiny_height_keeps_hide_threshold():
    """极矮兜底沿用 available < h+gap → 0（旧语义不变）"""
    h, gap = 23.0, 2.0
    assert CursorManager._column_capacity(h + gap - 0.1, h, gap) == 0
    assert CursorManager._column_capacity(h + gap, h, gap) == 1
    assert CursorManager._column_capacity(h - 1.0, h, gap) == 0


def test_single_label_threshold():
    """1 个标签也沿用旧隐藏阈值：available < h+gap 时整列隐藏"""
    h, gap = 23.0, 2.0
    assert CursorManager._column_capacity(h + gap, h, gap) == 1
    assert CursorManager._column_capacity(h + gap - 0.1, h, gap) == 0
    assert CursorManager._column_capacity(h, h, gap) == 0


def test_matches_old_formula_for_n_ge_2():
    """对 n≥2，新口径恰好比旧口径多容纳 1 个（仅当 available 落在
    旧公式 n*(h+gap) 与精确 n*h+(n-1)*gap 之间时）"""
    h, gap = 23.0, 2.0
    n = 6
    need = n * h + (n - 1) * gap
    old = int(need / (h + gap))  # 5（旧公式的误判）
    assert old == n - 1
    assert CursorManager._column_capacity(need, h, gap) == n