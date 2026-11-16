"""列方向分配（双侧交替）单元护栏。

2026-11-16 拍板口径：
- 单游标（alternate=True）：列方向 右/左/右… 交替；偏好侧无预算换另
  一侧；两侧都无预算时追加到占用较少的一侧（允许列重叠——重叠优于
  消失，不新增隐藏路径）；
- 双游标（alternate=False）：沿用旧「右先左后 + 越界压右」，中间缝隙
  视为禁区；
- force_side 优先级最高（跨组向外分裂）。
"""

from __future__ import annotations

from src.ui.widgets.cursor_manager import CursorManager


MAX_W = 40.0
COL_GAP = 10.0
COL_W = MAX_W + COL_GAP                       # 单列占用宽 = 50
MARGIN = CursorManager.COLUMN_BUDGET_MARGIN   # 预算余量 = 2
UNIT = COL_W + MARGIN                         # 一列最低可用宽 = 52


def _sides(count, right, left, **kwargs):
    return CursorManager._assign_column_sides(
        count, right, left, MAX_W, COL_GAP, **kwargs
    )


def _only_sides(pairs) -> list[str]:
    return [side for side, _ in pairs]


def test_single_column_goes_right():
    assert _sides(1, 500, 500) == [('right', 0)]


def test_two_columns_split_right_left():
    assert _sides(2, 500, 500) == [('right', 0), ('left', 0)]


def test_three_columns_alternate_right_left_right():
    pairs = _sides(3, 500, 500)
    assert pairs == [('right', 0), ('left', 0), ('right', 1)]
    # 每侧 col_idx 各自从 0 递增（同侧累计偏移依赖该语义）
    assert _only_sides(pairs) == ['right', 'left', 'right']


def test_no_budget_on_one_side_falls_back_to_other():
    # 可用宽差 1px 不到「列宽 + 余量」→ 该侧 0 列（margin 口径）
    assert _only_sides(_sides(3, 500, UNIT - 1)) == ['right'] * 3
    assert _only_sides(_sides(3, UNIT - 1, 500)) == ['left'] * 3
    # 恰好等于「列宽 + 余量」→ 该侧可放 1 列
    assert _only_sides(_sides(2, UNIT, UNIT)) == ['right', 'left']


def test_both_sides_without_budget_appends_to_less_used_side():
    """两侧都无预算的防御分支：按占用较少侧追加，不抛异常、列数不变。

    统一口径下 max_cols 截断与方向分配共用 _possible_columns（含
    COLUMN_BUDGET_MARGIN），调用方不会把列数截到两侧预算之外，因此该
    分支在正常布局路径中不可达；这里只锁「重叠优于消失」的防御语义。
    """
    pairs = _sides(3, 0.0, 0.0)
    assert len(pairs) == 3, "列数不得因无预算而减少（不新增隐藏路径）"
    assert _only_sides(pairs) == ['right', 'left', 'right']
    assert [idx for _, idx in pairs] == [0, 0, 1]


def test_alternate_false_keeps_legacy_right_first():
    # 双游标：右先左后（右侧只够 2 列 → 第 3 列才落到左侧）
    assert _only_sides(_sides(3, 110, 500, alternate=False)) == [
        'right', 'right', 'left',
    ]
    # 仍超出两侧预算 → 其余列压回右侧，不进中间缝隙
    assert _only_sides(_sides(5, 110, 60, alternate=False)) == [
        'right', 'right', 'left', 'right', 'right',
    ]


def test_force_side_overrides_alternation_and_budget():
    for side in ('left', 'right'):
        assert _only_sides(_sides(3, 500, 500, force_side=side)) == [side] * 3
        # 强制侧预算为 0 也照样全放（跨组向外分裂最高优先级）
        assert _only_sides(
            _sides(2, 0.0, 0.0, force_side=side)
        ) == [side] * 2