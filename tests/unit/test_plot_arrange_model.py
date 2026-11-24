"""整理布局纯模型测试（设计稿 v4 语义）。

覆盖：恒等式、可见区优先序补洞、L 形隐藏掩码、交换局部性、
块交换重叠不丢图、顶替一进一出、门禁与回填。
"""

from __future__ import annotations

from src.core.plot_arrange_model import (
    ArrangeState,
    PageState,
    backfill,
    can_commit,
    clone_state,
    insert_from_tray,
    kick_to_tray,
    move_within,
    ordered_ids,
    order_of,
    seat_visible,
    total_plots,
)


def make_state() -> ArrangeState:
    """3 页演示态：页1 3x3 可见2x2（L形隐藏）；页2 2x3 可见2x2；页3 1x3 全可见。"""
    page1 = PageState(
        "页1", rows=3, cols=3, vis_rows=2, vis_cols=2,
        cells=["a", "b", "c", None, "d", "e", "f", None, "g"],
    )
    page2 = PageState(
        "页2", rows=2, cols=3, vis_rows=2, vis_cols=2,
        cells=["h", "i", None, "j", "k", "l"],
    )
    page3 = PageState(
        "页3", rows=1, cols=3, vis_rows=1, vis_cols=3,
        cells=["m", None, "n"],
    )
    return ArrangeState([page1, page2, page3], [])


def test_total_plots_constant_after_kick_and_cross_page_insert():
    state = make_state()
    before = total_plots(state)
    assert before == 14
    kick_to_tray(state, 0, ["a"])
    kick_to_tray(state, 1, ["h"])  # 目标页腾出空位
    result = insert_from_tray(state, 1, ["a"], 0)
    assert result["ok"] is True
    assert total_plots(state) == before


def test_kick_compacts_in_visibility_first_order():
    state = make_state()
    page = state.pages[1]
    kick_to_tray(state, 1, ["h"])
    # 剩 4 图占可见序前 4 席（seat0/1/3/4）；右列 seat2 必须是空位
    assert page.cells[2] is None
    assert page.cells[3] is not None
    assert page.cells[4] is not None
    assert ordered_ids(page)[0] == "i"


def test_l_shaped_hidden_mask():
    page = make_state().pages[0]
    assert seat_visible(page, 2) is False   # 右列（1行3列）
    assert seat_visible(page, 6) is False   # 底行
    assert seat_visible(page, 3) is True    # 第二行第一列仍可见
    order = order_of(page)
    assert order[:4] == [0, 1, 3, 4]        # 可见区优先
    assert order[4:] == [2, 5, 6, 7, 8]     # 再隐藏区


def test_place_into_hole_only_moves_two_cells():
    state = make_state()
    page = state.pages[1]
    before = list(page.cells)
    move_within(state, 1, ["h"], 2)  # seat2 是空位
    assert page.cells[2] == "h"
    assert page.cells[0] is None
    assert page.cells[1] == before[1]
    assert page.cells[3] == before[3]


def test_in_page_swap_locality_2_2_to_2_3():
    """投诉案例：只动源/目标两格，隐藏右列不受牵连。"""
    state = make_state()
    page = state.pages[1]
    before = list(page.cells)
    move_within(state, 1, [before[4]], 5)
    diff = [i for i in range(len(before)) if before[i] != page.cells[i]]
    assert diff == [4, 5]
    assert page.cells[5] == before[4]
    assert page.cells[4] == before[5]


def test_full_page_replace_one_in_one_out():
    state = make_state()
    # 借页1一图把页3填满
    kick_to_tray(state, 0, ["a"])
    insert_from_tray(state, 2, ["a"], 1)
    assert len(ordered_ids(state.pages[2])) == 3
    kick_to_tray(state, 0, ["b"])
    occupant = state.pages[2].cells[0]
    result = insert_from_tray(state, 2, ["b"], 0)
    assert result["ok"] is True
    assert result["displaced"] == [occupant]
    assert state.tray == [occupant]
    assert len(ordered_ids(state.pages[2])) == 3


def test_non_full_page_replace_also_displaces():
    state = make_state()
    kick_to_tray(state, 0, ["a"])
    holes_before = sum(1 for c in state.pages[1].cells if c is None)
    occupant = state.pages[1].cells[0]
    result = insert_from_tray(state, 1, ["a"], 0)
    assert result["displaced"] == [occupant]
    assert state.tray == [occupant]
    holes_after = sum(1 for c in state.pages[1].cells if c is None)
    assert holes_after == holes_before


def test_backfill_fills_tail_holes():
    state = make_state()
    kick_to_tray(state, 0, ["a", "b"])
    placed = backfill(state, 0)
    assert placed == 2
    assert len(ordered_ids(state.pages[0])) == 7
    assert state.tray == []


def test_gate_requires_empty_tray():
    state = make_state()
    assert can_commit(state) is True
    kick_to_tray(state, 0, ["a"])
    assert can_commit(state) is False


def test_block_exchange_keeps_internal_order():
    state = make_state()
    page = state.pages[0]
    move_within(state, 0, ["a", "b"], 3)
    # 目标段 seat3/4 原本是 [None, 'd'] → 原图块回源段，块占目标段
    assert page.cells[3] == "a"
    assert page.cells[4] == "b"
    assert page.cells[0] == "d"
    assert page.cells[1] is None
    assert total_plots(state) == 14


def test_block_lands_in_visible_zone_order_not_hidden_column():
    """可见 2x2：块拖到 1-2 只填可见序 1-2、2-1，不跨进隐藏的 1-3。"""
    state = make_state()
    page = state.pages[1]  # 2x3 建，2x2 可见
    before = list(page.cells)
    result = move_within(state, 1, ["h", "i"], 1)  # 1-2 在格位 seat1
    assert result["ok"] is True
    assert page.cells[1] == "h"   # 1-2
    assert page.cells[3] == "i"   # 2-1（可见序下一个），不是隐藏的 1-3(seat2)
    assert page.cells[2] is None  # 隐藏右列原样
    assert page.cells[0] == before[3]  # 被顶下图的回到源段
    assert {c for c in page.cells if c} == {c for c in before if c}
    assert total_plots(state) == 14


def test_block_exchange_overlapping_source_and_target_keeps_all_plots():
    state = make_state()
    page = state.pages[0]
    ids_before = {c for c in page.cells if c}
    move_within(state, 0, ["a", "b"], 1)  # src seat0/1，可见序 tgt seat1/3
    assert page.cells[0] is None
    assert page.cells[1] == "a"
    assert page.cells[2] == "c"   # 隐藏右列不受影响
    assert page.cells[3] == "b"
    assert {c for c in page.cells if c} == ids_before
    assert total_plots(state) == 14


def test_block_over_hole_and_overlap_keeps_displaced_plot():
    """评审 P1 回归：目标段含空位且与源重叠时，被顶替图不得被覆盖丢失。"""
    page = PageState(
        "页", rows=2, cols=2, vis_rows=2, vis_cols=2,
        cells=["a", "X", None, "d"],
    )
    state = ArrangeState([page], [])
    result = move_within(state, 0, ["a", "d"], 0)
    assert result["ok"] is True
    assert page.cells[0] == "a" and page.cells[1] == "d"
    assert page.cells[3] == "X"      # 被顶替图回到源段中不在目标段的格
    assert page.cells[2] is None
    assert {c for c in page.cells if c} == {"a", "X", "d"}


def test_clone_state_is_deep():
    state = make_state()
    copied = clone_state(state)
    kicked = kick_to_tray(copied, 0, ["a"])
    assert kicked == 1
    assert "a" in state.pages[0].cells  # 原状态未被污染
    assert copied.pages[0].cells != state.pages[0].cells

# --------------------------------------------------------------------------
# Q1b：连续块刚性平移（anchor + bounds 才启用；散选回退打包）
# --------------------------------------------------------------------------
def _state(page: PageState) -> ArrangeState:
    return ArrangeState([page], [])


def test_rigid_vertical_block_moves_right_keeping_rows():
    page = PageState("页", rows=3, cols=2, vis_rows=3, vis_cols=2,
                     cells=["a", None, "b", None, "c", None])
    result = move_within(_state(page), 0, ["a", "b", "c"], 1,
                         anchor="a", bounds=(3, 2))
    assert result["ok"] is True
    assert result["mode"] == "rigid"
    assert page.cells == [None, "a", None, "b", None, "c"]
    assert result["anchor_seat"] == 1
    assert result["clamped"] is False
    assert result["moved"] is True


def test_rigid_anchor_in_middle_lands_on_target():
    page = PageState("页", rows=3, cols=2, vis_rows=3, vis_cols=2,
                     cells=["a", None, "b", None, "c", None])
    result = move_within(_state(page), 0, ["a", "b", "c"], 3,
                         anchor="b", bounds=(3, 2))
    assert result["mode"] == "rigid"
    assert result["anchor_seat"] == 3
    assert page.cells[1] == "a" and page.cells[3] == "b" and page.cells[5] == "c"


def test_rigid_clamps_at_edge_and_highlights_edge_seat():
    page = PageState("页", rows=3, cols=2, vis_rows=3, vis_cols=2,
                     cells=["a", None, "b", None, "c", None])
    result = move_within(_state(page), 0, ["a", "b", "c"], 3,
                         anchor="a", bounds=(3, 2))
    assert result["mode"] == "rigid"
    assert result["clamped"] is True
    assert result["moved"] is True
    assert result["anchor_seat"] == 1   # row0 col1（贴边后停在这里）
    assert page.cells[1] == "a" and page.cells[3] == "b" and page.cells[5] == "c"


def test_rigid_no_displacement_when_already_at_edge():
    page = PageState("页", rows=3, cols=2, vis_rows=3, vis_cols=2,
                     cells=["a", None, "b", None, "c", None])
    before = list(page.cells)
    result = move_within(_state(page), 0, ["a", "b", "c"], 4,
                         anchor="a", bounds=(3, 2))
    assert result["mode"] == "rigid"
    assert result["clamped"] is True
    assert result["moved"] is False
    assert result["anchor_seat"] == 0
    assert page.cells == before


def test_rigid_horizontal_block_moves_whole_row():
    page = PageState("页", rows=2, cols=3, vis_rows=2, vis_cols=3,
                     cells=["a", "b", "c", None, None, None])
    result = move_within(_state(page), 0, ["a", "b", "c"], 3,
                         anchor="a", bounds=(2, 3))
    assert result["mode"] == "rigid"
    assert page.cells[3:] == ["a", "b", "c"]
    assert page.cells[:3] == [None, None, None]


def test_rigid_full_rect_moves():
    page = PageState("页", rows=3, cols=3, vis_rows=3, vis_cols=3,
                     cells=["a", "b", None, "c", "d", None, None, None, None])
    result = move_within(_state(page), 0, ["a", "b", "c", "d"], 4,
                         anchor="a", bounds=(3, 3))
    assert result["mode"] == "rigid"
    assert page.cells[4] == "a" and page.cells[5] == "b"
    assert page.cells[7] == "c" and page.cells[8] == "d"
    assert sorted(c for c in page.cells if c) == ["a", "b", "c", "d"]


def test_rigid_l_shape_keeps_shape():
    page = PageState("页", rows=3, cols=3, vis_rows=3, vis_cols=3,
                     cells=["a", "b", None, "c", None, None, None, None, None])
    result = move_within(_state(page), 0, ["a", "b", "c"], 4,
                         anchor="a", bounds=(3, 3))
    assert result["mode"] == "rigid"
    assert page.cells[4] == "a" and page.cells[5] == "b" and page.cells[7] == "c"


def test_rigid_ring_is_4_connected():
    page = PageState(
        "页", rows=3, cols=4, vis_rows=3, vis_cols=4,
        cells=["a", "b", "c", None, "d", None, "e", None,
               "f", "g", "h", None],
    )
    result = move_within(_state(page), 0, list("abcdefgh"), 3,
                         anchor="a", bounds=(3, 4))
    assert result["mode"] == "rigid"
    assert result["clamped"] is True
    assert result["anchor_seat"] == 1
    assert {c for c in page.cells if c} == set("abcdefgh")


def test_scatter_falls_back_to_packed():
    page = PageState("页", rows=3, cols=3, vis_rows=3, vis_cols=3,
                     cells=["a", None, None, None, None, None, None, None, "b"])
    result = move_within(_state(page), 0, ["a", "b"], 4,
                         anchor="a", bounds=(3, 3))
    assert result["mode"] == "packed"
    assert page.cells[4] == "a" and page.cells[5] == "b"
    assert {c for c in page.cells if c} == {"a", "b"}


def test_diagonal_pair_falls_back_to_packed():
    page = PageState("页", rows=3, cols=3, vis_rows=3, vis_cols=3,
                     cells=["a", None, None, None, "b", None, None, None, None])
    result = move_within(_state(page), 0, ["a", "b"], 5,
                         anchor="a", bounds=(3, 3))
    assert result["mode"] == "packed"


def test_index_adjacent_but_not_4_connected_falls_back():
    page = PageState("页", rows=2, cols=2, vis_rows=2, vis_cols=2,
                     cells=[None, "a", "b", None])
    result = move_within(_state(page), 0, ["a", "b"], 0,
                         anchor="a", bounds=(2, 2))
    assert result["mode"] == "packed"


def test_hidden_source_with_visible_bounds_falls_back():
    page = PageState("页", rows=3, cols=3, vis_rows=2, vis_cols=2,
                     cells=[None, None, "a", "x", "y", None, None, None, None])
    result = move_within(_state(page), 0, ["a"], 0,
                         anchor="a", bounds=(2, 2))
    assert result["mode"] == "packed"
    assert page.cells[1] is None and page.cells[3] == "x" and page.cells[4] == "y"


def test_full_bounds_allows_rigid_across_virtual_boundary():
    page = PageState("页", rows=3, cols=3, vis_rows=2, vis_cols=2,
                     cells=[None, None, None, "a", None, None, "b", None, None])
    result = move_within(_state(page), 0, ["a", "b"], 4,
                         anchor="a", bounds=(3, 3))
    assert result["mode"] == "rigid"
    assert page.cells[4] == "a" and page.cells[7] == "b"


def test_rigid_overlap_down_puts_displaced_in_freed_source():
    page = PageState("页", rows=3, cols=2, vis_rows=3, vis_cols=2,
                     cells=["a", None, "b", None, "x", None])
    result = move_within(_state(page), 0, ["a", "b"], 4,
                         anchor="a", bounds=(3, 2))
    assert result["mode"] == "rigid"
    assert page.cells[0] == "x"      # 被顶图回到空出的源格
    assert page.cells[2] == "a" and page.cells[4] == "b"
    assert result["swapped"] == ["x"]


def test_rigid_mixed_holes_keeps_row_alignment():
    page = PageState("页", rows=3, cols=3, vis_rows=3, vis_cols=3,
                     cells=["a", "b", None, "d", "e", None, None, None, None])
    result = move_within(_state(page), 0, ["a", "b", "d", "e"], 5,
                         anchor="a", bounds=(3, 3))
    assert result["mode"] == "rigid"
    assert page.cells[4] == "a" and page.cells[5] == "b"
    assert page.cells[7] == "d" and page.cells[8] == "e"
    assert page.cells[0] is None and page.cells[3] is None


def test_rigid_displaced_multiset_no_loss_no_duplicate():
    page = PageState("页", rows=3, cols=3, vis_rows=3, vis_cols=3,
                     cells=["a", "b", "c", "x", "y", "z", None, None, None])
    result = move_within(_state(page), 0, ["a", "b", "c"], 3,
                         anchor="a", bounds=(3, 3))
    assert result["mode"] == "rigid"
    assert result["swapped"] == ["x", "y", "z"]
    assert page.cells[:3] == ["x", "y", "z"]
    assert page.cells[3:6] == ["a", "b", "c"]
    tokens = [c for c in page.cells if c]
    assert sorted(tokens) == sorted(["a", "b", "c", "x", "y", "z"])


def test_rigid_multi_displaced_fill_order():
    page = PageState("页", rows=2, cols=3, vis_rows=2, vis_cols=3,
                     cells=["a", "b", "x", "d", "e", "y"])
    result = move_within(_state(page), 0, ["a", "b", "d", "e"], 1,
                         anchor="a", bounds=(2, 3))
    assert result["mode"] == "rigid"
    assert result["swapped"] == ["x", "y"]
    assert page.cells[0] == "x" and page.cells[3] == "y"
    assert page.cells[1] == "a" and page.cells[2] == "b"
    assert page.cells[4] == "d" and page.cells[5] == "e"


def test_single_anchor_matches_legacy_packed_result():
    for target in (0, 4):
        page_new = PageState("页", rows=2, cols=3, vis_rows=2, vis_cols=2,
                             cells=["h", "i", None, "j", "k", "l"])
        page_old = PageState("页", rows=2, cols=3, vis_rows=2, vis_cols=2,
                             cells=["h", "i", None, "j", "k", "l"])
        new = move_within(_state(page_new), 0, ["h"], target,
                          anchor="h", bounds=(2, 2))
        old = move_within(_state(page_old), 0, ["h"], target)
        assert new["mode"] == "rigid"
        assert page_new.cells == page_old.cells
        assert new["swapped"] == old["swapped"]


def test_legacy_call_without_anchor_stays_packed():
    page = PageState("页", rows=3, cols=2, vis_rows=3, vis_cols=2,
                     cells=["a", None, "b", None, "c", None])
    result = move_within(_state(page), 0, ["a", "b", "c"], 1)
    assert result["mode"] == "packed"
    assert {"anchor_seat", "mode", "clamped", "moved"} <= set(result)


def test_invalid_anchor_falls_back_without_raising():
    page = PageState("页", rows=2, cols=2, vis_rows=2, vis_cols=2,
                     cells=["a", "b", None, None])
    result = move_within(_state(page), 0, ["a", "b"], 2,
                         anchor="missing", bounds=(2, 2))
    assert result["ok"] is True
    assert result["mode"] == "packed"
    assert result["anchor_seat"] is None

    page2 = PageState("页", rows=2, cols=2, vis_rows=2, vis_cols=2,
                      cells=["a", "b", None, None])
    result2 = move_within(_state(page2), 0, ["a"], 2,
                          anchor="b", bounds=(2, 2))
    assert result2["mode"] == "packed"
    assert result2["anchor_seat"] is None


def test_stride_uses_page_cols_not_visible_cols():
    page = PageState("页", rows=3, cols=3, vis_rows=2, vis_cols=2,
                     cells=["a", None, None, "b", None, None, None, None, None])
    result = move_within(_state(page), 0, ["a", "b"], 1,
                         anchor="a", bounds=(2, 2))
    assert result["mode"] == "rigid"
    assert result["anchor_seat"] == 1
    assert page.cells[1] == "a" and page.cells[4] == "b"
    assert page.cells[2] is None and page.cells[5] is None


def test_hidden_target_with_visible_bounds_falls_back():
    page = PageState("页", rows=3, cols=3, vis_rows=2, vis_cols=2,
                     cells=["a", None, None, None, None, None, None, None, None])
    result = move_within(_state(page), 0, ["a"], 2,   # seat2 在隐藏列
                         anchor="a", bounds=(2, 2))
    assert result["mode"] == "packed"
    assert page.cells[0] is None and page.cells[2] == "a"
