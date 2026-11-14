"""拖拽缩略图与热点的几何约定（四个拖拽入口共用的视觉层）。

这里只锁「结构不变量」：空输入、逻辑尺寸与 DPR、热点落在可见表面左上角、
头部提示与条数同一行、超行折叠。像素级观感交给真机与截图验收。
"""

from src.ui.drag_visuals import (
    HEADER_H,
    HEADER_SEP_H,
    MAX_VISIBLE_ROWS,
    PAD_Y,
    PIXMAP_PAD,
    ROW_H,
    DragItem,
    build_drag_pixmap,
    drag_hotspot,
)


def _logical_size(pixmap):
    dpr = pixmap.devicePixelRatio() or 1.0
    return round(pixmap.width() / dpr), round(pixmap.height() / dpr)


def test_empty_items_returns_none(qapp):
    assert build_drag_pixmap([], None) is None


def test_size_grows_with_rows_and_hint_stays_in_header(qapp):
    one = build_drag_pixmap([DragItem("a", "#1f77b4")], None)
    two = build_drag_pixmap([DragItem("a"), DragItem("b")], None)
    hint = build_drag_pixmap([DragItem("a")], None, hint="Shift = 替换")

    assert _logical_size(one)[1] + ROW_H == _logical_size(two)[1]
    # 提示与条数同行：不新增一整行，高度与无提示时一致
    assert _logical_size(hint)[1] == _logical_size(one)[1]


def test_hotspot_sits_on_visible_surface_corner(qapp):
    pixmap = build_drag_pixmap([DragItem("a")], None)

    hotspot = drag_hotspot()

    # 热点是 pixmap 内的逻辑坐标：扣掉四周阴影留白后正好是可见表面左上角
    assert hotspot.x() == PIXMAP_PAD
    assert hotspot.y() == PIXMAP_PAD
    assert hotspot.x() < _logical_size(pixmap)[0]
    assert hotspot.y() < _logical_size(pixmap)[1]


def test_overflow_folds_into_one_extra_row(qapp):
    rows = [DragItem(f"v{i}") for i in range(MAX_VISIBLE_ROWS + 2)]
    full = build_drag_pixmap(rows[:MAX_VISIBLE_ROWS], None)
    folded = build_drag_pixmap(rows, None)

    assert _logical_size(folded)[1] == _logical_size(full)[1] + ROW_H


def test_header_height_is_shared_by_hint_and_count(qapp):
    """头部只占 HEADER_H：条数/提示/分隔线都在这一节内。"""
    pixmap = build_drag_pixmap(
        [DragItem("demo_signal", "#1f77b4")], None, hint="拖到其他子图释放"
    )
    expected = (
        PIXMAP_PAD * 2
        + PAD_Y * 2
        + HEADER_H
        + HEADER_SEP_H
        + ROW_H
    )
    assert _logical_size(pixmap)[1] == expected
