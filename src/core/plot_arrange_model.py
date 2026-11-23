"""整理布局的纯数据模型（不依赖 Qt，可单测）。

两套语义各管一手势（设计稿 v4）：

- 拖拽落位按**格位（行优先视觉序）**交换/安放，只动源格段与目标格段；
- 踢出/回填按**可见区优先阅读序**（先可见矩形，再隐藏区）前向补洞，
  空位恒在该序末尾。

恒等式：各页格子里的 plot 数 + 等候区数 = 常数（空位不计）。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PageState:
    """一页的排列状态。``cells`` 按行优先格位存 plot id 或 None（空位）。"""

    title: str
    rows: int
    cols: int
    vis_rows: int
    vis_cols: int
    cells: list[str | None] = field(default_factory=list)
    built: bool = True


@dataclass
class ArrangeState:
    pages: list[PageState]
    tray: list[str] = field(default_factory=list)


def seat_visible(page: PageState, index: int) -> bool:
    """第 index 个格位是否落在当前可见矩形内（隐藏区为右列+底行 L 形）。"""
    if page.cols <= 0:
        return False
    row, col = divmod(index, page.cols)
    return row < page.vis_rows and col < page.vis_cols


def order_of(page: PageState) -> list[int]:
    """可见区优先阅读序：先可见矩形行优先，再隐藏区行优先。"""
    total = max(0, page.rows * page.cols)
    indices = list(range(total))
    return sorted(
        indices,
        key=lambda i: (0 if seat_visible(page, i) else 1, i),
    )


def ordered_ids(page: PageState) -> list[str]:
    return [pid for i in order_of(page) for pid in [page.cells[i]] if pid]


def _rewrite(page: PageState, ids: list[str]) -> None:
    """把有序 plot 列表写回格位；空位恒落在可见区优先序的末尾。"""
    order = order_of(page)
    total = len(order)
    if len(page.cells) != total:
        page.cells = (list(page.cells) + [None] * total)[:total]
    for seat_index, seat in enumerate(order):
        page.cells[seat] = ids[seat_index] if seat_index < len(ids) else None


def total_plots(state: ArrangeState) -> int:
    return sum(len(ordered_ids(page)) for page in state.pages) + len(state.tray)


def clone_state(state: ArrangeState) -> ArrangeState:
    return ArrangeState(
        pages=[
            PageState(
                title=page.title,
                rows=page.rows,
                cols=page.cols,
                vis_rows=page.vis_rows,
                vis_cols=page.vis_cols,
                cells=list(page.cells),
                built=page.built,
            )
            for page in state.pages
        ],
        tray=list(state.tray),
    )


def kick_to_tray(state: ArrangeState, page_index: int, ids: list[str]) -> int:
    """把选中图踢入等候区；其余图沿可见区优先序前向补洞。"""
    page = state.pages[page_index]
    current = ordered_ids(page)
    moved = [pid for pid in current if pid in ids]
    if not moved:
        return 0
    _rewrite(page, [pid for pid in current if pid not in ids])
    state.tray.extend(moved)
    return len(moved)


def insert_from_tray(
    state: ArrangeState, page_index: int, ids: list[str], seat: int
) -> dict:
    """等候区 → 格位：落空位=安放；落有图格=顶替（原图进等候区）。"""
    page = state.pages[page_index]
    items = [pid for pid in state.tray if pid in ids]
    if not items:
        return {"ok": False, "reason": "empty", "displaced": []}
    total = page.rows * page.cols
    if len(items) > total:
        return {"ok": False, "reason": "too_many", "displaced": []}
    start = min(max(0, int(seat)), max(0, total - len(items)))
    targets = [start + offset for offset in range(len(items))]
    displaced = [page.cells[i] for i in targets if page.cells[i] is not None]
    for offset, target in enumerate(targets):
        page.cells[target] = items[offset]
    state.tray = [pid for pid in state.tray if pid not in ids]
    state.tray.extend(displaced)
    return {"ok": True, "reason": "", "displaced": displaced}


def zone_seats(page: PageState, seat: int) -> list[int]:
    """目标格所在区的阅读序（可见区矩形 / 隐藏区 L 形各自的 row-major）。

    块落位只在目标所在区内铺，避免"拖在可见区、块却填进隐藏列/行"的观感。
    """
    order = order_of(page)
    visible = [i for i in order if seat_visible(page, i)]
    hidden = [i for i in order if not seat_visible(page, i)]
    return visible if seat_visible(page, seat) else hidden


def move_within(
    state: ArrangeState, page_index: int, ids: list[str], seat: int
) -> dict:
    """页内块交换：块占目标格位段，目标段原图（除块自身）回源格位段。

    目标段按**目标所在区的阅读序**取连续格位（可见区/隐藏区互不越界）；
    块比目标区还大时退回整网格行优先连续段。源/目标重叠也不丢图不重图，
    多余源格变空位。
    """
    page = state.pages[page_index]
    total = page.rows * page.cols
    src_seats = [
        i for i, cell in enumerate(page.cells) if cell is not None and cell in ids
    ]
    if not src_seats:
        return {"ok": False, "reason": "empty", "swapped": []}
    before = list(page.cells)
    block = [before[i] for i in src_seats]
    block_size = len(block)
    zone = zone_seats(page, seat)
    if block_size <= len(zone):
        anchor = zone.index(seat) if seat in zone else 0
        anchor = min(max(0, anchor), len(zone) - block_size)
        tgt_seats = zone[anchor : anchor + block_size]
    else:
        start = min(max(0, int(seat)), max(0, total - block_size))
        tgt_seats = [start + offset for offset in range(block_size)]
    displaced = [
        before[i]
        for i in tgt_seats
        if before[i] is not None and before[i] not in ids
    ]
    next_cells = list(page.cells)
    # 先只写「不在目标段里的源格」：源/目标重叠时，落在目标段的源格必须留给
    # block，否则先写 displaced 再被 block 覆盖会把被顶替图抹掉（评审 P1）
    tgt_set = set(tgt_seats)
    free_src = [seat for seat in src_seats if seat not in tgt_set]
    for offset, src_seat in enumerate(free_src):
        next_cells[src_seat] = (
            displaced[offset] if offset < len(displaced) else None
        )
    for offset, tgt_seat in enumerate(tgt_seats):
        next_cells[tgt_seat] = block[offset]
    page.cells = next_cells
    return {"ok": True, "reason": "", "swapped": displaced}


def backfill(state: ArrangeState, page_index: int) -> int:
    """等候区头部图依次填入当前页序末尾空位。"""
    page = state.pages[page_index]
    ids = ordered_ids(page)
    placed = 0
    total = page.rows * page.cols
    while state.tray and len(ids) < total:
        ids.append(state.tray.pop(0))
        placed += 1
    _rewrite(page, ids)
    return placed


def can_commit(state: ArrangeState) -> bool:
    return len(state.tray) == 0