"""把整理布局对话框的排列置换应用到真实 PlotPage。

提交才动真实页面（设计稿 §4）：曲线/样式/manager 链随容器整体搬运，
不销毁重建；空位复用或新建空容器（空容器可互换，D7）。

关键动作：
1. 置换各页 ``plot_widgets`` 列表并重新 ``addWidget``；
2. 跨页容器重注入 ``page`` / ``plot_context``；
3. 全部容器先 unlink，再按各页新 master 重连（XLink 不跨页）；
4. 各页恢复提交前的 X 范围（防 relink 静默回收）；
5. 补刷 mark region / 曲线样式 / 统计。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.app.plot_context import PlotContext
from src.core.config import widget_alive
from src.core.logger import get_logger
from src.core.plot_arrange_model import ArrangeState

logger = get_logger("ui.plot_arrange")


@dataclass
class ArrangeSnapshot:
    """对话框打开时的页面快照：token = 有内容的容器。"""

    pages: list
    dims: list[tuple[int, int]]  # 每页 (rows, cols)，取自真实 QGridLayout 坐标
    built: list[bool]
    tokens: dict[str, object]  # token id -> PlotContainerWidget
    names: dict[str, list[str]]  # token id -> 曲线名（快照指纹）
    row_pcts: list[list[int]]  # 每页各行行高百分比（100 = 均分）
    seat_of_token: dict[str, tuple[int, int]]  # token id -> 真实 (row, col)
    row_heights: list[list[float]]  # 每页各行相对高度（镜像真实布局，隐藏行取中位数）

    def container(self, token: str):
        return self.tokens[token]


def _container_names(container) -> list[str]:
    """容器上显示的曲线名列表（含失效占位）；空图返回空列表。"""
    plot_widget = getattr(container, "plot_widget", None)
    if plot_widget is None:
        return []
    try:
        names = list(getattr(plot_widget, "curves", {}) or {})
        if names:
            return names
        invalid = getattr(plot_widget, "invalid_curves", {}) or {}
        return list(invalid.keys())
    except RuntimeError:
        return []


def page_row_pcts(page, rows: int) -> list[int]:
    """本页各行行高百分比（缺省 100）；行高跟行走、不随图迁移。"""
    factors = getattr(page, "row_height_factors", {}) or {}
    return [int(factors.get(row, 100)) for row in range(rows)]


def _fill_sizes(sizes: list[float], weights: list[float]) -> list[float]:
    """格位尺寸兜底：未布局（全 0）时用权重；隐藏格（0）取可见中位数。

    隐藏格与可见格同尺寸——对话框以操作便利为先，不做"缩水"布局。
    """
    known = [value for value in sizes if value > 0]
    if not known:
        return [float(weight) if weight > 0 else 1.0 for weight in weights]
    median = sorted(known)[len(known) // 2]
    return [value if value > 0 else median for value in sizes]


def snapshot_pages(pages: list) -> ArrangeSnapshot:
    """为对话框拍快照。

    格位以 ``QGridLayout`` 的**真实坐标**为准（``getItemPosition``），不假设
    ``plot_widgets`` 列表下标等于行优先格位；只有带曲线（或失效占位）的
    容器是 token。空位、隐藏行列与行高都原样带进对话框。
    """
    tokens: dict[str, object] = {}
    names: dict[str, list[str]] = {}
    dims: list[tuple[int, int]] = []
    built: list[bool] = []
    row_pcts: list[list[int]] = []
    seat_of_token: dict[str, tuple[int, int]] = {}
    row_heights: list[list[float]] = []
    for page_index, page in enumerate(pages):
        layout = getattr(page, "plot_layout", None)
        widgets = list(getattr(page, "plot_widgets", []) or [])
        if layout is None or not widgets:
            rows = max(1, int(getattr(page, "rows_current", 1)))
            cols = max(1, int(getattr(page, "cols_current", 1)))
            dims.append((rows, cols))
            built.append(False)
            pcts = page_row_pcts(page, rows)
            row_pcts.append(pcts)
            row_heights.append(_fill_sizes([0.0] * rows, [float(p) for p in pcts]))
            continue
        # 真实格位：遍历布局项取 (row, col)，与列表顺序解耦
        positions: list[tuple[int, int, object]] = []
        seen_widgets: set[int] = set()
        max_row = 0
        max_col = 0
        for index in range(layout.count()):
            item = layout.itemAt(index)
            widget = item.widget() if item is not None else None
            if widget is None:
                continue
            row, col, _row_span, _col_span = layout.getItemPosition(index)
            positions.append((row, col, widget))
            seen_widgets.add(id(widget))
            max_row = max(max_row, row)
            max_col = max(max_col, col)
        # 未进布局的容器（理论不应存在）退回列表下标，保证不丢图
        for seat, container in enumerate(widgets):
            if id(container) in seen_widgets:
                continue
            row, col = divmod(seat, max_col + 1)
            positions.append((row, col, container))
            max_row = max(max_row, row)
        rows = max(1, max_row + 1)
        cols = max(1, max_col + 1)
        dims.append((rows, cols))
        built.append(True)
        pcts = page_row_pcts(page, rows)
        row_pcts.append(pcts)
        # 行高按可见格的真实 rect 取（隐藏格 geometry 可能是陈旧值，不可信），
        # 隐藏行取可见中位数兜底；列宽在对话框里统一等分
        vis_rows = min(rows, max(1, int(getattr(page, "rows_current", rows))))
        vis_cols = min(cols, max(1, int(getattr(page, "cols_current", cols))))
        heights = [0.0] * rows
        for row, col, container in positions:
            if row >= vis_rows or col >= vis_cols:
                continue
            try:
                rect = container.geometry()
            except RuntimeError:
                continue
            if rect.height() > 0:
                heights[row] = max(heights[row], float(rect.height()))
        row_heights.append(_fill_sizes(heights, [float(p) for p in pcts]))
        for row, col, container in positions:
            curve_names = _container_names(container)
            if not curve_names:
                continue
            token = f"p{page_index}:{row * cols + col}"
            tokens[token] = container
            names[token] = curve_names
            seat_of_token[token] = (row, col)
    return ArrangeSnapshot(
        pages, dims, built, tokens, names, row_pcts, seat_of_token,
        row_heights,
    )


def build_state(snapshot: ArrangeSnapshot) -> ArrangeState:
    """由快照构造对话框模型状态（cells 按行优先格位，取真实坐标）。"""
    from src.core.plot_arrange_model import PageState

    pages: list[PageState] = []
    for page_index, page in enumerate(snapshot.pages):
        rows, cols = snapshot.dims[page_index]
        cells: list[str | None] = [None] * (rows * cols)
        prefix = f"p{page_index}:"
        for token, (row, col) in snapshot.seat_of_token.items():
            if token.startswith(prefix):
                cells[row * cols + col] = token
        pages.append(
            PageState(
                title=page.title,
                rows=rows,
                cols=cols,
                vis_rows=min(rows, max(1, int(page.rows_current))),
                vis_cols=min(cols, max(1, int(page.cols_current))),
                cells=cells,
                built=snapshot.built[page_index],
            )
        )
    return ArrangeState(pages, [])


def validate_state(snapshot: ArrangeSnapshot, state: ArrangeState) -> bool:
    """提交前校验：快照失效（曲线变化）/ 维度漂移 / token 重复或丢失都拒绝。"""
    if len(state.pages) != len(snapshot.pages):
        return False
    seen: set[str] = set()
    for page_index, page_state in enumerate(state.pages):
        rows, cols = snapshot.dims[page_index]
        if (page_state.rows, page_state.cols) != (rows, cols):
            return False
        if len(page_state.cells) != rows * cols:
            return False
        for token in page_state.cells:
            if token is None:
                continue
            if token not in snapshot.tokens or token in seen:
                return False
            seen.add(token)
    for token in state.tray:
        if token not in snapshot.tokens or token in seen:
            return False
        seen.add(token)
    if seen != set(snapshot.tokens):
        return False
    # 曲线指纹：对话框打开期间数据被重载/清图则失效
    for token, container in snapshot.tokens.items():
        if not widget_alive(container):
            return False
        if _container_names(container) != snapshot.names[token]:
            return False
    return True


def _safe_view_range(page) -> tuple[float, float] | None:
    try:
        view_box = page.master_viewbox()
    except Exception:
        return None
    if view_box is None:
        return None
    try:
        x_min, x_max = view_box.viewRange()[0]
    except Exception:
        return None
    if math.isfinite(x_min) and math.isfinite(x_max) and x_max > x_min:
        return float(x_min), float(x_max)
    return None


def _destroy_container(container) -> None:
    plot_widget = getattr(container, "plot_widget", None)
    if plot_widget is not None:
        try:
            plot_widget.mark_being_destroyed()
        except RuntimeError:
            pass
    try:
        container.setParent(None)
        container.deleteLater()
    except RuntimeError:
        pass


def _initialize_built_page(page, rows: int, cols: int) -> None:
    """未建页在提交时补初始化结构字段（等价 build_matrix 的非容器部分）。"""
    page._built_rows = int(rows)
    page._built_cols = int(cols)
    page._apply_grid_stretches(rows, cols)
    for row in range(rows):
        page.row_height_factors.setdefault(row, 100)


def _prepare_empty_pool(page, page_index: int, snapshot, state, created: list) -> list:
    """预备阶段：为空位凑齐空容器；只在内存里造，不碰真实布局。"""
    if snapshot.built[page_index]:
        pool = [c for c in page.plot_widgets if not _container_names(c)]
    else:
        pool = []
    needed = sum(1 for cell in state.pages[page_index].cells if cell is None)
    services = getattr(page, "services", None)
    while len(pool) < needed:
        if services is None:
            raise ValueError("整理布局：页面缺少 PageServices，无法新建空容器")
        container, _master = page._new_plot_container(
            0, 0, services=services, master_viewbox=None
        )
        created.append(container)
        pool.append(container)
    return pool


def apply_arrangement(snapshot: ArrangeSnapshot, state: ArrangeState) -> dict:
    """按置换表把排列应用到真实页面；返回变更摘要。

    两阶段落地：**预备阶段先建好全部新空容器**（失败则真实页面零改动），
    落地阶段只做摘布局/搬运/重挂/重连，不再分配对象。
    """
    if state.tray:
        # apply 假定等候区已清空；程序化调用若带残留会把图变成孤儿
        raise ValueError("整理布局：等候区还有图未安置")
    if not validate_state(snapshot, state):
        raise ValueError("整理布局快照已失效（曲线或页面发生变化）")

    pages = snapshot.pages
    saved_ranges = {page: _safe_view_range(page) for page in pages}

    # 1) 预备：所有分配动作先做；任何失败都不动真实页面
    created: list = []
    try:
        pools = [
            _prepare_empty_pool(page, index, snapshot, state, created)
            for index, page in enumerate(pages)
        ]
    except Exception:
        for container in created:
            _destroy_container(container)
        raise

    # 2) 全部容器先摘出布局并断开 XLink（跨页搬运前必须无跨页 link）
    all_containers: list = []
    for page in pages:
        for container in list(page.plot_widgets):
            page.plot_layout.removeWidget(container)
            all_containers.append(container)
        page.plot_widgets = []
    for container in all_containers + created:
        view_box = getattr(getattr(container, "plot_widget", None), "view_box", None)
        if view_box is None:
            continue
        try:
            view_box.setXLink(None)
        except Exception:
            logger.debug("[ARRANGE] unlink 失败", exc_info=True)

    # 3) 逐页重建：plan 中的 token 就位，空位取预备好的空容器
    leftovers: list = []
    moved_ids = 0
    cross_page = 0
    hidden_placed = 0
    for page_index, page in enumerate(pages):
        rows, cols = snapshot.dims[page_index]
        if not snapshot.built[page_index]:
            _initialize_built_page(page, rows, cols)
        services = getattr(page, "services", None)
        pool = pools[page_index]
        new_list: list = []
        for seat, token in enumerate(state.pages[page_index].cells):
            row, col = divmod(seat, cols)
            if token is not None:
                container = snapshot.tokens[token]
                plot_widget = container.plot_widget
                if getattr(plot_widget, "page", None) is not page:
                    cross_page += 1
                    plot_widget.page = page
                    if services is not None:
                        plot_widget.plot_context = PlotContext(services)
                moved_ids += 1
            else:
                container = pool.pop()
            new_list.append(container)
            page.plot_layout.addWidget(container, row, col)
            # 可见性必须按页面的可见矩形重套：从可见格搬进隐藏格（或跨页
            # 搬入隐藏格）的容器若仍 visible，QGridLayout 会按最小尺寸把
            # 隐藏行列撑开——表现为"拖完多出一行/一列"
            visible = (
                row < int(page.rows_current) and col < int(page.cols_current)
            )
            try:
                container.setVisible(visible)
            except RuntimeError:
                pass
            if token is not None and not visible:
                hidden_placed += 1
        leftovers.extend(pool)
        page.plot_widgets = new_list

    # 4) 删除多余空容器（曲线容器不会进入 leftovers）
    for container in leftovers:
        _destroy_container(container)

    # 5) 重建各页 XLink（新 master），再恢复提交前 X 范围，最后补刷
    for page in pages:
        try:
            page.sync_xlink_after_visibility_change(
                page.rows_current, page.cols_current
            )
        except Exception:
            logger.debug("[ARRANGE] 重连 XLink 失败", exc_info=True)
        saved = saved_ranges.get(page)
        if saved is not None:
            try:
                page.apply_x_range(*saved)
            except Exception:
                logger.debug("[ARRANGE] 恢复 X 范围失败", exc_info=True)
        try:
            page.refresh_curve_styles()
        except Exception:
            logger.debug("[ARRANGE] 补刷曲线样式失败", exc_info=True)
        try:
            page.restore_mark_regions()
        except Exception:
            logger.debug("[ARRANGE] 重建 mark region 失败", exc_info=True)
        services = getattr(page, "services", None)
        layout_manager = getattr(services, "layout_manager", None)
        if layout_manager is not None:
            try:
                layout_manager.request_mark_stats_refresh(immediate=True)
            except Exception:
                logger.debug("[ARRANGE] 统计刷新失败", exc_info=True)

    # 最小 X 范围依赖可见性/链接，跨页搬入可见格后需立即重算（与
    # apply_visibility → adjust_stretch_and_range 同口径）
    for page in pages:
        cursor_sync = getattr(
            getattr(page, "services", None), "cursor_sync_manager", None
        )
        if cursor_sync is None:
            continue
        try:
            cursor_sync._sync_min_xrange()
        except Exception:
            logger.debug("[ARRANGE] 最小 X 范围重算失败", exc_info=True)
        break

    return {
        "pages": len(pages),
        "plots": moved_ids,
        "cross_page": cross_page,
        "hidden": hidden_placed,
    }
