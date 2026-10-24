"""绘图区标签页（PlotPage）：每页一份网格与视图状态。

状态分层见设计文档 §3：``plot_widgets / plot_layout / rows_* / cols_* /
row_height_factors`` 属于页；数据、时间基准、游标、标记区间、密度基线仍归
MainWindow，全局共享。

本类在 P0-1 阶段先只承载状态与纯查询（不改既有行为），网格重建的方法在
P0-2 从 LayoutManager 搬迁进来。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import QGridLayout, QWidget

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查
    from src.ui.widgets.plot_container import PlotContainerWidget


class PlotPage:
    """一个标签页：QWidget 容器 + QGridLayout + 页级视图状态。

    Args:
        parent: 页容器的父控件（首版为 MainWindow.plot_widget）。
        title: 标签标题，默认 ``图 N``。
        rows_max/cols_max: 本页网格容量上限。
        rows_current/cols_current: 当前可见行列。
        row_height_factors: 行高百分比（100 = 均分）。
        plot_widgets: 初始容器列表（构造期兜底用，通常为空）。
    """

    def __init__(
        self,
        *,
        parent: QWidget,
        title: str,
        rows_max: int,
        cols_max: int,
        rows_current: int,
        cols_current: int,
        row_height_factors: dict[int, int] | None = None,
        plot_widgets: list | None = None,
    ):
        self.title = title
        #: 每次切页/关页自增，迟到回调据此作废（设计文档 §5.2 / R4）
        self.epoch = 0

        self.widget = QWidget(parent)
        self.plot_layout = QGridLayout(self.widget)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_layout.setSpacing(0)

        self.plot_widgets: list[PlotContainerWidget] = list(plot_widgets or [])
        self.rows_max = int(rows_max)
        self.cols_max = int(cols_max)
        self.rows_current = int(rows_current)
        self.cols_current = int(cols_current)
        self.row_height_factors: dict[int, int] = dict(row_height_factors or {})

        #: 隐藏期间有 cursor / mark item 待补建，切回本页时处理（§5.1）
        self._scene_dirty = False
        #: 页面级 XLink 防抖标志（§3）
        self._pending_xlink_sync = False
        #: 非活动页几何变化后待补对齐（§6.2）
        self._xlink_dirty = False
        #: 切页前的 master X 范围，用于“切页跟随”（§5.2）
        self.saved_x_range: tuple[float, float] | None = None

    # ------------------------------------------------------------------
    # 纯查询：跨页路径必须用显式行列，不能用 isVisible（全局规则 R0）
    # ------------------------------------------------------------------
    def cell_visible(self, index: int) -> bool:
        """第 index 个网格是否落在当前可见行列内。"""
        if index < 0 or index >= len(self.plot_widgets):
            return False
        if self.cols_max <= 0:
            return False
        row, col = divmod(index, self.cols_max)
        return row < self.rows_current and col < self.cols_current

    def visible_containers(self) -> list[PlotContainerWidget]:
        return [
            container
            for index, container in enumerate(self.plot_widgets)
            if self.cell_visible(index)
        ]

    def container_at(self, row: int, col: int) -> PlotContainerWidget | None:
        if row < 0 or col < 0 or self.cols_max <= 0:
            return None
        index = row * self.cols_max + col
        if index >= len(self.plot_widgets):
            return None
        return self.plot_widgets[index]

    def row_of(self, plot_widget) -> int:
        """给定子图，返回它在本页的行号；找不到返回 -1。"""
        for index, container in enumerate(self.plot_widgets):
            if container is plot_widget or container.plot_widget is plot_widget:
                return index // self.cols_max if self.cols_max > 0 else -1
        return -1

    def master_container(self) -> PlotContainerWidget | None:
        """主视图容器：首个显式可见容器，全部隐藏时回退首个。"""
        for index, container in enumerate(self.plot_widgets):
            if self.cell_visible(index):
                return container
        return self.plot_widgets[0] if self.plot_widgets else None

    def master_viewbox(self):
        container = self.master_container()
        if container is None:
            return None
        return getattr(getattr(container, "plot_widget", None), "view_box", None)

    # ------------------------------------------------------------------
    # 状态标记 / 清理
    # ------------------------------------------------------------------
    def mark_dirty_for_activation(self) -> None:
        """隐藏页只记状态，切回本页时补建场景（§5.1）。"""
        self._scene_dirty = True

    def cancel_pending(self) -> None:
        """作废本页已排队的防抖动作（定时器由持有方的 epoch 兜底）。"""
        self._pending_xlink_sync = False
        self._xlink_dirty = False

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def dispose(self) -> None:
        """断开页容器与 Qt 父控件的关系（网格清理由 P0-2 的 clear_matrix 负责）。"""
        self.epoch += 1
        self.cancel_pending()
        self.plot_widgets.clear()
        self.widget.hide()
        self.widget.setParent(None)
        self.widget.deleteLater()

    def __repr__(self) -> str:  # pragma: no cover - 仅日志
        return (
            f"<PlotPage title={self.title!r} epoch={self.epoch} "
            f"grid={self.rows_current}x{self.cols_current}/{self.rows_max}x{self.cols_max} "
            f"plots={len(self.plot_widgets)}>"
        )
