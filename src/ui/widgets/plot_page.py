"""绘图区标签页（PlotPage）：每页一份网格与视图状态。

状态分层见设计文档 §3：``plot_widgets / plot_layout / rows_* / cols_* /
row_height_factors`` 属于页；数据、时间基准、游标、标记区间、密度基线仍归
MainWindow，全局共享。

本类在 P0-1 阶段先只承载状态与纯查询（不改既有行为），网格重建的方法在
P0-2 从 LayoutManager 搬迁进来。
"""

from __future__ import annotations

import math
from contextlib import ExitStack
from typing import TYPE_CHECKING

from PySide6.QtCore import QSignalBlocker, QTimer
from PySide6.QtWidgets import QGridLayout, QWidget

from src.core.config import safe_qt_op, widget_alive
from src.core.logger import get_logger
from src.app.plot_context import PlotContext
from src.ui.widgets.plot_container import PlotContainerWidget

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查
    pass

logger = get_logger(__name__)


class PlotPage:
    """一个标签页：QWidget 容器 + QGridLayout + 页级视图状态。

    Args:
        parent: 页容器的父控件（首版为 MainWindow.plot_widget）。
        title: 标签标题，默认 ``page N``。
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
        #: 页级服务（PageServices），由 MainWindow 建页后注入
        self.services = None
        #: 页容器（QWidget）上是否发生过布局变更的调试记录
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

        #: 最近一次实际构建的网格规模。容器扁平列表的列步长以它为准：
        #: 新建页只建可见区（§5.3），若仍按 cols_max 算 divmod 会把
        #: (r,c) 映射到错误单元格。
        self._built_rows = 0
        self._built_cols = 0

        #: 隐藏期间有 cursor / mark item 待补建，切回本页时处理（§5.1）
        self._scene_dirty = False
        #: 最近一次游标场景恢复对应的全局状态指纹；相同则激活时可跳过
        self._cursor_state_sig: tuple | None = None
        #: 页面级 XLink 防抖标志（§3）
        self._pending_xlink_sync = False
        #: 非活动页几何变化后待补对齐（§6.2）
        self._xlink_dirty = False
        #: 切页前的 master X 范围，用于“切页跟随”（§5.2）
        self.saved_x_range: tuple[float, float] | None = None

    @property
    def grid_cols(self) -> int:
        """容器扁平列表的列步长。

        新页只建 ``rows_current × cols_current``（§5.3），此时实际列数可能
        小于容量 ``cols_max``；凡按扁平索引反算 (row, col) 的地方都必须用
        本属性，不能再用 ``cols_max``。
        """
        if self._built_cols > 0:
            return self._built_cols
        return max(1, self.cols_max)

    # ------------------------------------------------------------------
    # 网格构建 / 拆除（从 LayoutManager 搬迁，顺序与原实现一致）
    # ------------------------------------------------------------------
    def mark_all_plots_being_destroyed(self) -> None:
        """deleteLater 到真正析构之间仍可能有定时器回调，先显式置位。"""
        for container in self.plot_widgets:
            plot_widget = getattr(container, "plot_widget", None)
            if plot_widget is not None:
                try:
                    plot_widget.mark_being_destroyed()
                except RuntimeError:
                    pass

    def clear_matrix(self) -> None:
        """摘掉全部子图容器（页容器本身保留）。"""
        self.mark_all_plots_being_destroyed()
        for index in reversed(range(self.plot_layout.count())):
            widget = self.plot_layout.itemAt(index).widget()
            if widget:
                widget.setParent(None)
                widget.deleteLater()
        self.plot_widgets.clear()
        self._built_rows = 0
        self._built_cols = 0

    def _new_plot_container(self, row: int, col: int, *, services, master_viewbox):
        """建一个子图容器；非首格自动 XLink 到 master。"""
        from src.ui.widgets.plot_widget import DraggableGraphicsLayoutWidget

        plot_widget = DraggableGraphicsLayoutWidget(
            services.units, services.data, services.time_channels_infos
        )
        #: 反查所属页（mark region / 统计 / 日志头）
        plot_widget.page = self
        # 关键：注入“本页”而不是“当前页”，页切走后 ctx 仍指向自己的兄弟集合
        plot_widget.plot_context = PlotContext(services)

        cursor_enabled = services.cursor_btn.isChecked()
        if cursor_enabled and services.cursor_values_hidden:
            plot_widget.toggle_cursor(False, hide_values_only=True)
        else:
            plot_widget.toggle_cursor(cursor_enabled)
        if cursor_enabled:
            plot_widget.apply_cursor_mode(
                services.cursor_mode, services.pinned_x_values
            )

        if master_viewbox is None:
            master_viewbox = plot_widget.view_box
        else:
            plot_widget.view_box.setXLink(master_viewbox)

        container = PlotContainerWidget(plot_widget)
        container.plot_widget = plot_widget
        return container, master_viewbox

    def _apply_grid_stretches(self, rows: int, cols: int) -> None:
        for row in range(rows):
            percentage = self.row_height_factors.get(row, 100)
            self.plot_layout.setRowStretch(row, max(1, percentage // 25))
        for col in range(cols):
            self.plot_layout.setColumnStretch(col, 1)

    def build_matrix(self, rows: int, cols: int, *, deps=None) -> None:
        """建立 rows×cols 子图矩阵并注入页作用域 PlotContext。

        Args:
            rows/cols: 本次要建的网格规模（既有调用方传页容量）。
            deps: PageServices；不传则用建页时注入的 ``self.services``。
        """
        services = deps if deps is not None else self.services
        if services is None:
            raise RuntimeError("PlotPage.build_matrix: PageServices 未注入")

        self.mark_all_plots_being_destroyed()
        for index in reversed(range(self.plot_layout.count())):
            widget = self.plot_layout.itemAt(index).widget()
            if widget:
                widget.setParent(None)
                widget.deleteLater()
        self.plot_widgets.clear()

        master_viewbox = None
        for row in range(rows):
            for col in range(cols):
                container, master_viewbox = self._new_plot_container(
                    row, col, services=services, master_viewbox=master_viewbox
                )
                self.plot_layout.addWidget(container, row, col)
                self.plot_widgets.append(container)

        self._built_rows = int(rows)
        self._built_cols = int(cols)
        self._apply_grid_stretches(rows, cols)
        if services.mark_region_btn.isChecked():
            # 给新建格子补 region 必须走 restore_mark_regions（按全局区间
            # 重建、不写回全局值）：toggle_mark_region(True) 是完整开启路径，
            # 会以本页空图默认 viewRange [0,1] 为参照把全局 saved_mark_range
            # 重置成 (1/3, 2/3)，污染所有页的统计区间（新建页拖入曲线后
            # 统计 count=0/1 的病征）。隐藏页在 restore 内部自动记脏延迟补建。
            self.restore_mark_regions()

        for row in range(rows):
            self.row_height_factors.setdefault(row, 100)

    def ensure_matrix(self, rows: int | None = None, cols: int | None = None) -> bool:
        """确保本页至少有 rows×cols 的容器；返回是否发生了创建。

        新建页激活时按“继承当前页可见行列”首次建矩阵（D-C）；用户随后在
        布局对话框里扩大行列时，只补建缺失单元格，已有子图与曲线原样保留
        （§5.3：不能为了补格把整页重建）。
        """
        target_rows = int(rows if rows is not None else self.rows_current)
        target_cols = int(cols if cols is not None else self.cols_current)
        target_rows = min(max(1, target_rows), max(1, self.rows_max))
        target_cols = min(max(1, target_cols), max(1, self.cols_max))

        if not self.plot_widgets:
            self.build_matrix(target_rows, target_cols)
            return True
        if target_rows <= self._built_rows and target_cols <= self._built_cols:
            return False
        self._grow_matrix(
            max(target_rows, self._built_rows), max(target_cols, self._built_cols)
        )
        return True

    def _grow_matrix(self, new_rows: int, new_cols: int) -> None:
        """补建缺失单元格；按 (r,c) 而非旧扁平索引搬运已有容器。"""
        services = self.services
        if services is None:
            raise RuntimeError("PlotPage._grow_matrix: PageServices 未注入")

        old_cols = self.grid_cols
        existing = {}
        for index, container in enumerate(self.plot_widgets):
            row, col = divmod(index, old_cols)
            existing[(row, col)] = container

        master = existing.get((0, 0))
        master_viewbox = None
        if master is not None:
            master_viewbox = getattr(
                getattr(master, "plot_widget", None), "view_box", None
            )

        self.plot_widgets.clear()
        for row in range(new_rows):
            for col in range(new_cols):
                container = existing.get((row, col))
                if container is None:
                    container, master_viewbox = self._new_plot_container(
                        row, col,
                        services=services,
                        master_viewbox=master_viewbox,
                    )
                    self.plot_layout.addWidget(container, row, col)
                self.plot_widgets.append(container)

        self._built_rows = int(new_rows)
        self._built_cols = int(new_cols)
        self._apply_grid_stretches(new_rows, new_cols)
        for row in range(new_rows):
            self.row_height_factors.setdefault(row, 100)

        if services.mark_region_btn.isChecked():
            self.restore_mark_regions()

    # ------------------------------------------------------------------
    # 可见性 / 行高 / XLink
    # ------------------------------------------------------------------
    def apply_visibility(self, rows: int, cols: int) -> None:
        """按显式行列设置可见区（不能用 isVisible 反推，R0）。

        行列超过已建网格时先补建缺失格（新页首次调布局对话框的入口），
        已有子图与曲线不会被重建。
        """
        rows = min(max(1, int(rows)), max(1, self.rows_max))
        cols = min(max(1, int(cols)), max(1, self.cols_max))
        self.ensure_matrix(rows, cols)
        self.rows_current = int(rows)
        self.cols_current = int(cols)
        grid_cols = self.grid_cols
        for index, container in enumerate(self.plot_widgets):
            row, col = divmod(index, grid_cols)
            container.setVisible(row < rows and col < cols)
        self.sync_xlink_after_visibility_change(rows, cols)
        self.adjust_stretch_and_range(rows, cols)

    def sync_xlink_after_visibility_change(self, rows: int, cols: int) -> None:
        """可见性变更后建立/断开本页 link（不跨页建 link）。"""
        master_container = self.master_container()
        master_vb = self.master_viewbox()
        if master_container is None or master_vb is None:
            return
        grid_cols = self.grid_cols
        for index, container in enumerate(self.plot_widgets):
            if container is master_container or not widget_alive(container):
                continue
            row, col = divmod(index, grid_cols)
            should_be_visible = row < rows and col < cols
            view_box = getattr(getattr(container, "plot_widget", None), "view_box", None)
            if view_box is None:
                continue
            currently_linked = view_box.linkedView(0) is not None
            if should_be_visible and not currently_linked:
                logger.debug(
                    "[XLINK] set_plots_visible: establishing link for plot idx=%d r=%d c=%d",
                    index, row, col,
                )
                view_box.setXLink(master_vb)
            elif not should_be_visible and currently_linked:
                logger.debug(
                    "[XLINK] set_plots_visible: breaking link for hidden plot idx=%d r=%d c=%d",
                    index, row, col,
                )
                view_box.setXLink(None)

    def adjust_stretch_and_range(self, rows: int, cols: int) -> None:
        """调整行列 stretch，并把可见子图对齐到首个可见 master 的 X 范围。"""
        services = self.services
        grid_cols = self.grid_cols
        # 按实际可见容器反推列号：已建列数可能小于容量 cols_max，
        # 不能用容量列号直接拼扁平索引（会回绕到真实列，形成幽灵列）。
        visible_cols = {
            index % grid_cols
            for index in range(len(self.plot_widgets))
            if self.cell_visible(index)
        }
        for col in range(self.cols_max):
            stretch = 1 if col in visible_cols else 0
            self.plot_layout.setColumnStretch(col, stretch)
            logger.debug(
                "[LAYOUT] setColumnStretch col=%d stretch=%s has_visible=%s",
                col, stretch, col in visible_cols,
            )

        for row in range(self.rows_max):
            if row < rows:
                percentage = self.row_height_factors.get(row, 100)
                self.plot_layout.setRowStretch(row, max(1, percentage // 25))
            else:
                self.plot_layout.setRowStretch(row, 0)

        self.rows_current = int(rows)
        self.cols_current = int(cols)
        self.restore_mark_regions()

        if not self.plot_widgets:
            services.cursor_sync_manager._sync_min_xrange()
            return

        visible_indices = [
            index for index in range(len(self.plot_widgets)) if self.cell_visible(index)
        ]
        if not visible_indices:
            logger.debug("[LAYOUT] no visible plots, skipping X-range sync")
            services.cursor_sync_manager._sync_min_xrange()
            return

        first_visible = self.plot_widgets[visible_indices[0]]
        global_min, global_max = None, None
        try:
            global_min, global_max = first_visible.plot_widget.view_box.viewRange()[0]
            logger.debug(
                "[LAYOUT] X-range sync source: first_visible viewRange=(%.4f, %.4f)",
                global_min, global_max,
            )
        except Exception:
            logger.debug("[LAYOUT] failed to read first visible plot viewRange, falling back")

        if global_min is None or global_max is None:
            global_min, global_max = services.cursor_sync_manager.collect_global_x_range()
            if global_min is not None:
                logger.debug(
                    "[LAYOUT] X-range sync fallback: collect_global_x_range=(%.4f, %.4f)",
                    global_min, global_max,
                )

        if global_min is None or global_max is None:
            services.cursor_sync_manager._sync_min_xrange()
            return

        synced_count = 0
        sync_error = None
        for index in visible_indices:
            self.plot_widgets[index].plot_widget._is_syncing_range = True
        try:
            for index in visible_indices:
                container = self.plot_widgets[index]
                widget = container.plot_widget
                view_box = widget.view_box
                linked = view_box.linkedView(0)
                geom = container.geometry()

                before_min, before_max = None, None
                try:
                    before_min, before_max = view_box.viewRange()[0]
                except Exception:
                    logger.debug("获取 viewRange 失败", exc_info=True)

                if linked is not None:
                    view_box.setXLink(None)
                    logger.debug(
                        "[XLINK] plot=%s temp unlink for sync_all (%.4f, %.4f)",
                        getattr(widget, "y_name", f"idx={index}"),
                        global_min, global_max,
                    )
                try:
                    view_box.enableAutoRange(x=False)
                    view_box.setXRange(global_min, global_max, padding=0)
                finally:
                    if linked is not None:
                        view_box.setXLink(linked)
                        logger.debug(
                            "[XLINK] plot=%s link restored after sync_all",
                            getattr(widget, "y_name", f"idx={index}"),
                        )

                after_min, after_max = None, None
                try:
                    after_min, after_max = view_box.viewRange()[0]
                except Exception:
                    logger.debug("获取 viewRange 失败", exc_info=True)

                row, col = divmod(index, grid_cols)
                match_ok = (
                    after_min is not None
                    and after_max is not None
                    and abs(after_min - global_min) < 1e-6
                    and abs(after_max - global_max) < 1e-6
                )
                logger.debug(
                    "[LAYOUT]   plot[%d,%d] idx=%d linked=%s before=(%.4f,%.4f) "
                    "after=(%.4f,%.4f) geom=(%d,%d %dx%d) match=%s",
                    row, col, index, linked is not None,
                    before_min if before_min is not None else -1,
                    before_max if before_max is not None else -1,
                    after_min if after_min is not None else -1,
                    after_max if after_max is not None else -1,
                    geom.x(), geom.y(), geom.width(), geom.height(),
                    match_ok,
                )
                if not match_ok:
                    logger.warning(
                        "[LAYOUT] X-RANGE MISMATCH! plot[%d,%d] linked=%s "
                        "target=(%.4f,%.4f) actual=(%.4f,%.4f)",
                        row, col, linked is not None,
                        global_min, global_max,
                        after_min if after_min is not None else -1,
                        after_max if after_max is not None else -1,
                    )
                synced_count += 1
        except Exception as exc:
            sync_error = exc
            logger.warning(
                "[LAYOUT] set_plots_visible: sync failed at plot %d/%d: %s",
                synced_count, len(visible_indices), exc,
            )
        finally:
            for index in visible_indices:
                self.plot_widgets[index].plot_widget._is_syncing_range = False

        services.cursor_sync_manager._sync_min_xrange()
        if sync_error is None:
            logger.debug(
                "[LAYOUT] set_plots_visible done: synced %d visible plot(s)", synced_count,
            )

        # 布局对齐同样走 _is_syncing_range 静默路径，范围落定后补刷样式
        self.refresh_curve_styles()
        # 布局变更后调度一次 X-link 健康检查，覆盖 reload 后 link 丢失场景
        self.schedule_xlink_sync()

    def sync_xlink(self) -> None:
        """按本页 master 重建各子图 X 范围；隐藏页只记 dirty。

        XLink 同步走 ``_sync_guard``，被改范围的子图收不到可刷新的
        ``sigRangeChanged``；范围落定后必须显式补刷样式（见
        ``refresh_curve_styles``）。
        """
        if not widget_alive(self.widget):
            return
        if self.widget.isHidden():
            self._xlink_dirty = True
            return
        from src.ui.layout_manager import sync_linked_x_ranges

        self._xlink_dirty = False
        sync_linked_x_ranges(
            self, master_resolver=lambda _containers: self.master_container()
        )
        self.refresh_curve_styles()

    def refresh_curve_styles(self) -> None:
        """X 范围被程序化改写后补刷曲线样式（细线+symbol ↔ 粗线）。

        ``apply_x_range`` / 布局对齐 / XLink 同步都用 ``_is_syncing_range``
        屏蔽 ``sigRangeChanged``，样式判据（X 宽度 × 采样密度）不会随范围
        自动更新；不在范围落定处补一次，切页后就会停在旧范围的样式。

        逐子图走统一调度器的 style 任务，``_apply_plot_style`` 自带
        pen/symbol 缓存，序列未变时不会重建画笔。
        """
        for container in self.plot_widgets:
            if not widget_alive(container):
                continue
            pw = getattr(container, "plot_widget", None)
            if pw is None or getattr(pw, "_is_being_destroyed", False):
                continue
            queue = getattr(pw, "_queue_ui_refresh", None)
            if queue is None:
                continue
            try:
                queue(style=True, cursor=False, stats=False, immediate=True)
            except RuntimeError:
                logger.debug("[PAGE] 样式补刷跳过已销毁子图", exc_info=True)

    def schedule_xlink_sync(self, delay_ms: int = 50) -> None:
        """页级 50ms 防抖；回调按 epoch 作废（R4）。"""
        if self._pending_xlink_sync:
            return
        self._pending_xlink_sync = True
        epoch = self.epoch
        QTimer.singleShot(delay_ms, lambda: self._run_pending_sync(epoch))

    def _run_pending_sync(self, epoch: int) -> None:
        if epoch != self.epoch or not widget_alive(self.widget):
            return
        # 定时器落地时页可能已随窗口销毁：sync_xlink 入口还有一道存活守卫，
        # 这里先挡一次，避免 _pending_xlink_sync 在残壳上被无意义地推进。
        self._pending_xlink_sync = False
        self.sync_xlink()

    # ------------------------------------------------------------------
    # 切页范围跟随（§5.2）
    # ------------------------------------------------------------------
    def grab_master_x_range(self) -> tuple[float, float] | None:
        """切页前抓 master 的 X 范围，供新页 ``apply_x_range`` 跟随。"""
        view_box = self.master_viewbox()
        if view_box is None:
            self.saved_x_range = None
            return None
        try:
            x_min, x_max = view_box.viewRange()[0]
            x_min, x_max = float(x_min), float(x_max)
        except Exception:
            logger.debug("[PAGE] grab_master_x_range 读取失败", exc_info=True)
            self.saved_x_range = None
            return None
        if not (math.isfinite(x_min) and math.isfinite(x_max)) or x_max <= x_min:
            self.saved_x_range = None
            return None
        self.saved_x_range = (x_min, x_max)
        return self.saved_x_range

    def apply_x_range(self, x_min: float, x_max: float) -> bool:
        """把本页 master 搬到给定 X 范围，页内其余子图由 XLink 级联。

        复用既有“临时 unlink → setXRange(padding=0) → 恢复 link”套路，
        避免 pyqtgraph 的 XLink 把程序化范围按源范围静默回收。
        """
        view_box = self.master_viewbox()
        if view_box is None:
            return False
        try:
            x_min, x_max = float(x_min), float(x_max)
        except (TypeError, ValueError):
            return False
        if not (math.isfinite(x_min) and math.isfinite(x_max)) or x_max <= x_min:
            return False

        flags: dict = {}
        for container in self.plot_widgets:
            candidate = getattr(container, "plot_widget", None)
            if candidate is None:
                continue
            flags[candidate] = getattr(candidate, "_is_syncing_range", False)
            candidate._is_syncing_range = True
        try:
            linked = view_box.linkedView(0)
            if linked is not None:
                view_box.setXLink(None)
            try:
                view_box.enableAutoRange(x=False)
                view_box.setXRange(x_min, x_max, padding=0)
            finally:
                if linked is not None:
                    view_box.setXLink(linked)
        except Exception:
            logger.debug("[PAGE] apply_x_range 失败", exc_info=True)
            return False
        finally:
            for candidate, previous in flags.items():
                try:
                    candidate._is_syncing_range = previous
                except RuntimeError:
                    pass
        logger.debug("[PAGE] apply_x_range %.4f..%.4f on %s", x_min, x_max, self)
        return True

    def restore_scene(self) -> None:
        """隐藏页切回：按全局状态补建场景 item（§5.1 / §6.3 #4）。

        游标是全局限定的一份状态（``mw.cursor_mode / pinned_x_values /
        cursor_btn``），隐藏期间只标脏、不碰 pyqtgraph 场景；这里统一补建
        vline/label 与标记区。
        """
        if not widget_alive(self.widget) or self.widget.isHidden():
            return
        self._scene_dirty = False
        self.restore_cursor_scene()
        self.restore_mark_regions()

    def _apply_cursor_domain_quietly(self, plot_widget) -> None:
        """重设 X limits / vline bounds，但不让钳制结果回写全局 pin。

        ``AxisManager._set_vline_bounds`` 直接调 ``vline.setBounds()``；
        pyqtgraph 会在内部 ``setValue(self.value())`` 把越界值钳到新 bounds
        并 emit ``sigPositionChanged``。切页恢复/补齐 limits 时这条回调会把
        旧值写进 ``pinned_x_values``（单游标模式下 vline2 还会补第二份），
        污染全局状态；这里用信号阻断 + 标志位兜住，随后外面再按全局 pin 落位。
        """
        previous = getattr(plot_widget, "_suppress_pin_update", False)
        plot_widget._suppress_pin_update = True
        try:
            with ExitStack() as stack:
                for attr in ("vline", "vline2"):
                    line = getattr(plot_widget, attr, None)
                    if line is not None:
                        stack.enter_context(QSignalBlocker(line))
                plot_widget._apply_cursor_x_domain_and_limits()
        finally:
            plot_widget._suppress_pin_update = previous

    def cursor_state_signature(self) -> tuple:
        """影响游标场景的全局状态指纹；相同即可判定本页无需重建。"""
        services = self.services
        if services is None:
            return ()
        cursor_btn = getattr(services, "cursor_btn", None)
        enabled = bool(cursor_btn.isChecked()) if cursor_btn is not None else False
        pinned = tuple(getattr(services, "pinned_x_values", []) or [])
        return (
            enabled,
            bool(getattr(services, "cursor_values_hidden", False)),
            getattr(services, "cursor_mode", "1 free cursor"),
            pinned,
        )

    def sync_cursor_scene(self) -> bool:
        """激活后按全局游标状态同步本页；状态未变则不重复写（§5.1）。

        非脏页此前完全不碰游标，而拖动固定游标只同步当前页的兄弟子图，
        导致切回目标页时 vline 仍是旧值；一旦叠加“切页跟随”的视图范围，
        旧游标就会落在 viewRange 之外。脏页刚在 ``restore_scene`` 里恢复过，
        指纹相同自然跳过。
        """
        if self.services is None or not self.plot_widgets:
            return False
        if not widget_alive(self.widget) or self.widget.isHidden():
            return False
        if self.cursor_state_signature() == self._cursor_state_sig:
            return False
        self.restore_cursor_scene()
        return True

    def restore_cursor_scene(self) -> None:
        """按全局游标状态重建本页 vline / 标签（仅当前页可调用）。"""
        services = self.services
        if services is None or not self.plot_widgets:
            return
        cursor_enabled = bool(services.cursor_btn.isChecked())
        values_hidden = bool(getattr(services, "cursor_values_hidden", False))
        mode = getattr(services, "cursor_mode", "1 free cursor")
        pinned = list(getattr(services, "pinned_x_values", []) or [])
        for container in self.plot_widgets:
            plot_widget = getattr(container, "plot_widget", None)
            if plot_widget is None:
                continue
            try:
                # 隐藏期间可能改过时间基准：X limits 与游标域按全局口径重算。
                # 必须静默执行：setBounds 的钳制回写会污染全局 pinned。
                self._apply_cursor_domain_quietly(plot_widget)
                if cursor_enabled and values_hidden:
                    plot_widget.toggle_cursor(False, hide_values_only=True)
                else:
                    plot_widget.toggle_cursor(cursor_enabled)
                if cursor_enabled:
                    plot_widget.apply_cursor_mode(mode, pinned)
                if hasattr(plot_widget, "_last_cursor_update_time"):
                    plot_widget._last_cursor_update_time = 0
                safe_qt_op(plot_widget.update_cursor_label)
            except RuntimeError:
                # 页正在销毁：后续容器也不再处理，不记录指纹
                logger.debug("[PAGE] restore_cursor_scene 撞到已销毁控件", exc_info=True)
                return
        self._cursor_state_sig = self.cursor_state_signature()

    def apply_global_x_limits(self) -> None:
        """把本页各子图的 X limits / 游标域对齐到全局数据域。

        新建页是在数据加载之后创建的，没有走 ``replots_after_loading``；
        首次激活且 ``_scene_dirty`` 为 False 时补这一次，避免 ViewBox 停在
        默认哨兵 limits 导致 X 轴可无限缩放。
        """
        if not widget_alive(self.widget) or self.widget.isHidden():
            return
        for container in self.plot_widgets:
            plot_widget = getattr(container, "plot_widget", None)
            if plot_widget is None:
                continue
            try:
                self._apply_cursor_domain_quietly(plot_widget)
            except RuntimeError:
                logger.debug(
                    "[PAGE] apply_global_x_limits 撞到已销毁控件", exc_info=True
                )
                return

    def set_row_height(self, row: int, percentage: int) -> None:
        if row < 0 or row >= self.rows_max:
            return
        self.row_height_factors[row] = percentage
        grid_cols = self.grid_cols
        for candidate in range(self.rows_max):
            visible = any(
                self.cell_visible(candidate * grid_cols + col)
                for col in range(grid_cols)
            )
            if visible:
                pct = self.row_height_factors.get(candidate, 100)
                self.plot_layout.setRowStretch(candidate, max(1, pct // 25))
            else:
                self.plot_layout.setRowStretch(candidate, 0)

    def set_all_row_height(self, percentage: int) -> None:
        for row in range(self.rows_max):
            self.row_height_factors[row] = percentage
        grid_cols = self.grid_cols
        for row in range(self.rows_max):
            visible = any(
                self.cell_visible(row * grid_cols + col)
                for col in range(grid_cols)
            )
            if visible:
                pct = self.row_height_factors.get(row, 100)
                self.plot_layout.setRowStretch(row, max(1, pct // 25))
            else:
                self.plot_layout.setRowStretch(row, 0)

    def get_row_height(self, row: int) -> int:
        return self.row_height_factors.get(row, 100)

    def restore_mark_regions(self) -> None:
        """布局变更后按全局区间重建本页 mark region（页作用域）。"""
        services = self.services
        if services is None or not services.mark_region_btn.isChecked():
            return
        if self.widget.isHidden():
            self.mark_dirty_for_activation()
            return
        master = self.master_container()
        if master is None:
            return
        mark = getattr(master.plot_widget, "mark_region", None)
        if mark is not None:
            services.saved_mark_range = mark.getRegion()

        for container in self.plot_widgets:
            try:
                container.plot_widget.remove_mark_region()
            except RuntimeError:
                pass

        try:
            view_min, view_max = master.plot_widget.view_box.viewRange()[0]
        except Exception:
            logger.debug("[MARK] 读取视图范围失败，跳过区域重建", exc_info=True)
            return
        saved = services.saved_mark_range
        if saved:
            min_x, max_x = saved
        else:
            width = view_max - view_min
            min_x = view_min + width / 3
            max_x = view_min + 2 * width / 3

        for index, container in enumerate(self.plot_widgets):
            if self.cell_visible(index):
                container.plot_widget.add_mark_region(min_x, max_x)
        services.layout_manager.request_mark_stats_refresh(immediate=True)

    # ------------------------------------------------------------------
    # 纯查询：跨页路径必须用显式行列，不能用 isVisible（全局规则 R0）
    # ------------------------------------------------------------------
    def cell_visible(self, index: int) -> bool:
        """第 index 个网格是否落在当前可见行列内。"""
        if index < 0 or index >= len(self.plot_widgets):
            return False
        grid_cols = self.grid_cols
        row, col = divmod(index, grid_cols)
        return row < self.rows_current and col < self.cols_current

    def visible_containers(self) -> list[PlotContainerWidget]:
        return [
            container
            for index, container in enumerate(self.plot_widgets)
            if self.cell_visible(index)
        ]

    def container_at(self, row: int, col: int) -> PlotContainerWidget | None:
        if row < 0 or col < 0 or col >= self.grid_cols:
            return None
        index = row * self.grid_cols + col
        if index >= len(self.plot_widgets):
            return None
        return self.plot_widgets[index]

    def row_of(self, plot_widget) -> int:
        """给定子图，返回它在本页的行号；找不到返回 -1。"""
        for index, container in enumerate(self.plot_widgets):
            if container is plot_widget or container.plot_widget is plot_widget:
                return index // self.grid_cols
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
        """关闭标签页：标记销毁 → 摘场景 → 删页容器。"""
        self.epoch += 1
        self.cancel_pending()
        self.clear_matrix()
        self.widget.hide()
        self.widget.setParent(None)
        self.widget.deleteLater()

    def __repr__(self) -> str:  # pragma: no cover - 仅日志
        return (
            f"<PlotPage title={self.title!r} epoch={self.epoch} "
            f"grid={self.rows_current}x{self.cols_current}/{self.rows_max}x{self.cols_max} "
            f"plots={len(self.plot_widgets)}>"
        )
