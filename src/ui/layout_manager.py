"""MainWindow 布局管理器 —— 处理布局、plot 矩阵、mark region 同步等"""

from __future__ import annotations

import os
import sys
import subprocess
from contextlib import contextmanager

import numpy as np

from PySide6.QtCore import QTimer, QEvent, QSignalBlocker
from PySide6.QtWidgets import QApplication, QWidget, QMessageBox, QDialog

from src.core.config import UI_DEBOUNCE_DELAY_MS, widget_alive as _widget_alive
from src.utils.paths import normalize_input_path
from src.core.logger import get_logger
from src.ui.main_window_base_manager import MainWindowBaseManager
from src.ui.table_dialog import DataTableDialog
from src.ui.mark_stats import MarkStatsWindow
from src.ui.dialogs.help import HelpDialog
from src.ui.dialogs.layout_grid_selector import LayoutGridSelector
from src.ui.dialogs.time_correction import TimeCorrectionDialog
from src.ui.widgets.plot_container import PlotContainerWidget

logger = get_logger(__name__)


@contextmanager
def _sync_guard(*plot_widgets):
    """程序化改范围期间屏蔽子图的 ``_on_range_changed``。

    少了这层保护，XLink 同步发出的 ``sigRangeChanged`` 会让子图误入「用户交互」
    态（置 ``_is_interacting`` 并起防抖定时器）：既白跑一轮 immediate 刷新 + 兄弟
    广播，又在这段窗口里让真正的交互源被 ``_sibling_is_interaction_source()``
    当成级联结果跳过收尾。退出时按原值复原，嵌套调用拆不掉外层守卫。
    """
    saved = [(pw, getattr(pw, "_is_syncing_range", False)) for pw in plot_widgets]
    for pw, _ in saved:
        pw._is_syncing_range = True
    try:
        yield
    finally:
        for pw, was in saved:
            pw._is_syncing_range = was

def sync_linked_x_ranges(grid, *, master_resolver=None) -> None:
    """对“能提供 plot_widgets 的网格对象”做 XLink 健康检查与范围对齐。

    这是网格级算法，只依赖鸭子类型的 ``plot_widgets`` / ``_pending_xlink_sync``：
    ``PlotPage`` 与旧式 MainWindow 替身都能直接消费（组件测试依赖此兼容性）。

    Args:
        grid: 提供 ``plot_widgets`` 的页或窗口替身。
        master_resolver: 可选，接收容器列表并返回 master 容器。缺省取首个容器，
            保持 LayoutManager 既有行为；PlotPage 传入“首个显式可见容器”。
    """
    containers = list(getattr(grid, "plot_widgets", None) or [])
    if hasattr(grid, "_pending_xlink_sync"):
        grid._pending_xlink_sync = False
    if not containers:
        return

    if master_resolver is not None:
        master_container = master_resolver(containers)
    else:
        master_container = containers[0]
    if master_container is None or not _widget_alive(master_container):
        # 首块都没了 = 整窗正在销毁，后面的同步无从谈起
        return

    master_pw = getattr(master_container, "plot_widget", None)
    if master_pw is None or not hasattr(master_pw, "view_box"):
        return
    master_vb = master_pw.view_box

    # === X-link 健康检查：丢失 link 的 plot 自动重建 ===
    for index, container in enumerate(containers):
        if container is master_container or not _widget_alive(container):
            continue
        if not hasattr(container, "plot_widget"):
            continue
        view_box = container.plot_widget.view_box
        # isHidden()：隐藏页里的显式可见单元格不应被误判为「不可见」（R0）
        if view_box.linkedView(0) is None and not container.isHidden():
            logger.warning(
                "[XLINK_SYNC] plot idx=%d lost X-link, re-establishing", index
            )
            # linkView 末尾会用子图自己的 sigRangeChanged 回报新范围，
            # 不守卫就会让该子图入交互态
            with _sync_guard(container.plot_widget):
                view_box.setXLink(master_vb)

    try:
        x_range = master_vb.viewRange()[0]
    except Exception as e:
        logger.debug("[XLINK_SYNC] 获取源视图范围失败: %s", e)
        return

    xmin, xmax = x_range
    if xmin is None or xmax is None:
        return
    if abs(xmin - xmax) < 1e-12:
        return

    # 相对容差：消除 float32 精度导致的误报
    range_width = abs(xmax - xmin)
    tolerance = 1e-4 * range_width

    first_geom = master_container.geometry()
    logger.debug(
        "[XLINK_SYNC] _sync_linked_x_ranges: source range=(%.4f, %.4f) width=%.4f "
        "first_geom=(%d,%d %dx%d)",
        xmin, xmax, xmax - xmin,
        first_geom.x(), first_geom.y(), first_geom.width(), first_geom.height(),
    )

    for container in containers:
        if container is master_container:
            continue
        if not _widget_alive(container) or not hasattr(container, "plot_widget"):
            continue
        pw = container.plot_widget
        if not hasattr(pw, "view_box"):
            continue
        vb = pw.view_box
        try:
            cur_range = vb.viewRange()[0]
            if abs(cur_range[0] - xmin) < 1e-12 and abs(cur_range[1] - xmax) < 1e-12:
                continue
        except Exception:
            logger.debug(
                "[XLINK] viewRange 获取失败，跳过同步",
                exc_info=True,
            )
            continue
        linked = vb.linkedView(0)
        with _sync_guard(pw):
            if linked is not None:
                vb.setXLink(None)
                logger.debug(
                    "[XLINK] plot=%s temp unlink for sync (%.4f, %.4f)",
                    getattr(pw, 'y_name', '?'), xmin, xmax,
                )
            try:
                vb.enableAutoRange(x=False)
                vb.setXRange(xmin, xmax, padding=0)
            finally:
                if linked is not None:
                    vb.setXLink(linked)
                    logger.debug(
                        "[XLINK] plot=%s link restored after sync",
                        getattr(pw, 'y_name', '?'),
                    )

        cur_geom = container.geometry()
        try:
            new_range = vb.viewRange()[0]
            ok = (
                abs(new_range[0] - xmin) < tolerance
                and abs(new_range[1] - xmax) < tolerance
            )
        except Exception:
            logger.debug(
                "[XLINK_SYNC] viewRange 验证失败",
                exc_info=True,
            )
            new_range = (None, None)
            ok = False
        logger.debug(
            "[XLINK_SYNC]   synced plot: was=(%.4f,%.4f) now=(%.4f,%.4f) "
            "geom=(%d,%d %dx%d) verified=%s",
            cur_range[0] if cur_range else -1, cur_range[1] if cur_range else -1,
            new_range[0] if new_range[0] is not None else -1,
            new_range[1] if new_range[1] is not None else -1,
            cur_geom.x(), cur_geom.y(), cur_geom.width(), cur_geom.height(),
            ok,
        )
        if not ok:
            plot_id = getattr(pw, 'y_name', '') or str(list(getattr(pw, 'curves', {}).keys())[:2])
            vb_xlimits = vb.state.get('limits', {}).get('xLimits', [None, None])
            # 不同宽度的 linked plot 会由 linkedViewChanged 按像素几何计算 range，
            # 与 master 的 range 不同是正常行为，降级为 DEBUG
            logger.debug(
                "[XLINK_SYNC] range mismatch (expected by geometry): expected=(%.4f, %.4f) got=(%s, %s) "
                "target_xLimits=%s plot=%s",
                xmin, xmax, new_range[0], new_range[1],
                vb_xlimits, plot_id,
            )



class LayoutManager(MainWindowBaseManager):
    """布局管理器：splitter 调节、plot 矩阵、mark region 同步、事件过滤等"""

    def _mark_all_plots_being_destroyed(self, page=None):
        """把在架 plot 标成销毁期（默认全部页，兼容既有 e2e 无参调用）。

        deleteLater/setParent(None) 到骨架真正析构之间隔着一轮事件循环，期间
        排着的定时器、防抖回调仍会打到旧 plot；`destroyed` 对 QWidget 派生类
        不会回调 Python，只能在这里显式置位。
        """
        pages = [page] if page is not None else list(getattr(self.mw, "pages", []) or [])
        if pages:
            for target in pages:
                if hasattr(target, "mark_all_plots_being_destroyed"):
                    target.mark_all_plots_being_destroyed()
                else:
                    for container in target.plot_widgets:
                        plot_widget = getattr(container, "plot_widget", None)
                        if plot_widget is not None:
                            plot_widget.mark_being_destroyed()
            return
        # 旧替身（没有 pages）走原路径
        for container in getattr(self.mw, "plot_widgets", []):
            plot_widget = getattr(container, "plot_widget", None)
            if plot_widget is not None:
                plot_widget.mark_being_destroyed()

    def _handle_close(self):
        self._mark_all_plots_being_destroyed()
        if DataTableDialog._instance is not None:
            DataTableDialog._instance.set_skip_close_confirmation(True)
        self._unregister_global_event_filter()

    def _on_splitter_moved(self, pos, index):
        self.mw.var_table_user_adjusted = True
        self.mw._splitter_ready = True

        sizes = self.mw.main_splitter.sizes()
        if len(sizes) >= 1:
            self.mw.var_table_default_width = sizes[0]

    def _ensure_splitter_ready(self):
        mw = self._mw_ref()
        if mw is None:
            return
        if not _widget_alive(mw):
            # 窗口 C++ 侧已销毁：既不能摸 main_splitter，也不该再续排 50ms
            return
        if not hasattr(mw, "main_splitter"):
            return
        sizes = mw.main_splitter.sizes()
        if len(sizes) >= 2 and all(size > 0 for size in sizes):
            mw._splitter_ready = True
        elif getattr(mw, "_plot_area_visible", True):
            QTimer.singleShot(50, self._ensure_splitter_ready)
        #  else: 绘图区被用户主动隐藏时右 pane 恒为 0，这是预期状态而非「还没
        # 布局好」—— 再续排就是整段隐藏期的 20Hz 空转，由 toggle_plot_area
        # 恢复可见时重新起一次

    def _apply_fixed_splitter_width(self):
        mw = self._mw_ref()
        if mw is None or not _widget_alive(mw):
            return
        mw._pending_splitter_adjustment = False
        if (
            mw.var_table_user_adjusted
            or not getattr(mw, "_splitter_ready", False)
            or not hasattr(mw, "main_splitter")
        ):
            return

        sizes = mw.main_splitter.sizes()
        if len(sizes) < 2:
            return

        total_width = sum(sizes)
        if total_width <= 0 or total_width <= mw.var_table_default_width:
            return

        right_width = max(total_width - mw.var_table_default_width, 0)
        if right_width <= 0:
            return

        with QSignalBlocker(mw.main_splitter):
            mw.main_splitter.setSizes([mw.var_table_default_width, right_width])

    def _schedule_xlink_sync(self):
        """节流调度 X-link 健康检查与同步（50ms 防抖，按页隔离）。"""
        mw = self._mw_ref()
        if mw is None or not _widget_alive(mw):
            return
        page = getattr(mw, "current_page", None)
        if page is not None and hasattr(page, "schedule_xlink_sync"):
            page.schedule_xlink_sync(50)
            return
        # 旧替身（只有 plot_widgets/_pending_xlink_sync）走原路径
        if getattr(mw, "_pending_xlink_sync", False):
            return
        mw._pending_xlink_sync = True
        QTimer.singleShot(50, self._sync_linked_x_ranges)

    def _handle_resize(self, _event):
        if (
            not self.mw.var_table_user_adjusted
            and getattr(self.mw, "_splitter_ready", False)
            and hasattr(self.mw, "main_splitter")
        ):
            if not getattr(self.mw, "_pending_splitter_adjustment", False):
                self.mw._pending_splitter_adjustment = True
                QTimer.singleShot(0, self._apply_fixed_splitter_width)
        self._schedule_xlink_sync()

    def _sync_linked_x_ranges(self):
        mw = self._mw_ref()
        if mw is None or not _widget_alive(mw):
            # 窗口已随事件循环销毁：残留的 plot_widgets 无从同步，静默返回。
            # 注意不能先摸 self.mw —— 弱引用断开时那个 property 本身就抛 RuntimeError。
            return
        page = getattr(mw, "current_page", None)
        if page is not None and hasattr(page, "master_container"):
            sync_linked_x_ranges(
                page, master_resolver=lambda _containers: page.master_container()
            )
            return
        # 旧替身：直接消费 mw.plot_widgets（组件测试路径）
        sync_linked_x_ranges(mw)

    def toggle_plot_area(self, checked):
        if checked:
            # 记录窗口状态和 splitter 原始尺寸
            self.mw._was_maximized = self.mw.isMaximized()
            self.mw._was_fullscreen = self.mw.isFullScreen()
            self.mw._saved_geometry = self.mw.saveGeometry()
            self.mw._saved_splitter_sizes = self.mw.main_splitter.sizes()
            
            self.mw.plot_widget.hide()
            self.mw.toggle_plot_btn.setText("显示绘图区")
            
            # 通过调整 splitter 将右侧空间压缩为 0
            self.mw.main_splitter.setChildrenCollapsible(True)
            self.mw.main_splitter.setSizes([self.mw.main_splitter.width(), 0])
            self.mw._plot_area_visible = False
        else:
            self.mw.plot_widget.show()
            self.mw.toggle_plot_btn.setText("隐藏绘图区")
            
            # 恢复 splitter 原始尺寸
            if hasattr(self.mw, '_saved_splitter_sizes') and self.mw._saved_splitter_sizes:
                self.mw.main_splitter.setSizes(self.mw._saved_splitter_sizes)
            self.mw.main_splitter.setChildrenCollapsible(False)
            
            # 根据窗口状态恢复
            if self.mw.isFullScreen() or self.mw._was_fullscreen:
                self.mw.showFullScreen()
            elif self.mw.isMaximized() or self.mw._was_maximized:
                self.mw.showMaximized()
            elif self.mw._saved_geometry:
                self.mw.restoreGeometry(self.mw._saved_geometry)
            
            self.mw._was_maximized = False
            self.mw._was_fullscreen = False
            self.mw._plot_area_visible = True
            if not getattr(self.mw, "_splitter_ready", False):
                # 隐藏期间没有续排轮询，这里补起一次；否则「就绪」标记永久
                # 停在 False，_apply_fixed_splitter_width/_handle_resize 再也不工作
                QTimer.singleShot(0, self._ensure_splitter_ready)

    def show_help(self):
        dlg = HelpDialog(self.mw)
        try:
            dlg.exec()
        finally:
            dlg.deleteLater()

    def _get_plot_container(self, plot_widget) -> PlotContainerWidget | None:
        parent = plot_widget.parentWidget()
        if isinstance(parent, PlotContainerWidget):
            return parent
        return None

    def _show_drag_indicator_for_plot(
        self, plot_widget, var_names: list[str], text_override: str | None = None
    ):
        container = self._get_plot_container(plot_widget)
        if not container:
            return
        if (
            self.mw._active_drag_container
            and self.mw._active_drag_container is not container
        ):
            self.mw._active_drag_container.hide_drag_indicator()
        container.show_drag_indicator(var_names, text_override)
        self.mw._active_drag_container = container

    def _hide_drag_indicator_for_plot(self, plot_widget):
        container = self._get_plot_container(plot_widget)
        if not container:
            return
        container.hide_drag_indicator()
        if self.mw._active_drag_container is container:
            self.mw._active_drag_container = None

    def spawn_clone_window(self):
        try:
            if getattr(sys, "frozen", False):
                args = [sys.executable, "--clone-window"]
            else:
                entry_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                entry_script = os.path.join(entry_dir, "csv_plot.py")
                args = [sys.executable, entry_script, "--clone-window"]

            if sys.platform == "win32":
                subprocess.Popen(
                    args,
                    cwd=os.getcwd(),
                    creationflags=(
                        subprocess.CREATE_NEW_PROCESS_GROUP
                        | subprocess.DETACHED_PROCESS
                        | subprocess.CREATE_NO_WINDOW
                    ),
                    close_fds=True,
                )
            else:
                subprocess.Popen(
                    args,
                    cwd=os.getcwd(),
                    start_new_session=True,
                    close_fds=True,
                )
        except (OSError, subprocess.SubprocessError) as e:
            QMessageBox.warning(self.mw, "错误", f"启动独立实例失败: {e}")

    def toggle_mark_region(self, checked):
        if checked:
            self.mw.mark_region_btn.setText("关闭标记")
            self.mw.mark_region_btn.setChecked(True)
            if len(self.mw.plot_widgets) == 0:
                self.mw.mark_region_btn.setChecked(False)
                return
            current = getattr(self.mw, "current_page", None)
            reference = self._mark_reference_plot(current)
            if reference is None:
                self.mw.mark_region_btn.setChecked(False)
                return
            try:
                view_min, view_max = reference.view_box.viewRange()[0]
            except Exception:
                logger.debug("[MARK] 读取视图范围失败，放弃开启标记区", exc_info=True)
                self.mw.mark_region_btn.setChecked(False)
                return
            if view_min is None or view_max is None:
                self.mw.mark_region_btn.setChecked(False)
                return
            if self.mw.saved_mark_range:
                min_x, max_x = self.mw.saved_mark_range
                # 保存的范围超出当前视图时，重置为视图中间 1/3 区域
                if min_x < view_min or max_x > view_max:
                    width = view_max - view_min
                    min_x = view_min + width / 3
                    max_x = view_min + 2 * width / 3
            else:
                width = view_max - view_min
                min_x = view_min + width / 3
                max_x = view_min + 2 * width / 3
            # 区间是全局量：隐藏页没有 item 时按它补建（§6.4 #2）
            self.mw.saved_mark_range = (min_x, max_x)

            pages = list(getattr(self.mw, "pages", []) or [])
            if pages:
                for page in pages:
                    if page is current:
                        for index, container in enumerate(page.plot_widgets):
                            if not page.cell_visible(index):
                                continue
                            container.plot_widget.add_mark_region(min_x, max_x)
                    else:
                        # 隐藏页只记脏，切过去时 restore_mark_regions 补建
                        page.mark_dirty_for_activation()
            else:
                for container in self.mw.plot_widgets:
                    if container.isVisible():
                        container.plot_widget.add_mark_region(min_x, max_x)

            self.mw.mark_stats_window = MarkStatsWindow.get_instance(self.mw)
            self.mw.mark_stats_window.load_geom()  # 内部完成恢复，无返回值

            self.mw.mark_stats_window.showNormal()
            self.request_mark_stats_refresh(immediate=True)
        else:
            self.mw.mark_region_btn.setText("标记区域")
            self.mw.mark_region_btn.setChecked(False)
            pages = list(getattr(self.mw, "pages", []) or [])
            current = getattr(self.mw, "current_page", None)
            if pages:
                containers = [
                    container for page in pages for container in page.plot_widgets
                ]
            else:
                containers = list(getattr(self.mw, "plot_widgets", []) or [])
            # 关闭必须全量拆（§7.2 #16）：隐藏页遗留的 region 仍连着全局槽，
            # 会继续响应拖拽并改写他页区间（幽灵 region）。
            saved_range = None
            for container in containers:
                pw = getattr(container, "plot_widget", None)
                mark = getattr(pw, "mark_region", None) if pw is not None else None
                if mark is None:
                    continue
                if saved_range is None or (
                    current is not None and getattr(pw, "page", None) is current
                ):
                    try:
                        saved_range = tuple(mark.getRegion())
                    except Exception:
                        logger.debug("[MARK] 回读区间失败", exc_info=True)
            if saved_range is not None:
                self.mw.saved_mark_range = saved_range
            for container in containers:
                pw = getattr(container, "plot_widget", None)
                if pw is None:
                    continue
                try:
                    pw.remove_mark_region()
                except RuntimeError:
                    pass
            if self.mw.mark_stats_window:
                self.mw.mark_stats_window.save_geom()
                self.mw.mark_stats_window.hide()

    def _mark_reference_plot(self, page=None):
        """标记区默认区间的参考子图：页内首个可见格 → 页内首个格。"""
        target = page if page is not None else getattr(self.mw, "current_page", None)
        if target is not None:
            if hasattr(target, "master_container"):
                master = target.master_container()
                if master is not None:
                    return getattr(master, "plot_widget", None)
            for container in getattr(target, "plot_widgets", []) or []:
                pw = getattr(container, "plot_widget", None)
                if pw is not None:
                    return pw
        return None

    def sync_mark_regions(self, region_item, *, page=None):
        """同步标记区区间（全局量）。

        任一页拖动都广播到全部页（§6.4 #3）；``page`` 仅标识来源页，供诊断
        与后续页作用域扩展使用，不改变广播语义。隐藏页若尚未补建 item
        （``mark_region is None``）自然跳过，切页时由 ``restore_mark_regions``
        按 ``saved_mark_range`` 补建。
        """
        if self.mw._is_syncing_mark_region:
            return
        self.mw._is_syncing_mark_region = True
        try:
            min_x, max_x = region_item.getRegion()
            # 全局区间随拖拽即时更新：隐藏页统计与切页补建都读它
            self.mw.saved_mark_range = (min_x, max_x)
            pages = list(getattr(self.mw, "pages", []) or [])
            if pages:
                containers = [
                    container for target in pages for container in target.plot_widgets
                ]
            else:
                containers = list(getattr(self.mw, "plot_widgets", []) or [])
            for container in containers:
                mark = getattr(container.plot_widget, "mark_region", None)
                if not (mark and mark is not region_item):
                    continue
                # P1-3: 必须用 with —— QSignalBlocker(mark) 作为临时对象语句，
                # 语句结束即析构并解除阻塞，setRegion 照常发射 sigRegionChanged，
                # 阻塞完全无效（同文件 :101 是正确写法）。
                with QSignalBlocker(mark):
                    mark.setRegion([min_x, max_x])
            if page is not None:
                logger.debug(
                    "[MARK] sync from page=%s range=(%.4f, %.4f)",
                    getattr(page, "title", "?"), min_x, max_x,
                )
            self.request_mark_stats_refresh()
        finally:
            self.mw._is_syncing_mark_region = False

    def request_mark_stats_refresh(self, *, immediate: bool = False):
        if not getattr(self.mw, "mark_stats_window", None):
            return
        if immediate:
            if self.mw._mark_stats_timer.isActive():
                self.mw._mark_stats_timer.stop()
            self.mw._mark_stats_dirty = False
            self.update_mark_stats()
            return
        self.mw._mark_stats_dirty = True
        self.mw._mark_stats_timer.start(UI_DEBOUNCE_DELAY_MS)

    def _flush_mark_stats_refresh(self):
        if not self.mw._mark_stats_dirty:
            return
        self.mw._mark_stats_dirty = False
        self.update_mark_stats()

    def update_mark_stats(self):
        """汇总所有页的布局内子图，按页分组交给统计窗口（§6.4 #9）。

        跨页口径用显式行列（``page.cell_visible``），不能用 ``isVisible``
        （R0）。隐藏页尚未补建 region item 时，用全局 ``saved_mark_range``
        直接计算，保证统计覆盖全部页而不是只显示当前页。
        """
        window = getattr(self.mw, "mark_stats_window", None)
        if not window:
            return
        saved = getattr(self.mw, "saved_mark_range", None)
        entries: list[tuple[str, int, object]] = []
        pages = list(getattr(self.mw, "pages", []) or [])
        if pages:
            for page in pages:
                for index, container in enumerate(page.plot_widgets):
                    if hasattr(page, "cell_visible") and not page.cell_visible(index):
                        continue
                    pw = container.plot_widget
                    stats = pw.get_mark_stats()
                    if stats is None and getattr(pw, "mark_region", None) is None:
                        stats = pw.get_mark_stats(saved) if saved else None
                    entries.append((page.title, index, stats))
        else:
            for index, container in enumerate(self.mw.plot_widgets):
                if container.isVisible():
                    entries.append(
                        ("图 1", index, container.plot_widget.get_mark_stats())
                    )
        window.update_stats(entries)

    def open_layout_dialog(self):
        if self.mw.reject_when_loading("修改布局"):
            return
        dlg = LayoutGridSelector(
            max_rows=self.mw._plot_row_max_default,
            max_cols=self.mw._plot_col_max_default,
            cur_rows=self.mw._plot_row_current,
            cur_cols=self.mw._plot_col_current,
            parent=self.mw,
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            r, c = dlg.values()
            self.set_plots_visible(r, c)
            if hasattr(self.mw, 'plot_config_manager'):
                self.mw.plot_config_manager.save_auto_save(self.mw)
        dlg.deleteLater()

    def open_time_correction_dialog(self):
        self.mw._is_time_correction_active = False
        self.mw._time_correction_pinned_index_values = []
        dialog = TimeCorrectionDialog(self.mw.factor, self.mw.offset, self.mw)
        if dialog.window_geometry:
            dialog.restoreGeometry(dialog.window_geometry)
        accepted = dialog.exec() == QDialog.DialogCode.Accepted
        # values() 读的是弹窗里的控件，必须在 deleteLater 之前取完
        new_factor, new_offset = dialog.values() if accepted else (None, None)
        dialog.deleteLater()
        self.mw._is_time_correction_active = False
        self.mw._time_correction_pinned_index_values = []
        if not accepted:
            return
        # 校验与落地都在 apply_time_correction 里：状态栏中段的 x 轴抽屉走的是
        # 同一条，对话框这里只负责用模态框报错
        if not self.apply_time_correction(new_factor, new_offset):
            QMessageBox.warning(self.mw, "错误", "Factor 必须是正数")

    def apply_time_correction(self, new_factor: float, new_offset: float) -> bool:
        """把全局 x 轴基准改成 factor/offset，并重画所有已开子图。

        顶部「时间修正」对话框与状态栏中段的 x 轴抽屉**共用这一条**：改基准
        顺带要保住当前可视 x 范围、反算已固定游标的位置、刷新标记区统计，两处
        各写一份必然漂移。返回 False 表示 factor 非法，由调用方按自己的方式提示
        （对话框弹模态、抽屉走状态栏播报）。
        """
        if not np.isfinite(new_factor) or new_factor <= 0:
            return False

        old_factor = self.mw.factor
        old_offset = self.mw.offset
        self.mw.factor = new_factor
        self.mw.offset = new_offset
        self.mw._is_time_correction_active = True
        self.mw._time_correction_pinned_index_values = []
        try:
            if self.mw.cursor_btn.isChecked():
                mode = getattr(self.mw, "cursor_mode", "1 free cursor")
                if (
                    mode != "1 free cursor"
                    and old_factor != 0
                    and self.mw.pinned_x_values
                ):
                    for x_val in self.mw.pinned_x_values:
                        if x_val is None or not np.isfinite(x_val):
                            continue
                        index_pos = (x_val - old_offset) / old_factor
                        if np.isfinite(index_pos):
                            self.mw._time_correction_pinned_index_values.append(
                                index_pos
                            )
        except Exception:
            logger.warning(
                "时间修正 pin 值反算失败，重置为空列表",
                exc_info=True,
            )
            self.mw._time_correction_pinned_index_values = []

        try:
            if self.mw.plot_widgets:
                curr_min, curr_max = self.mw.plot_widgets[
                    0
                ].plot_widget.view_box.viewRange()[0]
            else:
                curr_min, curr_max = 0, 1

            # 数据层：所有页的曲线都必须按新基准重算；场景层（X limits /
            # vline / 标记区）只对当前页立即执行，隐藏页标脏等切页补建
            # （设计文档 §6.7 时间修正落地顺序）。
            pages = list(getattr(self.mw, "pages", None) or [])
            if pages:
                current_page = getattr(self.mw, "current_page", None)
                for page in pages:
                    scene = page is current_page
                    for container in list(getattr(page, "plot_widgets", [])):
                        widget = getattr(container, "plot_widget", None)
                        if widget is None:
                            continue
                        widget.update_time_correction(
                            new_factor, new_offset, scene=scene
                        )
            else:
                for container in self.mw.plot_widgets:
                    container.plot_widget.update_time_correction(
                        new_factor, new_offset
                    )

            if old_factor != 0:
                index_min = (curr_min - old_offset) / old_factor
                index_max = (curr_max - old_offset) / old_factor
                new_min = new_offset + new_factor * index_min
                new_max = new_offset + new_factor * index_max
            else:
                datalength = (
                    self.mw.loader.datalength
                    if hasattr(self.mw, "loader")
                    else 1
                )
                new_min = new_offset + new_factor * 1
                new_max = new_offset + new_factor * datalength

            if self.mw.plot_widgets:
                first_plot = self.mw.plot_widgets[0].plot_widget
                first_plot.view_box.enableAutoRange(x=False)
                first_plot.view_box.setXRange(new_min, new_max, padding=0)
                self.mw.cursor_sync_manager._realign_pinned_cursor_after_time_correction(
                    old_factor, old_offset, new_factor, new_offset
                )

            self.request_mark_stats_refresh(immediate=True)
        finally:
            self.mw._is_time_correction_active = False
            self.mw._time_correction_pinned_index_values = []

        self.mw._update_axis_segment()
        self.mw._broadcast(
            f"已应用 x 轴基准：系数 {new_factor:g}，偏移 {new_offset:g}"
        )
        return True

    def update_mark_regions_on_layout_change(self, page=None):
        """布局变更后重建标记区（页作用域，设计文档 §6.4 #5）。"""
        target = page if page is not None else getattr(self.mw, "current_page", None)
        if target is not None and hasattr(target, "restore_mark_regions"):
            target.restore_mark_regions()

    def _unregister_global_event_filter(self):
        if not getattr(self.mw, "_drop_event_filter_registered", False):
            return
        app = QApplication.instance()
        if app:
            app.removeEventFilter(self.mw)
        self.mw._drop_event_filter_registered = False

    def _is_supported_drop(self, mime_data) -> bool:
        """检查 MIME 数据是否包含支持的文件类型 URL

        先 normalize 再判后缀：Windows 上拖拽的 UNC 路径经 Qt 会变成
        ``//host/share/x.csv``，不规范化会让判定与后续加载用两种形态。
        """
        if not mime_data.hasUrls():
            return False
        urls = mime_data.urls()
        return any(
            normalize_input_path(u.toLocalFile()).lower().endswith(
                (".csv", ".txt", ".mfile", ".t00", ".t01", ".t10", ".t11", ".xlsx", ".xlsm")
            )
            or self.mw.file_loader_manager._extract_file_extension(
                normalize_input_path(u.toLocalFile())
            )
            is not None
            for u in urls
        )

    def _handle_event_filter(self, obj, event):
        mw = self._mw_ref()
        if mw is None or not isinstance(obj, QWidget):
            return False
        if obj.window() is not mw:
            return False
        etype = event.type()
        if etype == QEvent.Type.DragEnter:
            if event.mimeData().hasUrls():
                if self._is_supported_drop(event.mimeData()):
                    self.show_drop_overlay()
                    self.mw.drop_overlay.adjust_text(file_type_supported=True)
                    event.acceptProposedAction()
                    return True
                else:
                    self.show_drop_overlay()
                    self.mw.drop_overlay.adjust_text(file_type_supported=False)
                    event.ignore()
                    return True
            # 非 URL 拖入（如文本变量拖拽），交由子控件处理
            return False
        elif etype == QEvent.Type.DragLeave:
            self.hide_drop_overlay()
            return True
        elif etype == QEvent.Type.DragMove:
            if event.mimeData().hasUrls():
                if self._is_supported_drop(event.mimeData()):
                    event.acceptProposedAction()
                    return True
            # 非 URL 拖入（如文本变量拖拽），交由子控件处理
            return False
        elif etype == QEvent.Type.Drop:
            self.hide_drop_overlay()
            if event.mimeData().hasUrls():
                urls = event.mimeData().urls()
                for u in urls:
                    # 拖拽是本次路径报障的唯一源头：Qt 的 toLocalFile() 在 Windows
                    # 上把 UNC 产出成 "//host/share/x.csv"（正斜杠 + 主机名小写），
                    # 不规范化就会一路原样存进 loader.path 并被用户复制出去
                    path = normalize_input_path(u.toLocalFile())
                    if (
                        path.lower().endswith(
                            (".csv", ".txt", ".mfile", ".t00", ".t01", ".t10", ".t11", ".xlsx", ".xlsm")
                        )
                        or self.mw.file_loader_manager._extract_file_extension(path) is not None
                    ):
                        self.mw.file_loader_manager.load_csv_file(path)
                        event.accept()
                        return True
        return False

    def show_drop_overlay(self):
        self.mw.drop_overlay.setGeometry(self.mw.centralWidget().rect())
        self.mw.drop_overlay.raise_()
        self.mw.drop_overlay.show()
        self.mw.drop_overlay.activateWindow()

    def hide_drop_overlay(self):
        self.mw.drop_overlay.hide()

    def create_subplots_matrix(self, m: int, n: int):
        """建矩阵门面（默认当前页）：既有 e2e 直接调用，签名保持不变。"""
        page = getattr(self.mw, "current_page", None)
        if page is None or not hasattr(page, "build_matrix"):
            logger.warning("[LAYOUT] create_subplots_matrix: 当前没有可用页，忽略")
            return
        page.build_matrix(m, n, deps=getattr(page, "services", None))

    def set_row_height(self, row: int, percentage: int) -> None:
        page = getattr(self.mw, "current_page", None)
        if page is not None:
            page.set_row_height(row, percentage)

    def set_all_row_height(self, percentage: int) -> None:
        page = getattr(self.mw, "current_page", None)
        if page is not None:
            page.set_all_row_height(percentage)

    def get_row_height(self, row: int) -> int:
        page = getattr(self.mw, "current_page", None)
        if page is not None:
            return page.get_row_height(row)
        return 100

    def set_plots_visible(self, row_set: int = 1, col_set: int = 1, page=None):
        """设置可见绘图区（门面，默认当前页）。"""
        target = page if page is not None else getattr(self.mw, "current_page", None)
        if target is None:
            return
        logger.debug(
            "[LAYOUT] set_plots_visible: rows=%d cols=%d (max %dx%d)",
            row_set, col_set, target.rows_max, target.cols_max,
        )
        target.apply_visibility(row_set, col_set)

    def _apply_visibility(self, row_set, col_set, m=None, n=None):
        """兼容旧签名：可见性 + 同步 + stretch 统一由页处理。"""
        page = getattr(self.mw, "current_page", None)
        if page is not None:
            page.apply_visibility(row_set, col_set)

    def _sync_xlink_after_visibility_change(self, row_set, col_set):
        page = getattr(self.mw, "current_page", None)
        if page is not None:
            page.sync_xlink_after_visibility_change(row_set, col_set)

    def _adjust_stretch_and_range(self, row_set, col_set):
        page = getattr(self.mw, "current_page", None)
        if page is not None:
            page.adjust_stretch_and_range(row_set, col_set)
