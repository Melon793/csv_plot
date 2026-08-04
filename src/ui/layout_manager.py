"""MainWindow 布局管理器 —— 处理布局、plot 矩阵、mark region 同步等"""

from __future__ import annotations

import os
import sys
import subprocess

import numpy as np

from PySide6.QtCore import QTimer, QEvent, QSignalBlocker
from PySide6.QtWidgets import QApplication, QWidget, QMessageBox, QDialog

from src.core.config import UI_DEBOUNCE_DELAY_MS
from src.core.logger import get_logger
from src.ui.main_window_base_manager import MainWindowBaseManager
from src.ui.table_dialog import DataTableDialog
from src.ui.mark_stats import MarkStatsWindow
from src.ui.dialogs.help import HelpDialog
from src.ui.dialogs.layout_grid_selector import LayoutGridSelector
from src.ui.dialogs.time_correction import TimeCorrectionDialog
from src.ui.widgets.plot_container import PlotContainerWidget
from src.app.plot_context import PlotContext

logger = get_logger(__name__)


class LayoutManager(MainWindowBaseManager):
    """布局管理器：splitter 调节、plot 矩阵、mark region 同步、事件过滤等"""

    def _handle_close(self):
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
        if not hasattr(mw, "main_splitter"):
            return
        sizes = mw.main_splitter.sizes()
        if len(sizes) >= 2 and all(size > 0 for size in sizes):
            mw._splitter_ready = True
        else:
            QTimer.singleShot(50, self._ensure_splitter_ready)

    def _apply_fixed_splitter_width(self):
        mw = self._mw_ref()
        if mw is None:
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
        """节流调度 X-link 健康检查与同步（50ms 防抖）"""
        if getattr(self.mw, "_pending_xlink_sync", False):
            return
        self.mw._pending_xlink_sync = True
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
        self.mw._pending_xlink_sync = False
        if not self.mw.plot_widgets:
            return

        # === X-link 健康检查：丢失 link 的 plot 自动重建 ===
        master_vb = self.mw.plot_widgets[0].plot_widget.view_box
        for idx, container in enumerate(self.mw.plot_widgets):
            if idx == 0 or not container or not hasattr(container, "plot_widget"):
                continue
            vb = container.plot_widget.view_box
            if vb.linkedView(0) is None and container.isVisible():
                logger.warning(
                    "[XLINK_SYNC] plot idx=%d lost X-link, re-establishing", idx
                )
                vb.setXLink(master_vb)

        first_container = self.mw.plot_widgets[0]
        if not first_container or not hasattr(first_container, "plot_widget"):
            return
        first_pw = first_container.plot_widget
        if not hasattr(first_pw, "view_box"):
            return

        first_vb = first_pw.view_box
        try:
            x_range = first_vb.viewRange()[0]
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

        first_geom = first_container.geometry()
        logger.debug(
            "[XLINK_SYNC] _sync_linked_x_ranges: source range=(%.4f, %.4f) width=%.4f "
            "first_geom=(%d,%d %dx%d)",
            xmin, xmax, xmax - xmin,
            first_geom.x(), first_geom.y(), first_geom.width(), first_geom.height(),
        )

        for container in self.mw.plot_widgets[1:]:
            if not container or not hasattr(container, "plot_widget"):
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

    def show_help(self):
        dlg = HelpDialog(self.mw)
        dlg.exec()

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
            if self.mw.saved_mark_range:
                min_x, max_x = self.mw.saved_mark_range
                view_min, view_max = self.mw.plot_widgets[
                    0
                ].plot_widget.view_box.viewRange()[0]
                # 保存的范围超出当前视图时，重置为视图中间 1/3 区域
                if min_x < view_min or max_x > view_max:
                    width = view_max - view_min
                    min_x = view_min + width / 3
                    max_x = view_min + 2 * width / 3
            else:
                view_min, view_max = self.mw.plot_widgets[
                    0
                ].plot_widget.view_box.viewRange()[0]
                width = view_max - view_min
                min_x = view_min + width / 3
                max_x = view_min + 2 * width / 3

            for container in self.mw.plot_widgets:
                if container.isVisible():
                    container.plot_widget.add_mark_region(min_x, max_x)

            self.mw.mark_stats_window = MarkStatsWindow.get_instance(self.mw)
            geom = self.mw.mark_stats_window.load_geom()
            if geom:
                self.mw.mark_stats_window.restoreGeometry(geom)

            self.mw.mark_stats_window.showNormal()
            self.request_mark_stats_refresh(immediate=True)
        else:
            self.mw.mark_region_btn.setText("标记区域")
            self.mw.mark_region_btn.setChecked(False)
            if self.mw.plot_widgets and self.mw.plot_widgets[0].plot_widget.mark_region:
                self.mw.saved_mark_range = self.mw.plot_widgets[
                    0
                ].plot_widget.mark_region.getRegion()
            for container in self.mw.plot_widgets:
                container.plot_widget.remove_mark_region()
            if self.mw.mark_stats_window:
                self.mw.mark_stats_window.save_geom()
                self.mw.mark_stats_window.hide()

    def sync_mark_regions(self, region_item):
        if self.mw._is_syncing_mark_region:
            return
        self.mw._is_syncing_mark_region = True
        try:
            min_x, max_x = region_item.getRegion()
            for container in self.mw.plot_widgets:
                mark = getattr(container.plot_widget, "mark_region", None)
                if not (container.isVisible() and mark and mark is not region_item):
                    continue
                QSignalBlocker(mark)
                mark.setRegion([min_x, max_x])
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
        if hasattr(self.mw, "mark_stats_window") and self.mw.mark_stats_window:
            stats_list = []
            for container in self.mw.plot_widgets:
                if container.isVisible():
                    stats = container.plot_widget.get_mark_stats()
                    stats_list.append(stats)
            self.mw.mark_stats_window.update_stats(stats_list)

    def open_layout_dialog(self):
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

    def open_time_correction_dialog(self):
        self.mw._is_time_correction_active = False
        self.mw._time_correction_pinned_index_values = []
        dialog = TimeCorrectionDialog(self.mw.factor, self.mw.offset, self.mw)
        if dialog.window_geometry:
            dialog.restoreGeometry(dialog.window_geometry)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_factor, new_offset = dialog.values()
            if new_factor <= 0:
                QMessageBox.warning(self.mw, "错误", "Factor 必须是正数")
                return
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
            return
        self.mw._is_time_correction_active = False
        self.mw._time_correction_pinned_index_values = []

    def update_mark_regions_on_layout_change(self):
        if self.mw.mark_region_btn.isChecked():
            if (
                self.mw.plot_widgets[0]
                and self.mw.plot_widgets[0].plot_widget.mark_region
            ):
                self.mw.saved_mark_range = self.mw.plot_widgets[
                    0
                ].plot_widget.mark_region.getRegion()

            for container in self.mw.plot_widgets:
                container.plot_widget.remove_mark_region()
            view_min, view_max = self.mw.plot_widgets[
                0
            ].plot_widget.view_box.viewRange()[0]
            min_x, max_x = (
                self.mw.saved_mark_range
                if self.mw.saved_mark_range
                else (
                    view_min + (view_max - view_min) / 3,
                    view_min + 2 * (view_max - view_min) / 3,
                )
            )
            for container in self.mw.plot_widgets:
                if container.isVisible():
                    container.plot_widget.add_mark_region(min_x, max_x)
            self.request_mark_stats_refresh(immediate=True)

    def _unregister_global_event_filter(self):
        if not getattr(self.mw, "_drop_event_filter_registered", False):
            return
        app = QApplication.instance()
        if app:
            app.removeEventFilter(self.mw)
        self.mw._drop_event_filter_registered = False

    def _is_supported_drop(self, mime_data) -> bool:
        """检查 MIME 数据是否包含支持的文件类型 URL"""
        if not mime_data.hasUrls():
            return False
        urls = mime_data.urls()
        return any(
            u.toLocalFile().lower().endswith(
                (".csv", ".txt", ".mfile", ".t00", ".t01", ".t10", ".t11", ".xlsx", ".xlsm")
            )
            or self.mw.file_loader_manager._extract_file_extension(u.toLocalFile()) is not None
            for u in urls
        )

    def _handle_event_filter(self, obj, event):
        if not isinstance(obj, QWidget):
            return False
        if obj.window() is not self.mw:
            return False
        etype = event.type()
        if etype == QEvent.Type.DragEnter:
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
        elif etype == QEvent.Type.DragLeave:
            self.hide_drop_overlay()
            return True
        elif etype == QEvent.Type.DragMove:
            if self._is_supported_drop(event.mimeData()):
                event.acceptProposedAction()
                return True
        elif etype == QEvent.Type.Drop:
            self.hide_drop_overlay()
            if event.mimeData().hasUrls():
                urls = event.mimeData().urls()
                for u in urls:
                    path = u.toLocalFile()
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
        from src.ui.widgets.plot_widget import DraggableGraphicsLayoutWidget

        for i in reversed(range(self.mw.plot_layout.count())):
            w = self.mw.plot_layout.itemAt(i).widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        self.mw.plot_widgets.clear()

        first_viewbox = None

        for r in range(m):
            for c in range(n):
                plot_widget = DraggableGraphicsLayoutWidget(
                    self.mw.units, self.mw.data, self.mw.time_channels_infos
                )
                plot_widget.plot_context = PlotContext(self.mw)
                cursor_enabled = self.mw.cursor_btn.isChecked()
                if cursor_enabled and self.mw.cursor_values_hidden:
                    plot_widget.toggle_cursor(False, hide_values_only=True)
                else:
                    plot_widget.toggle_cursor(cursor_enabled)
                if cursor_enabled:
                    plot_widget.apply_cursor_mode(
                        self.mw.cursor_mode, self.mw.pinned_x_values
                    )

                if c == 0 and r == 0:
                    first_viewbox = plot_widget.view_box
                else:
                    plot_widget.view_box.setXLink(first_viewbox)

                container = PlotContainerWidget(plot_widget)
                container.plot_widget = plot_widget

                self.mw.plot_layout.addWidget(container, r, c)
                self.mw.plot_widgets.append(container)

        for r in range(m):
            percentage = self.mw.row_height_factors.get(r, 100)
            stretch_factor = max(1, percentage // 25)
            self.mw.plot_layout.setRowStretch(r, stretch_factor)
        for c in range(n):
            self.mw.plot_layout.setColumnStretch(c, 1)
        if self.mw.mark_region_btn.isChecked():
            self.toggle_mark_region(True)

        for r in range(m):
            if r not in self.mw.row_height_factors:
                self.mw.row_height_factors[r] = 100

    def set_row_height(self, row: int, percentage: int) -> None:
        if row < 0 or row >= self.mw._plot_row_max_default:
            return

        self.mw.row_height_factors[row] = percentage

        ncols = self.mw._plot_col_max_default
        for r in range(self.mw._plot_row_max_default):
            visible = False
            for c in range(ncols):
                idx = r * ncols + c
                if (
                    idx < len(self.mw.plot_widgets)
                    and self.mw.plot_widgets[idx].isVisible()
                ):
                    visible = True
                    break

            if visible:
                pct = self.mw.row_height_factors.get(r, 100)
                stretch_factor = max(1, pct // 25)
                self.mw.plot_layout.setRowStretch(r, stretch_factor)
            else:
                self.mw.plot_layout.setRowStretch(r, 0)

    def set_all_row_height(self, percentage: int) -> None:
        for r in range(self.mw._plot_row_max_default):
            self.mw.row_height_factors[r] = percentage

        ncols = self.mw._plot_col_max_default
        for r in range(self.mw._plot_row_max_default):
            visible = False
            for c in range(ncols):
                idx = r * ncols + c
                if (
                    idx < len(self.mw.plot_widgets)
                    and self.mw.plot_widgets[idx].isVisible()
                ):
                    visible = True
                    break

            if visible:
                pct = self.mw.row_height_factors.get(r, 100)
                stretch_factor = max(1, pct // 25)
                self.mw.plot_layout.setRowStretch(r, stretch_factor)
            else:
                self.mw.plot_layout.setRowStretch(r, 0)

    def get_row_height(self, row: int) -> int:
        return self.mw.row_height_factors.get(row, 100)

    def set_plots_visible(self, row_set: int = 1, col_set: int = 1):
        """设置可见 Plot 区域"""
        m, n = self.mw._plot_row_max_default, self.mw._plot_col_max_default
        logger.debug(
            "[LAYOUT] set_plots_visible: rows=%d cols=%d (max %dx%d)",
            row_set, col_set, m, n,
        )
        self._apply_visibility(row_set, col_set, m, n)
        self._sync_xlink_after_visibility_change(row_set, col_set)
        self._adjust_stretch_and_range(row_set, col_set)

    def _apply_visibility(self, row_set, col_set, m, n):
        """批量设置 container 可见性"""
        for idx, container in enumerate(self.mw.plot_widgets):
            r, c = divmod(idx, n)
            container.setVisible(r < row_set and c < col_set)

    def _sync_xlink_after_visibility_change(self, row_set, col_set):
        """可见性变更后同步 X-link"""
        n = self.mw._plot_col_max_default
        master_vb = self.mw.plot_widgets[0].plot_widget.view_box if self.mw.plot_widgets else None
        for idx, container in enumerate(self.mw.plot_widgets):
            if idx == 0:
                continue
            r, c = divmod(idx, n)
            should_be_visible = r < row_set and c < col_set
            vb = container.plot_widget.view_box
            currently_linked = vb.linkedView(0) is not None
            if should_be_visible and not currently_linked:
                logger.debug("[XLINK] set_plots_visible: establishing link for plot idx=%d r=%d c=%d", idx, r, c)
                vb.setXLink(master_vb)
            elif not should_be_visible and currently_linked:
                logger.debug("[XLINK] set_plots_visible: breaking link for hidden plot idx=%d r=%d c=%d", idx, r, c)
                vb.setXLink(None)

    def _adjust_stretch_and_range(self, row_set, col_set):
        """调整 stretch 因子和 X 范围"""
        m = self.mw._plot_row_max_default
        n = self.mw._plot_col_max_default

        for c in range(n):
            has_visible = any(
                self.mw.plot_widgets[r * n + c].isVisible()
                for r in range(m)
                if r * n + c < len(self.mw.plot_widgets)
            )
            self.mw.plot_layout.setColumnStretch(c, 1 if has_visible else 0)
            logger.debug(
                "[LAYOUT] setColumnStretch col=%d stretch=%s has_visible=%s",
                c, 1 if has_visible else 0, has_visible,
            )

        for r in range(m):
            visible = r < row_set
            if visible:
                percentage = self.mw.row_height_factors.get(r, 100)
                stretch_factor = max(1, percentage // 25)
                self.mw.plot_layout.setRowStretch(r, stretch_factor)
            else:
                self.mw.plot_layout.setRowStretch(r, 0)

        self.mw._plot_row_current = row_set
        self.mw._plot_col_current = col_set
        self.update_mark_regions_on_layout_change()

        if not self.mw.plot_widgets:
            self.mw.cursor_sync_manager._sync_min_xrange()
            return

        visible_count = sum(1 for c in self.mw.plot_widgets if c.isVisible())
        if visible_count == 0:
            logger.debug("[LAYOUT] no visible plots, skipping X-range sync")
            self.mw.cursor_sync_manager._sync_min_xrange()
            return

        # 优先使用第一个可见 plot 的当前视图范围（保留用户的缩放状态），
        # 曲线数据全范围仅作 fallback
        first_visible = next(
            (c for c in self.mw.plot_widgets if c.isVisible()), None
        )
        global_min, global_max = None, None

        if first_visible is not None:
            try:
                global_min, global_max = first_visible.plot_widget.view_box.viewRange()[0]
                logger.debug(
                    "[LAYOUT] X-range sync source: first_visible viewRange=(%.4f, %.4f)",
                    global_min, global_max,
                )
            except Exception:
                logger.debug("[LAYOUT] failed to read first visible plot viewRange, falling back")

        if global_min is None or global_max is None:
            global_min, global_max = self.mw.cursor_sync_manager.collect_global_x_range()
            if global_min is not None:
                logger.debug(
                    "[LAYOUT] X-range sync fallback: collect_global_x_range=(%.4f, %.4f)",
                    global_min, global_max,
                )

        if global_min is None or global_max is None:
            self.mw.cursor_sync_manager._sync_min_xrange()
            return

        synced_count = 0
        sync_error = None
        for container in self.mw.plot_widgets:
            if container.isVisible():
                container.plot_widget._is_syncing_range = True
        try:
            for idx, container in enumerate(self.mw.plot_widgets):
                if not container.isVisible():
                    continue
                widget = container.plot_widget
                vb = widget.view_box
                linked = vb.linkedView(0)
                geom = container.geometry()

                before_min, before_max = None, None
                try:
                    before_min, before_max = vb.viewRange()[0]
                except Exception:
                    logger.debug("获取 viewRange 失败", exc_info=True)

                if linked is not None:
                    vb.setXLink(None)
                    logger.debug(
                        "[XLINK] plot=%s temp unlink for sync_all (%.4f, %.4f)",
                        getattr(container.plot_widget, 'y_name', f'idx={idx}'),
                        global_min, global_max,
                    )
                try:
                    vb.enableAutoRange(x=False)
                    vb.setXRange(global_min, global_max, padding=0)
                finally:
                    if linked is not None:
                        vb.setXLink(linked)
                        logger.debug(
                            "[XLINK] plot=%s link restored after sync_all",
                            getattr(container.plot_widget, 'y_name', f'idx={idx}'),
                        )

                after_min, after_max = None, None
                try:
                    after_min, after_max = vb.viewRange()[0]
                except Exception:
                    logger.debug("获取 viewRange 失败", exc_info=True)

                r, c = divmod(idx, n)
                is_xlinked = linked is not None
                match_ok = (
                    after_min is not None
                    and after_max is not None
                    and abs(after_min - global_min) < 1e-6
                    and abs(after_max - global_max) < 1e-6
                )
                logger.debug(
                    "[LAYOUT]   plot[%d,%d] idx=%d linked=%s before=(%.4f,%.4f) "
                    "after=(%.4f,%.4f) geom=(%d,%d %dx%d) match=%s",
                    r, c, idx, is_xlinked,
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
                        r, c, is_xlinked,
                        global_min, global_max,
                        after_min if after_min is not None else -1,
                        after_max if after_max is not None else -1,
                    )
                synced_count += 1
        except Exception as e:
            sync_error = e
            logger.warning(
                "[LAYOUT] set_plots_visible: sync failed at plot %d/%d: %s",
                synced_count, visible_count, e,
            )
        finally:
            for container in self.mw.plot_widgets:
                if container.isVisible():
                    container.plot_widget._is_syncing_range = False

        self.mw.cursor_sync_manager._sync_min_xrange()
        if sync_error is None:
            logger.debug(
                "[LAYOUT] set_plots_visible done: synced %d visible plot(s)", synced_count,
            )

        # 布局变更后调度一次 X-link 健康检查，覆盖 reload 后 link 丢失场景
        self._schedule_xlink_sync()
