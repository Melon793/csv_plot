"""MainWindow 布局管理器 —— 处理布局、plot 矩阵、mark region 同步等"""

from __future__ import annotations

import os
import sys
import subprocess

import numpy as np

from PySide6.QtCore import QTimer, QEvent, QSignalBlocker
from PySide6.QtWidgets import QApplication, QWidget, QMessageBox, QDialog

from src.core.config import UI_DEBOUNCE_DELAY_MS
from src.ui.main_window_base_manager import MainWindowBaseManager
from src.ui.table_dialog import DataTableDialog
from src.ui.mark_stats import MarkStatsWindow
from src.ui.dialogs.help import HelpDialog
from src.ui.dialogs.layout_input import LayoutInputDialog
from src.ui.dialogs.time_correction import TimeCorrectionDialog
from src.ui.widgets.plot_container import PlotContainerWidget
from src.app.plot_context import PlotContext


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

        mw.main_splitter.blockSignals(True)
        mw.main_splitter.setSizes([mw.var_table_default_width, right_width])
        mw.main_splitter.blockSignals(False)

    def _handle_resize(self, _event):
        if (
            not self.mw.var_table_user_adjusted
            and getattr(self.mw, "_splitter_ready", False)
            and hasattr(self.mw, "main_splitter")
        ):
            if not getattr(self.mw, "_pending_splitter_adjustment", False):
                self.mw._pending_splitter_adjustment = True
                QTimer.singleShot(0, self._apply_fixed_splitter_width)
        # 窗口 resize（含最大化/还原）后，显式同步所有联动 ViewBox 的 x 范围。
        # pyqtgraph 的 setXLink 仅在源 ViewBox 的 range 发生变化时才同步，
        # 而 resize 时源 ViewBox 的 range 可能不变（仅像素尺寸变化），
        # 导致被联动的 ViewBox 不同步，出现 x 轴不一致的问题。
        # 使用去抖标志避免快速连续 resize 时 timer 堆积。
        if not getattr(self.mw, "_pending_xlink_sync", False):
            self.mw._pending_xlink_sync = True
            QTimer.singleShot(50, self._sync_linked_x_ranges)

    def _sync_linked_x_ranges(self):
        """显式同步所有联动 ViewBox 的 x 范围到第一个 plot"""
        self.mw._pending_xlink_sync = False
        if not self.mw.plot_widgets:
            return
        first_container = self.mw.plot_widgets[0]
        if not first_container or not hasattr(first_container, "plot_widget"):
            return
        first_pw = first_container.plot_widget
        if not hasattr(first_pw, "view_box"):
            return

        first_vb = first_pw.view_box
        try:
            x_range = first_vb.viewRange()[0]
        except Exception:
            return

        xmin, xmax = x_range
        if xmin is None or xmax is None:
            return
        if abs(xmin - xmax) < 1e-12:
            return

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
                    continue  # 已同步，跳过
            except Exception:
                continue
            # 临时断开联动，设置范围后再恢复，避免触发递归信号
            linked = vb.linkedView(0)
            if linked is not None:
                vb.setXLink(None)
            # 被 setXLink 联动的 ViewBox 不应独立 auto-range
            # （否则会在源范围变化时弹回自身数据范围，与联动语义冲突），
            # 此处显式禁用以确保联动行为正确。
            vb.enableAutoRange(x=False)
            vb.setXRange(xmin, xmax, padding=0)
            if linked is not None:
                vb.setXLink(linked)

    def toggle_plot_area(self, checked):
        if checked:
            self.mw._saved_geometry = self.mw.saveGeometry()
            self.mw.plot_widget.hide()
            self.mw.toggle_plot_btn.setText("显示绘图区")

            self.mw._old_max_width = self.mw.maximumWidth()
            left_width = self.mw.left_widget.width()
            main_margin = self.mw.centralWidget().layout().contentsMargins()
            left_width += main_margin.left() + main_margin.right()
            frame_width = self.mw.frameGeometry().width() - self.mw.width()
            new_width = left_width + frame_width
            self.mw.setFixedWidth(new_width)
            self.mw._plot_area_visible = False
        else:
            self.mw.setMaximumWidth(self.mw._old_max_width)
            self.mw.setMinimumWidth(0)
            self.mw.plot_widget.show()
            self.mw.toggle_plot_btn.setText("隐藏绘图区")
            if self.mw._saved_geometry:
                self.mw.restoreGeometry(self.mw._saved_geometry)
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
                if min_x >= view_min and max_x <= view_max:
                    pass
                else:
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
        dlg = LayoutInputDialog(
            max_rows=self.mw._plot_row_max_default,
            max_cols=self.mw._plot_col_max_default,
            cur_rows=self.mw._plot_row_current,
            cur_cols=self.mw._plot_col_current,
            parent=self.mw,
        )
        if dlg.exec() == QDialog.DialogCode.Accepted:
            r, c = dlg.values()
            self.set_plots_visible(r, c)
            self.update_mark_regions_on_layout_change()
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
                    self.mw._realign_pinned_cursor_after_time_correction(
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

    def _handle_event_filter(self, obj, event):
        if not isinstance(obj, QWidget):
            return False
        if obj.window() is not self.mw:
            return False
        etype = event.type()
        if etype == QEvent.Type.DragEnter:
            if event.mimeData().hasUrls():
                urls = event.mimeData().urls()
                supported = any(
                    u.toLocalFile()
                    .lower()
                    .endswith(
                        (".csv", ".txt", ".mfile", ".t00", ".t01", ".t10", ".t11", ".xlsx", ".xlsm")
                    )
                    or self.mw.file_loader_manager._extract_file_extension(u.toLocalFile()) is not None
                    for u in urls
                )

                if supported:
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
            if event.mimeData().hasUrls():
                urls = event.mimeData().urls()
                supported = any(
                    u.toLocalFile()
                    .lower()
                    .endswith(
                        (".csv", ".txt", ".mfile", ".t00", ".t01", ".t10", ".t11", ".xlsx", ".xlsm")
                    )
                    or self.mw.file_loader_manager._extract_file_extension(u.toLocalFile()) is not None
                    for u in urls
                )
                if supported:
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
        m, n = self.mw._plot_row_max_default, self.mw._plot_col_max_default

        for idx, container in enumerate(self.mw.plot_widgets):
            r, c = divmod(idx, n)
            visible = r < row_set and c < col_set
            container.setVisible(visible)

            if visible:
                self.mw.plot_layout.setColumnStretch(c, 1)
            else:
                self.mw.plot_layout.setColumnStretch(c, 0)

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

        if self.mw.plot_widgets:
            first_plot = self.mw.plot_widgets[0].plot_widget
            curr_min, curr_max = first_plot.view_box.viewRange()[0]
            for container in self.mw.plot_widgets:
                if container.isVisible():
                    widget = container.plot_widget
                    widget.view_box.setXRange(curr_min, curr_max, padding=0)
                    widget.plot_item.update()

        self.mw._sync_min_xrange()
