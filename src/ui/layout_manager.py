"""
LayoutManager —— 布局管理与标记区域同步

负责 MainWindow 的布局管理、行高控制、子图矩阵创建、
标记区域同步以及时间修正对话框等功能。
"""

from __future__ import annotations
import weakref
import numpy as np

from PyQt6.QtCore import QSignalBlocker
from PyQt6.QtWidgets import QDialog, QMessageBox

from src.core.config import debug_log, UI_DEBOUNCE_DELAY_MS
from src.ui.mark_stats import MarkStatsWindow
from src.ui.dialogs.layout_input import LayoutInputDialog
from src.ui.dialogs.time_correction import TimeCorrectionDialog
from src.ui.widgets.plot_container import PlotContainerWidget
from src.app.plot_context import PlotContext


class LayoutManager:
    def __init__(self, main_window):
        self._mw_ref = weakref.ref(main_window)

    @property
    def _mw(self):
        mw = self._mw_ref()
        if mw is None:
            raise RuntimeError("MainWindow has been garbage collected")
        return mw

    def toggle_mark_region(self, checked):
        if checked:
            self._mw.mark_region_btn.setText("关闭标记")
            if len(self._mw.plot_widgets) == 0:
                self._mw.mark_region_btn.setChecked(False)
                return
            if self._mw.saved_mark_range:
                min_x, max_x = self._mw.saved_mark_range
                view_min, view_max = self._mw.plot_widgets[0].plot_widget.view_box.viewRange()[0]
                if min_x >= view_min and max_x <= view_max:
                    pass
                else:
                    width = view_max - view_min
                    min_x = view_min + width / 3
                    max_x = view_min + 2 * width / 3
            else:
                view_min, view_max = self._mw.plot_widgets[0].plot_widget.view_box.viewRange()[0]
                width = view_max - view_min
                min_x = view_min + width / 3
                max_x = view_min + 2 * width / 3

            for container in self._mw.plot_widgets:
                if container.isVisible():
                    container.plot_widget.add_mark_region(min_x, max_x)

            self._mw.mark_stats_window = MarkStatsWindow.get_instance(self._mw)
            geom = self._mw.mark_stats_window.load_geom()
            if geom:
                self._mw.mark_stats_window.restoreGeometry(geom)

            self._mw.mark_stats_window.showNormal()
            self.request_mark_stats_refresh(immediate=True)
        else:
            self._mw.mark_region_btn.setText("标记区域")
            if self._mw.plot_widgets and self._mw.plot_widgets[0].plot_widget.mark_region:
                self._mw.saved_mark_range = self._mw.plot_widgets[0].plot_widget.mark_region.getRegion()
            for container in self._mw.plot_widgets:
                container.plot_widget.remove_mark_region()
            if self._mw.mark_stats_window:
                self._mw.mark_stats_window.save_geom()
                self._mw.mark_stats_window.hide()

    def sync_mark_regions(self, region_item):
        if self._mw._is_syncing_mark_region:
            return
        self._mw._is_syncing_mark_region = True
        try:
            min_x, max_x = region_item.getRegion()
            for container in self._mw.plot_widgets:
                mark = getattr(container.plot_widget, 'mark_region', None)
                if not (container.isVisible() and mark and mark is not region_item):
                    continue
                blocker = QSignalBlocker(mark)
                mark.setRegion([min_x, max_x])
            self.request_mark_stats_refresh()
        finally:
            self._mw._is_syncing_mark_region = False

    def request_mark_stats_refresh(self, *, immediate: bool = False):
        if not getattr(self._mw, 'mark_stats_window', None):
            return
        if immediate:
            if self._mw._mark_stats_timer.isActive():
                self._mw._mark_stats_timer.stop()
            self._mw._mark_stats_dirty = False
            self.update_mark_stats()
            return
        self._mw._mark_stats_dirty = True
        self._mw._mark_stats_timer.start(UI_DEBOUNCE_DELAY_MS)

    def _flush_mark_stats_refresh(self):
        if not self._mw._mark_stats_dirty:
            return
        self._mw._mark_stats_dirty = False
        self.update_mark_stats()

    def update_mark_stats(self):
        if hasattr(self._mw, 'mark_stats_window') and self._mw.mark_stats_window:
            stats_list = []
            for container in self._mw.plot_widgets:
                if container.isVisible():
                    stats = container.plot_widget.get_mark_stats()
                    stats_list.append(stats)
            self._mw.mark_stats_window.update_stats(stats_list)

    def open_layout_dialog(self):
        dlg = LayoutInputDialog(max_rows=self._mw._plot_row_max_default,
                                max_cols=self._mw._plot_col_max_default,
                                cur_rows=self._mw._plot_row_current,
                                cur_cols=self._mw._plot_col_current,
                                parent=self._mw)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            r, c = dlg.values()
            self.set_plots_visible(r, c)
            self.update_mark_regions_on_layout_change()

    def open_time_correction_dialog(self):
        self._mw._is_time_correction_active = False
        self._mw._time_correction_pinned_index_values = []
        dialog = TimeCorrectionDialog(self._mw.factor, self._mw.offset, self._mw)
        if dialog.window_geometry:
            dialog.restoreGeometry(dialog.window_geometry)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_factor, new_offset = dialog.values()
            if new_factor <= 0:
                QMessageBox.warning(self._mw, "错误", "Factor 必须是正数")
                return
            old_factor = self._mw.factor
            old_offset = self._mw.offset
            self._mw.factor = new_factor
            self._mw.offset = new_offset
            self._mw._is_time_correction_active = True
            self._mw._time_correction_pinned_index_values = []
            try:
                if self._mw.cursor_btn.isChecked():
                    mode = getattr(self._mw, "cursor_mode", "1 free cursor")
                    if mode != "1 free cursor" and old_factor != 0 and self._mw.pinned_x_values:
                        for x_val in self._mw.pinned_x_values:
                            if x_val is None or not np.isfinite(x_val):
                                continue
                            index_pos = (x_val - old_offset) / old_factor
                            if np.isfinite(index_pos):
                                self._mw._time_correction_pinned_index_values.append(index_pos)
            except Exception:
                self._mw._time_correction_pinned_index_values = []

            if self._mw.plot_widgets:
                curr_min, curr_max = self._mw.plot_widgets[0].plot_widget.view_box.viewRange()[0]
            else:
                curr_min, curr_max = 0, 1

            for container in self._mw.plot_widgets:
                container.plot_widget.update_time_correction(new_factor, new_offset)

            if old_factor != 0:
                index_min = (curr_min - old_offset) / old_factor
                index_max = (curr_max - old_offset) / old_factor
                new_min = new_offset + new_factor * index_min
                new_max = new_offset + new_factor * index_max
            else:
                datalength = self._mw.loader.datalength if hasattr(self._mw, 'loader') else 1
                new_min = new_offset + new_factor * 1
                new_max = new_offset + new_factor * datalength

            if self._mw.plot_widgets:
                first_plot = self._mw.plot_widgets[0].plot_widget
                first_plot.view_box.enableAutoRange(x=False)
                first_plot.view_box.setXRange(new_min, new_max, padding=0)
                self._mw._realign_pinned_cursor_after_time_correction(old_factor, old_offset, new_factor, new_offset)

            self.request_mark_stats_refresh(immediate=True)
            self._mw._is_time_correction_active = False
            self._mw._time_correction_pinned_index_values = []
            return
        self._mw._is_time_correction_active = False
        self._mw._time_correction_pinned_index_values = []

    def update_mark_regions_on_layout_change(self):
        if self._mw.mark_region_btn.isChecked():
            if self._mw.plot_widgets[0] and self._mw.plot_widgets[0].plot_widget.mark_region:
                self._mw.saved_mark_range = self._mw.plot_widgets[0].plot_widget.mark_region.getRegion()

            for container in self._mw.plot_widgets:
                container.plot_widget.remove_mark_region()
            view_min, view_max = self._mw.plot_widgets[0].plot_widget.view_box.viewRange()[0]
            min_x, max_x = self._mw.saved_mark_range if self._mw.saved_mark_range else (view_min + (view_max - view_min) / 3, view_min + 2 * (view_max - view_min) / 3)
            for container in self._mw.plot_widgets:
                if container.isVisible():
                    container.plot_widget.add_mark_region(min_x, max_x)
            self.request_mark_stats_refresh(immediate=True)

    def create_subplots_matrix(self, m: int, n: int):
        from csv_plot_pyqt6 import DraggableGraphicsLayoutWidget

        for i in reversed(range(self._mw.plot_layout.count())):
            w = self._mw.plot_layout.itemAt(i).widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        self._mw.plot_widgets.clear()

        first_viewbox = None

        for r in range(m):
            for c in range(n):
                plot_widget = DraggableGraphicsLayoutWidget(self._mw.units, self._mw.data, self._mw.time_channels_infos)
                plot_widget.plot_context = PlotContext(self._mw)
                cursor_enabled = self._mw.cursor_btn.isChecked()
                if cursor_enabled and self._mw.cursor_values_hidden:
                    plot_widget.toggle_cursor(False, hide_values_only=True)
                else:
                    plot_widget.toggle_cursor(cursor_enabled)
                if cursor_enabled:
                    plot_widget.apply_cursor_mode(self._mw.cursor_mode, self._mw.pinned_x_values)

                if c == 0 and r == 0:
                    first_viewbox = plot_widget.view_box
                else:
                    plot_widget.view_box.setXLink(first_viewbox)

                container = PlotContainerWidget(plot_widget)
                container.plot_widget = plot_widget

                self._mw.plot_layout.addWidget(container, r, c)
                self._mw.plot_widgets.append(container)

        for r in range(m):
            percentage = self._mw.row_height_factors.get(r, 100)
            stretch_factor = max(1, percentage // 25)
            self._mw.plot_layout.setRowStretch(r, stretch_factor)
        for c in range(n):
            self._mw.plot_layout.setColumnStretch(c, 1)
        if self._mw.mark_region_btn.isChecked():
            self.toggle_mark_region(True)

        for r in range(m):
            if r not in self._mw.row_height_factors:
                self._mw.row_height_factors[r] = 100

    def set_row_height(self, row: int, percentage: int) -> None:
        if row < 0 or row >= self._mw._plot_row_max_default:
            return

        self._mw.row_height_factors[row] = percentage

        ncols = self._mw._plot_col_max_default
        for r in range(self._mw._plot_row_max_default):
            visible = False
            for c in range(ncols):
                idx = r * ncols + c
                if idx < len(self._mw.plot_widgets) and self._mw.plot_widgets[idx].isVisible():
                    visible = True
                    break

            if visible:
                pct = self._mw.row_height_factors.get(r, 100)
                stretch_factor = max(1, pct // 25)
                self._mw.plot_layout.setRowStretch(r, stretch_factor)
            else:
                self._mw.plot_layout.setRowStretch(r, 0)

        debug_log("LayoutManager.set_row_height row=%s percentage=%s", row, percentage)

    def set_all_row_height(self, percentage: int) -> None:
        for r in range(self._mw._plot_row_max_default):
            self._mw.row_height_factors[r] = percentage

        ncols = self._mw._plot_col_max_default
        for r in range(self._mw._plot_row_max_default):
            visible = False
            for c in range(ncols):
                idx = r * ncols + c
                if idx < len(self._mw.plot_widgets) and self._mw.plot_widgets[idx].isVisible():
                    visible = True
                    break

            if visible:
                pct = self._mw.row_height_factors.get(r, 100)
                stretch_factor = max(1, pct // 25)
                self._mw.plot_layout.setRowStretch(r, stretch_factor)
            else:
                self._mw.plot_layout.setRowStretch(r, 0)

        debug_log("LayoutManager.set_all_row_height percentage=%s", percentage)

    def get_row_height(self, row: int) -> int:
        return self._mw.row_height_factors.get(row, 100)

    def set_plots_visible(self, row_set: int = 1, col_set: int = 1):
        m, n = self._mw._plot_row_max_default, self._mw._plot_col_max_default

        for idx, container in enumerate(self._mw.plot_widgets):
            r, c = divmod(idx, n)
            visible = r < row_set and c < col_set
            container.setVisible(visible)

            if visible:
                self._mw.plot_layout.setColumnStretch(c, 1)
            else:
                self._mw.plot_layout.setColumnStretch(c, 0)

        for r in range(m):
            visible = r < row_set
            if visible:
                percentage = self._mw.row_height_factors.get(r, 100)
                stretch_factor = max(1, percentage // 25)
                self._mw.plot_layout.setRowStretch(r, stretch_factor)
            else:
                self._mw.plot_layout.setRowStretch(r, 0)

        self._mw._plot_row_current = row_set
        self._mw._plot_col_current = col_set
        self.update_mark_regions_on_layout_change()

        if self._mw.plot_widgets:
            first_plot = self._mw.plot_widgets[0].plot_widget
            curr_min, curr_max = first_plot.view_box.viewRange()[0]
            for container in self._mw.plot_widgets:
                if container.isVisible():
                    widget = container.plot_widget
                    widget.view_box.setXRange(curr_min, curr_max, padding=0)
                    widget.plot_item.update()

        self._mw._sync_min_xrange()
