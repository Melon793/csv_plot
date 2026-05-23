"""
MultiCurveManager - 多曲线绘图管理

负责 DraggableGraphicsLayoutWidget 的多曲线绘图、样式和可见性管理功能：
- 多曲线添加和移除
- 曲线样式管理
- 可见性切换
- 图例管理

此模块从 csv_plot_pyqt6.py 迁移而来。
"""

from __future__ import annotations
from typing import Any, TYPE_CHECKING

from src.core.config import (
    DEFAULT_LINE_WIDTH,
    DEFAULT_PADDING_VAL_Y,
)
from src.core.data_types import CurveInfo

if TYPE_CHECKING:
    from src.ui.widgets.plot_data_manager import PlotDataManager


class MultiCurveManager:
    """负责多曲线绘图和样式管理"""

    def __init__(self, plot_data_manager: PlotDataManager):
        if plot_data_manager is None:
            raise ValueError(
                "MultiCurveManager requires a valid PlotDataManager instance"
            )
        self._data_manager = plot_data_manager

    @property
    def pw(self) -> Any:
        return self._data_manager.pw

    def update_multi_curve_mode(self):
        """更新多曲线模式状态"""
        pw = self.pw
        curve_count = len(pw.curves)

        if not hasattr(pw, "_batch_adding"):
            pw._batch_adding = False

        if not pw._batch_adding:
            pw.is_multi_curve_mode = curve_count > 1

        if pw.is_multi_curve_mode:
            pw.plot_item.legend.setVisible(True)
        else:
            pw.plot_item.legend.setVisible(False)

    def update_legend(self):
        """更新图例"""
        pw = self.pw
        legend = pw.plot_item.legend
        legend.clear()

        if not pw.is_multi_curve_mode:
            legend.setVisible(False)
            return

        for var_name, ci in pw.curves.items():
            if ci.curve is not None and ci.visible:
                legend.addItem(ci.curve, var_name)

    def toggle_curve_visibility_by_name(self, var_name: str):
        """通过变量名切换曲线可见性"""
        pw = self.pw

        if var_name not in pw.curves:
            return

        ci = pw.curves[var_name]
        ci.visible = not ci.visible

        if ci.curve is not None:
            ci.curve.setVisible(ci.visible)

        if not pw.is_multi_curve_mode:
            return

        visible_count = sum(1 for c in pw.curves.values() if c.visible)
        if visible_count <= 1:
            pw.is_multi_curve_mode = False
            legend = pw.plot_item.legend
            legend.setVisible(False)
            self.update_multi_curve_mode()

    def _recreate_curve(self, var_name: str):
        """重新创建曲线"""
        pw = self.pw

        if var_name not in pw.curves:
            return

        ci = pw.curves[var_name]
        if ci.curve is not None:
            try:
                ci.curve.scene().removeItem(ci.curve)
            except Exception:
                pass

        import pyqtgraph as pg

        pen = pg.mkPen(color=ci.color, width=DEFAULT_LINE_WIDTH)
        curve = pw.plot_item.plot(
            ci.x_data, ci.y_data, pen=pen, name=var_name, skipFiniteCheck=True
        )

        pw.curves[var_name] = CurveInfo(
            var_name=var_name,
            curve=curve,
            x_data=ci.x_data,
            y_data=ci.y_data,
            color=ci.color,
            y_format=ci.y_format,
            visible=ci.visible,
        )

    def _collect_visible_curve_arrays(self, key: str) -> list:
        """收集可见曲线的数据数组"""
        pw = self.pw
        result = []
        for ci in pw.curves.values():
            if ci.visible and getattr(ci, key, None) is not None:
                data = getattr(ci, key)
                if data is not None:
                    result.append(data)
        return result

    def _collect_visible_curve_pairs(self) -> list:
        """收集可见曲线的 x-y 数据对"""
        pw = self.pw
        result = []
        for ci in pw.curves.values():
            if ci.visible and ci.x_data is not None and ci.y_data is not None:
                result.append((ci.x_data, ci.y_data))
        return result

    def get_curve_x_limits(self, curves_filter: str = "visible") -> tuple:
        """获取曲线 X 轴限制"""
        pw = self.pw

        if curves_filter == "visible":
            arrays = self._collect_visible_curve_arrays("x_data")
        else:
            arrays = [getattr(ci, "x_data") for ci in pw.curves.values()]

        if not arrays:
            return None, None

        all_values = []
        for arr in arrays:
            if arr is not None:
                all_values.extend(arr.tolist() if hasattr(arr, "tolist") else list(arr))

        if not all_values:
            return None, None

        return min(all_values), max(all_values)

    def _update_axes_for_multi_curve(self, update_x_range: bool = False):
        """更新多曲线的坐标轴"""
        pw = self.pw

        if not pw.is_multi_curve_mode:
            return

        pw.axis_x.setTicks(None)
        pw.axis_y.setTicks(None)

        if update_x_range:
            x_min, x_max = self.get_curve_x_limits()
            if x_min is not None and x_max is not None:
                pw.view_box.setXRange(x_min, x_max, padding=0.02)

        y_arrays = self._collect_visible_curve_arrays("y_data")
        if y_arrays:
            import numpy as np

            combined = np.concatenate(y_arrays)
            if combined.size:
                min_y = np.nanmin(combined)
                max_y = np.nanmax(combined)
                pw.view_box.setYRange(min_y, max_y, padding=DEFAULT_PADDING_VAL_Y)

    def _on_legend_clicked(self, event):
        """图例点击事件"""
        legend_item = event.currentItem

        if legend_item is None:
            return

        name = legend_item.text()
        self.toggle_curve_visibility_by_name(name)

    def _apply_plot_style(self, show_symbols: bool = False):
        """应用绘图样式"""
        pw = self.pw

        for ci in pw.curves.values():
            if ci.curve is not None:
                if show_symbols:
                    ci.curve.setSymbol("o")
                else:
                    ci.curve.setSymbol(None)
