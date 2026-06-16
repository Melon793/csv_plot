"""
PlotContext —— 绘图上下文服务层

为 DraggableGraphicsLayoutWidget / CustomViewBox 等绘图组件提供
对 MainWindow 服务的干净接口，替代 self.window() 直接访问。

MainWindow 在创建子图矩阵时注入此对象。
"""

from __future__ import annotations
from typing import Protocol, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from PySide6.QtWidgets import QPushButton, QGridLayout
    from src.ui.widgets.plot_widget import DraggableGraphicsLayoutWidget


class PlotServices(Protocol):
    """MainWindow 暴露给绘图组件的服务协议"""

    @property
    def loader(self) -> Any: ...
    @property
    def plot_widgets(self) -> list[DraggableGraphicsLayoutWidget]: ...
    @property
    def cursor_mode(self) -> str: ...
    @property
    def cursor_btn(self) -> QPushButton: ...
    @property
    def cursor_values_hidden(self) -> bool: ...
    @cursor_values_hidden.setter
    def cursor_values_hidden(self, value: bool): ...
    @property
    def pinned_x_values(self) -> list[float]: ...
    @pinned_x_values.setter
    def pinned_x_values(self, value: list[float]): ...
    @property
    def _plot_col_max_default(self) -> int: ...
    @property
    def plot_layout(self) -> QGridLayout: ...
    @property
    def _is_loading_new_data(self) -> bool: ...
    @property
    def _is_time_correction_active(self) -> bool: ...
    @property
    def _global_max_density(self) -> float: ...
    @property
    def value_cache(self) -> dict[str, Any]: ...
    @property
    def _enum_text_maps(self) -> dict[str, dict[int, str]]: ...

    def sync_crosshair(self, x: float, source: Any) -> None: ...
    def request_mark_stats_refresh(self, *, immediate: bool = False) -> None: ...
    def toggle_cursor_all(self, enabled: bool) -> None: ...
    def get_row_height(self, row: int) -> int: ...
    def set_row_height(self, row: int, percentage: int) -> None: ...
    def set_all_row_height(self, percentage: int) -> None: ...
    def set_cursor_mode(
        self,
        mode: str,
        *,
        source_plot: Any | None = None,
        context_x: float | None = None,
    ) -> None: ...
    def sync_mark_regions(self, mark_region: Any) -> None: ...
    def _sync_min_xrange(self) -> None: ...
    def _get_plot_container(self, plot_widget: Any) -> Any: ...
    def _show_drag_indicator_for_plot(
        self, plot_widget: Any, var_names: list[str], text_override: str | None = None
    ) -> None: ...
    def _hide_drag_indicator_for_plot(self, plot_widget: Any) -> None: ...
    def auto_y_in_x_range(self) -> None: ...
    def collect_global_x_range(
        self, curves_filter: str = "visible"
    ) -> tuple[float | None, float | None]: ...
    def set_cursor_enabled(self, enabled: bool) -> None: ...
    def is_cursor_enabled(self) -> bool: ...


class PlotContext:
    """绘图上下文 —— 将 MainWindow 服务暴露为属性"""

    def __init__(self, services: PlotServices):
        self._services = services

    @property
    def loader(self) -> Any:
        return self._services.loader

    @property
    def plot_widgets(self) -> list[DraggableGraphicsLayoutWidget]:
        return self._services.plot_widgets

    @property
    def cursor_mode(self) -> str:
        return self._services.cursor_mode

    @property
    def cursor_btn(self) -> QPushButton:
        return self._services.cursor_btn

    @property
    def cursor_values_hidden(self) -> bool:
        return self._services.cursor_values_hidden

    @cursor_values_hidden.setter
    def cursor_values_hidden(self, value: bool):
        self._services.cursor_values_hidden = value

    @property
    def pinned_x_values(self) -> list[float]:
        return self._services.pinned_x_values

    @pinned_x_values.setter
    def pinned_x_values(self, value: list[float]):
        self._services.pinned_x_values = value

    @property
    def _plot_col_max_default(self) -> int:
        return self._services._plot_col_max_default

    @property
    def plot_layout(self) -> QGridLayout:
        return self._services.plot_layout

    @property
    def _is_loading_new_data(self) -> bool:
        return self._services._is_loading_new_data

    @property
    def _data_version(self) -> int:
        return self._services._data_version

    @property
    def _is_time_correction_active(self) -> bool:
        return self._services._is_time_correction_active

    @property
    def _global_max_density(self) -> float:
        return self._services._global_max_density

    @property
    def value_cache(self) -> dict[str, Any]:
        return self._services.value_cache

    @property
    def _enum_text_maps(self) -> dict[str, dict[int, str]]:
        return self._services._enum_text_maps

    def sync_crosshair(self, x: float, source: Any) -> None:
        self._services.sync_crosshair(x, source)

    def request_mark_stats_refresh(self, *, immediate: bool = False) -> None:
        self._services.request_mark_stats_refresh(immediate=immediate)

    def toggle_cursor_all(self, enabled: bool) -> None:
        self._services.toggle_cursor_all(enabled)

    def get_row_height(self, row: int) -> int:
        return self._services.get_row_height(row)

    def set_row_height(self, row: int, percentage: int) -> None:
        self._services.set_row_height(row, percentage)

    def set_all_row_height(self, percentage: int) -> None:
        self._services.set_all_row_height(percentage)

    def set_cursor_mode(
        self, mode: str, source_plot: Any | None = None, context_x: float | None = None
    ) -> None:
        self._services.set_cursor_mode(
            mode, source_plot=source_plot, context_x=context_x
        )

    def sync_mark_regions(self, mark_region: Any) -> None:
        self._services.sync_mark_regions(mark_region)

    def _sync_min_xrange(self) -> None:
        self._services._sync_min_xrange()

    def _get_plot_container(self, plot_widget: Any) -> Any:
        return self._services._get_plot_container(plot_widget)

    def _show_drag_indicator_for_plot(
        self, plot_widget: Any, var_names: list[str], text_override: str | None = None
    ) -> None:
        self._services._show_drag_indicator_for_plot(
            plot_widget, var_names, text_override
        )

    def _hide_drag_indicator_for_plot(self, plot_widget: Any) -> None:
        self._services._hide_drag_indicator_for_plot(plot_widget)

    def auto_y_in_x_range(self) -> None:
        self._services.auto_y_in_x_range()

    def collect_global_x_range(
        self, curves_filter: str = "visible"
    ) -> tuple[float | None, float | None]:
        return self._services.collect_global_x_range(curves_filter)

    def set_cursor_enabled(self, enabled: bool) -> None:
        self._services.set_cursor_enabled(enabled)

    def is_cursor_enabled(self) -> bool:
        return self._services.is_cursor_enabled()

    def set_cursor_checked(self, checked: bool) -> None:
        btn = self._services.cursor_btn
        if btn:
            btn.setChecked(checked)
