"""
PlotContext —— 绘图上下文服务层

为 DraggableGraphicsLayoutWidget / CustomViewBox 等绘图组件提供
对 MainWindow 服务的干净接口，替代 self.window() 直接访问。

MainWindow 在创建子图矩阵时注入此对象。
"""

from __future__ import annotations
from typing import Protocol, Any


class PlotServices(Protocol):
    """MainWindow 暴露给绘图组件的服务协议"""

    @property
    def loader(self) -> Any: ...
    @property
    def plot_widgets(self) -> list[Any]: ...
    @property
    def cursor_mode(self) -> str: ...
    @property
    def cursor_btn(self) -> Any: ...
    @property
    def cursor_values_hidden(self) -> bool: ...
    @property
    def pinned_x_values(self) -> list[float]: ...
    @pinned_x_values.setter
    def pinned_x_values(self, value: list[float]): ...
    @property
    def _plot_col_max_default(self) -> int: ...
    @property
    def plot_layout(self) -> Any: ...
    @property
    def _is_loading_new_data(self) -> bool: ...
    @property
    def _is_time_correction_active(self) -> bool: ...

    def sync_crosshair(self, x: float, source: Any) -> None: ...
    def request_mark_stats_refresh(self, immediate: bool = False) -> None: ...
    def toggle_cursor_all(self, enabled: bool) -> None: ...
    def get_row_height(self, row: int) -> int: ...
    def set_row_height(self, row: int, percentage: int) -> None: ...
    def set_all_row_height(self, percentage: int) -> None: ...
    def set_cursor_mode(self, mode: str, *, source_plot: Any | None = None, context_x: float | None = None) -> None: ...
    def sync_mark_regions(self, mark_region: Any) -> None: ...


class PlotContext:
    """绘图上下文 —— 将 MainWindow 服务暴露为属性"""

    def __init__(self, services: PlotServices):
        self._services = services

    @property
    def loader(self) -> Any:
        return self._services.loader

    @property
    def plot_widgets(self) -> list[Any]:
        return self._services.plot_widgets

    @property
    def cursor_mode(self) -> str:
        return self._services.cursor_mode

    @property
    def cursor_btn(self) -> Any:
        return self._services.cursor_btn

    @property
    def cursor_values_hidden(self) -> bool:
        return self._services.cursor_values_hidden

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
    def plot_layout(self) -> Any:
        return self._services.plot_layout

    def sync_crosshair(self, x: float, source: Any) -> None:
        self._services.sync_crosshair(x, source)

    def request_mark_stats_refresh(self, immediate: bool = False) -> None:
        self._services.request_mark_stats_refresh(immediate)

    def toggle_cursor_all(self, enabled: bool) -> None:
        self._services.toggle_cursor_all(enabled)

    def get_row_height(self, row: int) -> int:
        return self._services.get_row_height(row)

    def set_row_height(self, row: int, percentage: int) -> None:
        self._services.set_row_height(row, percentage)

    def set_all_row_height(self, percentage: int) -> None:
        self._services.set_all_row_height(percentage)

    def set_cursor_mode(self, mode: str, source_plot: Any | None = None, context_x: float | None = None) -> None:
        self._services.set_cursor_mode(mode, source_plot=source_plot, context_x=context_x)

    def sync_mark_regions(self, mark_region: Any) -> None:
        self._services.sync_mark_regions(mark_region)

    def is_cursor_enabled(self) -> bool:
        btn = self._services.cursor_btn
        return btn.isChecked() if btn else False

    def set_cursor_checked(self, checked: bool) -> None:
        btn = self._services.cursor_btn
        if btn:
            btn.setChecked(checked)
