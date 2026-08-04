"""曲线数据访问策略（统一版）

单线/多线模式统一后，所有曲线均存储在 pw.curves 字典中，
不再区分 SingleCurveStrategy / MultiCurveStrategy。
"""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class CurveStrategy(ABC):
    @abstractmethod
    def get_curve_names(self) -> list[str]: ...

    @abstractmethod
    def get_y_arrays(self) -> list[np.ndarray]: ...

    @abstractmethod
    def get_x_data(self) -> np.ndarray: ...

    @abstractmethod
    def has_data(self) -> bool: ...

    @abstractmethod
    def get_curve_by_name(self, name: str) -> Any: ...

    @abstractmethod
    def get_curve_color(self, name: str) -> Any: ...

    @abstractmethod
    def get_curve_line_width(self, name: str) -> int: ...


class UnifiedCurveStrategy(CurveStrategy):
    """统一曲线策略 — 始终从 curves 字典获取数据"""

    def __init__(self, plot_widget):
        self._pw = plot_widget

    def has_data(self) -> bool:
        return bool(self._pw.curves)

    def get_curve_names(self) -> list[str]:
        return list(self._pw.curves.keys()) if self._pw.curves else []

    def get_y_arrays(self) -> list[np.ndarray]:
        if not self._pw.curves:
            return []
        return [c.y_data for c in self._pw.curves.values() if c.y_data is not None]

    def get_x_data(self) -> np.ndarray:
        # 前提假设：同一 plot 内所有曲线共享同一 x_data（同源数据/同一时间轴）。
        # 当前 MDF 多 group 场景下每个 plot 仅绘制单一 group 的数据，该假设成立；
        # 若未来引入跨 group（不同时间轴）曲线混绘，必须重新评估本方法。
        if not self._pw.curves:
            return np.array([])
        first_ci = next(iter(self._pw.curves.values()))
        return first_ci.x_data if first_ci.x_data is not None else np.array([])

    def get_curve_by_name(self, name: str) -> Any:
        info = self._pw.curves.get(name)
        return info.curve if info else None

    def get_curve_color(self, name: str) -> Any:
        info = self._pw.curves.get(name)
        return info.color if info else None

    def get_curve_line_width(self, name: str) -> int:
        info = self._pw.curves.get(name)
        if info and info.curve:
            pen = info.curve.opts.get("pen", None)
            return pen.width() if pen else 1
        return 1
