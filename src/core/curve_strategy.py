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


class SingleCurveStrategy(CurveStrategy):
    def __init__(self, plot_widget):
        self._pw = plot_widget

    def has_data(self) -> bool:
        return self._pw.curve is not None and getattr(self._pw, "y_name", None) is not None

    def get_curve_names(self) -> list[str]:
        if not self.has_data():
            return []
        return [self._pw.y_name]

    def get_y_arrays(self) -> list[np.ndarray]:
        if not self.has_data():
            return []
        return [self._pw.curve.y_data]

    def get_x_data(self) -> np.ndarray:
        if self._pw.curve is None:
            return np.array([])
        return self._pw.curve.x_data

    def get_curve_by_name(self, name: str) -> Any:
        if name == getattr(self._pw, "y_name", None):
            return self._pw.curve
        return None

    def get_curve_color(self, name: str) -> Any:
        curve = self.get_curve_by_name(name)
        if curve is not None:
            pen = curve.opts.get("pen", None)
            return pen.color() if pen else None
        return None

    def get_curve_line_width(self, name: str) -> int:
        curve = self.get_curve_by_name(name)
        if curve is not None:
            pen = curve.opts.get("pen", None)
            return pen.width() if pen else 1
        return 1


class MultiCurveStrategy(CurveStrategy):
    def __init__(self, plot_widget):
        self._pw = plot_widget

    def has_data(self) -> bool:
        return bool(self._pw.curves)

    def get_curve_names(self) -> list[str]:
        return list(self._pw.curves.keys()) if self._pw.curves else []

    def get_y_arrays(self) -> list[np.ndarray]:
        if not self._pw.curves:
            return []
        return [c.y_data for c in self._pw.curves.values()]

    def get_x_data(self) -> np.ndarray:
        names = self.get_curve_names()
        if not names:
            return np.array([])
        return self._pw.curves[names[0]].x_data

    def get_curve_by_name(self, name: str) -> Any:
        info = self._pw.curves.get(name)
        return info.curve_item if info else None

    def get_curve_color(self, name: str) -> Any:
        info = self._pw.curves.get(name)
        return info.color if info else None

    def get_curve_line_width(self, name: str) -> int:
        info = self._pw.curves.get(name)
        return info.line_width if info else 1
