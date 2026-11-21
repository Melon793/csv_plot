"""曲线数据访问策略（统一版）

单线/多线模式统一后，所有曲线均存储在 pw.curves 字典中，
不再区分 SingleCurveStrategy / MultiCurveStrategy。
"""

from abc import ABC, abstractmethod
from typing import Any
import numpy as np

from src.core.data_types import (
    DRAW_MODE_SOURCE_AUTO,
    DRAW_MODE_SOURCE_USER,
    LINE_PRESET_AUTO,
    SYMBOL_POLICY_AUTO,
)


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

    @abstractmethod
    def get_curve_styles(self, name: str) -> dict: ...


class UnifiedCurveStrategy(CurveStrategy):
    """统一曲线策略 — 始终从 curves 字典获取数据"""

    def __init__(self, plot_widget):
        self._pw = plot_widget

    def has_data(self) -> bool:
        return bool(self._pw.curves)

    def get_curve_names(self) -> list[str]:
        return list(self._pw.curves.keys()) if self._pw.curves else []

    def get_invalid_curve_names(self) -> list[str]:
        """失效占位变量名（方案 B2，§8.2）：与 get_curve_names 并列，供导出写入。

        占位不在 curves 中，故独立取 invalid_curves；替身/旧 widget 无该属性时
        容错为空列表。
        """
        invalid = getattr(self._pw, "invalid_curves", None)
        return list(invalid.keys()) if invalid else []

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

    def get_curve_styles(self, name: str) -> dict:
        """曲线外观束（模板/auto-save 捕获端）：color 恒写，其余仅偏离默认值。

        draw_mode 仅 user 钉住才写（auto 来源让套用端按新数据元数据重新
        判定，与 reload saved_state 同口径）；变量不存在 → {}。
        """
        info = self._pw.curves.get(name)
        if info is None:
            return {}
        styles: dict = {"color": info.color}
        preset = getattr(info, "line_preset", LINE_PRESET_AUTO)
        if preset != LINE_PRESET_AUTO:
            styles["line_preset"] = preset
        if getattr(info, "draw_mode_source", DRAW_MODE_SOURCE_AUTO) == DRAW_MODE_SOURCE_USER:
            styles["draw_mode"] = info.draw_mode
        policy = getattr(info, "symbol_policy", SYMBOL_POLICY_AUTO)
        if policy != SYMBOL_POLICY_AUTO:
            styles["symbol_policy"] = policy
        return styles
