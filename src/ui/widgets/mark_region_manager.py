"""
MarkRegionManager - 标记区域管理

负责 DraggableGraphicsLayoutWidget 的标记区域功能：
- 添加/移除标记区域
- 更新标记区域位置
- 计算标记区域内的数据统计

此模块从 csv_plot_pyqt6.py 迁移而来。
"""

from __future__ import annotations
from typing import Any, TYPE_CHECKING

import numpy as np
import pyqtgraph as pg

from src.core.config import _evaluate_float32_safety
from src.core.data_types import MarkStatEntry

if TYPE_CHECKING:
    from src.ui.widgets.cursor_manager import CursorManager


def _nearest_index(x_data: np.ndarray, value: float) -> int:
    """升序 `x_data` 上离 `value` 最近的下标，与 `argmin(abs(x - value))` 严格等价。

    平手取左侧、重复 x 回退到该值首次出现处，都是在复刻 `argmin` 的「首个最小值」
    语义 —— 同一重复段里 `y_data` 各不相同，下标差了就等于换了个采样点算 slope。
    """
    idx = int(np.searchsorted(x_data, value, side="left"))
    if idx <= 0:
        return 0
    if idx < len(x_data) and (value - x_data[idx - 1]) > (x_data[idx] - value):
        return idx
    best = idx - 1
    return int(np.searchsorted(x_data, x_data[best], side="left"))


class MarkRegionManager:
    """负责标记区域的管理和统计计算"""

    def __init__(self, cursor_manager: CursorManager):
        if cursor_manager is None:
            raise ValueError(
                "MarkRegionManager requires a valid CursorManager instance"
            )
        self._cursor_manager = cursor_manager

    @property
    def pw(self) -> Any:
        cm = self._cursor_manager
        if cm is None:
            raise RuntimeError(
                "MarkRegionManager: dependency chain broken (_cursor_manager is None)"
            )
        return cm.pw

    def add_mark_region(self, min_x: float, max_x: float):
        """添加标记区域（已存在时先拆干净，避免留下拖不动也关不掉的幽灵区域）"""
        self._teardown_mark_region()

        self.pw.mark_region = pg.LinearRegionItem([min_x, max_x], movable=True)
        for line in self.pw.mark_region.lines:
            line.setHoverPen(pg.mkPen(color="r", width=10))

        self.pw.plot_item.addItem(self.pw.mark_region)
        # 槽内按 pw.page 反查所属页（§6.4 #6）：连接/断开都指向同一个成员方法，
        # 不用 lambda（每次都是新对象，无法可靠 disconnect）
        self.pw.mark_region.sigRegionChanged.connect(self._on_region_changed)
        # 拖动结束信号：松手立即精确刷新统计。本环境 pyqtgraph 0.14 的
        # LinearRegionItem 没有 sigRegionChangeStarted，拖动中的节流刷新
        # 由 sync_mark_regions → request_mark_stats_refresh(allow_throttled)
        # 承担（首个事件立即刷，其后 150ms 节流）——否则纯 trailing 防抖
        # 在 60Hz 事件下（间隔 < 50ms 防抖窗）拖动全程冻结统计表。
        self.pw.mark_region.sigRegionChangeFinished.connect(self._on_region_change_finished)

    def _on_region_changed(self, region):
        """任一页的区间被拖动：交给 LayoutManager 广播到全部页。"""
        page = getattr(self.pw, "page", None)
        window = self.pw.window()
        if window is None or not hasattr(window, "layout_manager"):
            return
        window.layout_manager.sync_mark_regions(region, page=page)

    def _on_region_change_finished(self, region):
        """拖动结束：绕过节流立即精确刷新统计（落定值）。"""
        window = self.pw.window()
        if window is None or not hasattr(window, "layout_manager"):
            return
        window.layout_manager.request_mark_stats_refresh(immediate=True)

    def _teardown_mark_region(self):
        """断开信号 → 摘出场景 → 销毁 C++ 对象。

        连接挂在 region item（发送端）上，只 `removeItem` + 置 None 时一旦还有
        引用（例如紧接着又 add 了一次），旧区域会继续留在图里响应拖拽并回调
        `sync_mark_regions` 去改写新区域。
        """
        region = self.pw.mark_region
        if region is None:
            return
        for sig, slot in (
            (region.sigRegionChanged, self._on_region_changed),
            (region.sigRegionChangeFinished, self._on_region_change_finished),
        ):
            try:
                sig.disconnect(slot)
            except (RuntimeError, TypeError):
                # 构造期尚无连接 / C++ 侧已销毁
                pass
        if region.scene() is not None:
            self.pw.plot_item.removeItem(region)
        region.deleteLater()
        self.pw.mark_region = None

    def remove_mark_region(self):
        """移除标记区域"""
        self._teardown_mark_region()

    def get_mark_stats(self, region: tuple[float, float] | None = None) -> list | None:
        """获取标记区域的统计信息（统一版：始终走 curves 字典路径）

        Args:
            region: 显式区间；隐藏页尚未补建 region item 时由调用方传入全局
                ``saved_mark_range``，保证统计窗口覆盖所有页（§6.4 #9）。

        【NumPy优化】使用NumPy掩码数组批量计算统计值，避免循环过滤
        """
        if region is None:
            if not self.pw.mark_region:
                return None
            min_x, max_x = self.pw.mark_region.getRegion()
        else:
            min_x, max_x = region
        return self._get_mark_stats_multi_curve(min_x, max_x)

    def _get_mark_stats_multi_curve(self, min_x: float, max_x: float) -> list | None:
        """多曲线模式的统计计算"""
        stats_list = []

        for var_name, ci in self.pw.curves.items():
            if not ci.visible:
                continue

            if ci.curve is None:
                continue

            if ci.x_data is not None and ci.y_data is not None:
                x_data = ci.x_data
                y_data = ci.y_data
            elif ci.y_data is not None:
                x_data = self.pw.offset + self.pw.factor * np.arange(
                    1, len(ci.y_data) + 1, dtype=np.float32
                )
                y_data = ci.y_data
            else:
                curve = ci.curve
                x_data, y_data = curve.getData()
                if x_data is None or len(x_data) == 0:
                    continue

            x_data = np.asarray(x_data)
            y_data = np.asarray(y_data)
            if x_data.dtype.kind in "iu":
                safe_x, _ = _evaluate_float32_safety(x_data)
                x_dtype = np.float32 if safe_x else np.float64
                x_data = x_data.astype(x_dtype)
            if y_data.dtype.kind in "iu":
                safe_y, _ = _evaluate_float32_safety(y_data)
                y_dtype = np.float32 if safe_y else np.float64
                y_data = y_data.astype(y_dtype)

            idx_left = _nearest_index(x_data, min_x)
            idx_right = _nearest_index(x_data, max_x)
            x1 = x_data[idx_left]
            y1 = y_data[idx_left]
            x2 = x_data[idx_right]
            y2 = y_data[idx_right]
            dx = x2 - x1
            dy = y2 - y1
            slope = float("inf") if dx == 0 else dy / dx

            mask = (x_data >= min_x) & (x_data <= max_x)
            # 新增指标默认值：区间无有效点时保持 NaN / 0（与 y_avg 语义一致）
            x_at_y_max = x_at_y_min = y_pp = np.nan
            count = 0
            if np.any(mask):
                y_masked = y_data[mask]
                valid = ~np.isnan(y_masked)
                valid_y = y_masked[valid]
                if len(valid_y) > 0:
                    y_avg = np.mean(valid_y)
                    y_max = np.max(valid_y)
                    y_min = np.min(valid_y)
                    # x 与 valid_y 对齐：valid_y 已滤 NaN，x 必须套同一掩码。
                    # 注意：不能用 np.nanargmin —— 它内部转调 np.argmin
                    # （NaN→inf 后求 argmin），会被统计链路的 argmin
                    # 调用禁令测试捕获；最小值位置用 nanargmax(-y) 等价
                    # 实现（首处最小值语义与 nanargmin 完全一致）。
                    x_valid = x_data[mask][valid]
                    x_at_y_max = float(x_valid[np.nanargmax(valid_y)])
                    x_at_y_min = float(x_valid[np.nanargmax(-valid_y)])
                    y_pp = float(y_max - y_min)
                    count = int(len(valid_y))
                else:
                    y_avg = y_max = y_min = np.nan
            else:
                y_avg = y_max = y_min = np.nan

            unit = self.pw.units.get(var_name, "")
            label = f"{var_name} ({unit})" if unit else var_name

            stats_list.append(
                MarkStatEntry(
                    x1=x1,
                    x2=x2,
                    y1=y1,
                    y2=y2,
                    dx=dx,
                    dy=dy,
                    slope=slope,
                    label=label,
                    y_avg=y_avg,
                    y_max=y_max,
                    y_min=y_min,
                    var_name=var_name,
                    unit=unit,
                    x_at_y_max=x_at_y_max,
                    x_at_y_min=x_at_y_min,
                    y_pp=y_pp,
                    count=count,
                )
            )

        return stats_list if stats_list else None
