"""数据结构和类型定义

包含项目共用的数据类、异常类型和枚举：
- FormatInfo: 文件格式自动检测结果
- CurveInfo: 单条曲线元数据
- MarkStatEntry: 标记区域统计条目
- AutoDetectError: 格式检测失败异常
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pyqtgraph as pg


class AutoDetectError(Exception):
    """自动检测文件格式失败，需要用户手动指定分隔符/标题行/单位行"""

    pass


@dataclass
class FormatInfo:
    """文件格式自动检测结果

    包含一次文件扫描得到的全部格式信息：编码、分隔符、标题行位置、是否包含单位行。
    供 auto_detect() 统一入口返回，也供未来的 ImportDialog 消费。
    """

    encoding: str | None  # 检测到的编码名称，None 表示检测失败
    sep: str | None  # 检测到的分隔符，None 表示检测失败
    header_row: int  # 标题行 0-based 行号
    has_unit: bool  # 是否包含单位行
    sheet_name: str | None = None  # 选中的 Sheet 名（仅 Excel 使用）


@dataclass
class CurveInfo:
    """单条曲线的元数据（统一版）

    统一单线/多线模式后，所有曲线（含首条）均以 CurveInfo 存储在 pw.curves 字典中。
    """

    var_name: str
    curve: "pg.PlotDataItem"
    x_data: np.ndarray
    y_data: np.ndarray
    original_index: np.ndarray = None  # 原始索引/时间戳（用于时间修正重算，消除反算精度问题）
    color: str = "blue"
    y_format: str = ""
    visible: bool = True
    x_min: float = 0.0
    x_max: float = 0.0
    point_density: float = 0.0

    def __post_init__(self):
        """自动从 x_data 计算缓存的 x_min / x_max 和 point_density"""
        self._refresh_x_cache()

    def _refresh_x_cache(self):
        """刷新 x_min/x_max/point_density 缓存"""
        import numpy as np
        if self.x_data is not None and len(self.x_data) > 1:
            self.x_min = float(np.min(self.x_data))
            self.x_max = float(np.max(self.x_data))
            span = self.x_max - self.x_min
            self.point_density = len(self.x_data) / span if span > 0 else 0.0
        elif self.x_data is not None and len(self.x_data) == 1:
            self.x_min = float(self.x_data[0])
            self.x_max = float(self.x_data[0])
            self.point_density = 0.0

    def update_x_range(self):
        """当 x_data 变更后调用，同步更新缓存"""
        self._refresh_x_cache()


@dataclass
class MarkStatEntry:
    """标记区域统计条目"""
    x1: float
    x2: float
    y1: float
    y2: float
    dx: float
    dy: float
    slope: float
    label: str
    y_avg: float
    y_max: float
    y_min: float
