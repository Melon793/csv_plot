"""时间戳 X 轴的索引 → 时间文本映射（纯数据层，无 Qt 依赖）

背景：CSV/Excel 路径下全部子图、游标、标注、跳转数据表都工作在**索引空间**
（显示 X = offset + factor × 行号，1-based）。把曲线 X 数据真换成时间戳会波及
游标同步 / 标注坐标 / offset-factor 时间校正 —— 违背"不影响原功能"。

因此时间戳 X 轴做成**轴渲染层映射**：数据空间不动，只把"索引刻度"翻译成
对应行的时间戳文本。本模块就是那台翻译机：

    显示 X --(pw.factor/offset)--> 行号(1-based) --(本模块)--> "13:03:45"

格式约定（与加载器的时间列识别口径一致，见 src/data/loader.py 的
date_candidates）：

- 纯时刻列（fmt 以 %H:%M:%S 开头，pandas 补"今天"日期）→ 只显示 ``HH:MM:SS``
- 完整日期列且所有行同一天 → ``HH:MM:SS``
- 完整日期列跨天 → ``MM-dd HH:MM:SS``
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TICK_FORMAT_TIME_ONLY = "%H:%M:%S"
TICK_FORMAT_WITH_DATE = "%m-%d %H:%M:%S"

_NS_PER_SECOND = 1e9


def build_timestamps(series: pd.Series, date_format: str | None = None) -> np.ndarray:
    """时间列（datetime64 或字符串）→ Unix 秒（float64），无效值 → NaN

    生产路径 CSV 的时间列保持**字符串** dtype（FastDataLoader.do_parse_date
    默认 False，只有 date_formats 记录了格式），所以这里统一走
    ``pd.to_datetime``：datetime64 输入是无操作；字符串输入按 date_format
    解析（纯时刻格式如 ``%H:%M:%S`` 由 pandas 补"今天"日期，与
    plot_data_manager 的既有口径一致）。
    """
    values = pd.to_datetime(series, format=date_format, errors="coerce")
    raw = values.to_numpy(dtype="datetime64[ns]").view("int64")
    seconds = raw.astype(np.float64) / _NS_PER_SECOND
    # NaT 的 int64 是最小哨兵（约 -9.2e18），洗成 NaN
    seconds[raw < -(2**62)] = np.nan
    return seconds


def _is_pure_time_format(fmt: str | None) -> bool:
    """纯时刻格式（%H:%M:%S 开头）：pandas 解析时补了"今天"，日期部分无意义"""
    return bool(fmt) and str(fmt).startswith("%H:%M:%S")


def _detect_tick_format(
    timestamps: np.ndarray, fmt: str | None
) -> str:
    """按数据形态决定刻度格式（见模块 docstring 的三条约定）"""
    if _is_pure_time_format(fmt):
        return TICK_FORMAT_TIME_ONLY
    finite = timestamps[np.isfinite(timestamps)]
    if finite.size == 0:
        return TICK_FORMAT_TIME_ONLY
    days = (finite // 86400).astype(np.int64)
    if days.max() == days.min():
        return TICK_FORMAT_TIME_ONLY
    return TICK_FORMAT_WITH_DATE


class TimeAxisMapping:
    """行号（1-based，可为小数）→ 时间戳文本

    小数行号按相邻两行的时间线性插值：缩放后刻度落在行间时，
    与真实采样时刻的偏差不超过一个采样间隔。
    """

    def __init__(self, timestamps: np.ndarray, date_format: str | None = None):
        self._timestamps = np.asarray(timestamps, dtype=np.float64)
        self._tick_format = _detect_tick_format(self._timestamps, date_format)

    @property
    def tick_format(self) -> str:
        return self._tick_format

    @property
    def size(self) -> int:
        return int(self._timestamps.size)

    def text_for_index(self, index_1based: float) -> str | None:
        """行号 → 刻度文本；越界 / 无数据 / 非有限值返回 None（调用方回退数字）"""
        if self.size == 0:
            return None
        if not np.isfinite(index_1based):
            return None
        # 允许半行容差：视图范围比数据略宽时（autorange padding），边界刻度
        # 仍能取到最近一行的时刻；再往外就是无数据区，回退留空
        if index_1based < 0.5 or index_1based > self.size + 0.5:
            return None
        ts = self._timestamps
        value = float(np.interp(index_1based - 1.0, np.arange(ts.size), ts))
        if not np.isfinite(value):
            return None
        return pd.Timestamp(value, unit="s").strftime(self._tick_format)
