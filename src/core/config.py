"""config"""

from __future__ import annotations
import sys
import os
from typing import Any
import logging

from src.core.logger import get_logger

logger = get_logger(__name__)

DEFAULT_SHOW_X_AXIS_LABEL = False


def safe_callback(func):
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            err_msg = str(e).lower()
            if "deleted" in err_msg or "wrapped" in err_msg or "c++ object" in err_msg:
                return None
            raise
        except Exception:
            logger.exception("safe_callback suppressed exception in %s", func.__name__)
            return None

    return wrapper

DEFAULT_PADDING_VAL_X = 0.05  # 默认x轴padding，单位为plot宽度
DEFAULT_PADDING_VAL_Y = 0.1  # 默认y轴padding，单位为plot高度
FILE_SIZE_LIMIT_BACKGROUND_LOADING = (
    2  # 2MB：区分平均值文件(<100点)和连续测量文件(~10000点)
)
RATIO_RESET_PLOTS = 0.3  # 重置plot比例，超过此比例时，重置plot
FROZEN_VIEW_WIDTH_DEFAULT = 180  # 冻结视图宽度，默认值为180px
XRANGE_THRESHOLD_FOR_SYMBOLS = 100.0  # xRange宽度阈值（考虑factor后），小于此值显示symbols（细线+symbol），否则粗线无symbol
BLINK_PULSE = 200
FACTOR_SCROLL_ZOOM = 0.3
MIN_INDEX_LENGTH = 3  # 每个plot，至少显示MIN_INDEX_LENGTH个点
DEFAULT_LINE_WIDTH = 2  # 默认线宽
THICK_LINE_WIDTH = 2  # 粗线宽
THIN_LINE_WIDTH = 1  # 细线宽
UI_DEBOUNCE_DELAY_MS = 50  # UI事件防抖延迟时间
# 默认绘图布局配置
PLOT_ROW_MAX_DEFAULT = 4
PLOT_COL_MAX_DEFAULT = 3
PLOT_ROW_CURRENT_DEFAULT = 3
PLOT_COL_CURRENT_DEFAULT = 1

FLOAT32_REPRESENTABLE_MAX = 3.4028234663852886e+38

# 单位行自动检测阈值
UNIT_KEYWORD_RATIO_THRESHOLD = 0.2  # 单位关键字列比例超过此值，判定为单位行
VALID_NUMERIC_RATIO_THRESHOLD = 0.6  # 有效数值列比例超过此值，判定为数据行

# Excel 自动检测：最大扫描行数
EXCEL_MAX_SCAN_ROWS = 30

# 单位关键字列表（子字符串匹配，用于自动检测标题行下方的单位行）
_UNIT_KEYWORDS = [
    "m",
    "s",
    "g",
    "A",
    "K",
    "mol",
    "cd",
    "V",
    "Ω",
    "F",
    "H",
    "W",
    "J",
    "N",
    "Nm",
    "Pa",
    "bar",
    "m2",
    "/min",
    "/h",
    "kWh",
    "mm",
    "°CA",
    "L",
    "m3",
    "ppm",
    "ppb",
    "%",
    "rpm",
    "℃",
    "°F",
    "°C",
    "#/",
    "-",
]


def _evaluate_float32_safety(values: Any) -> tuple[bool, float | None]:
    """
    判断数值是否能安全表示为 float32。

    参数:
        values: pandas Series、NumPy 数组或其它可迭代的数值序列。

    返回:
        tuple[bool, float | None]: (是否安全、绝对值最大值)
            当数据中不存在有限值时，绝对值最大值为 None。
    """
    if values is None:
        return False, None

    import numpy as np
    import pandas as pd

    try:
        if isinstance(values, pd.Series):
            arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
        else:
            try:
                arr = np.asarray(values, dtype=np.float64)
            except (ValueError, TypeError, OverflowError):
                arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(
                    dtype=np.float64
                )
    except Exception:
        return False, None

    if arr.size == 0:
        return True, 0.0

    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return False, None

    abs_max = float(np.max(np.abs(arr[finite_mask])))
    return abs_max <= FLOAT32_REPRESENTABLE_MAX, abs_max
