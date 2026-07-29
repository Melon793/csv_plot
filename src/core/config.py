"""全局配置常量和工具函数

包含项目共用的配置常量、安全回调装饰器和数值安全检测函数：
- safe_callback: C++ 对象已销毁时的异常保护装饰器
- safe_qt_op: 安全执行 Qt 对象操作，忽略 C++ 对象已销毁的异常
- _evaluate_float32_safety: float32 安全表示范围检测
- 绘图/布局/加载相关的全局常量
"""

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


def safe_qt_op(func, *args, **kwargs):
    """安全执行 Qt 对象操作，忽略 C++ 对象已销毁的异常

    用于保护对 PySide6/pyqtgraph 对象的属性访问和方法调用，
    这些对象可能因 C++ 侧提前销毁而抛出 RuntimeError 或 AttributeError。

    Args:
        func: 要执行的可调用对象（方法引用、lambda 等）
        *args, **kwargs: 传给 func 的参数

    Returns:
        func 的返回值；异常时返回 None

    Example::

        safe_qt_op(item.setVisible, False)
        safe_qt_op(scene.removeItem, item)
        safe_qt_op(lambda: item.setText(""))
    """
    try:
        return func(*args, **kwargs)
    except (RuntimeError, AttributeError) as e:
        logger.debug(
            "safe_qt_op 捕获异常: %s → %s",
            getattr(func, "__name__", repr(func)), e,
        )

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


def compute_global_x_limits(
    loader, factor: float = 1.0, offset: float = 0.0
) -> tuple[float, float, float, float] | None:
    """计算全局统一的 X 轴数据范围和 limits。

    基于 loader 的全局数据范围（而非 per-plot 曲线数据），
    确保所有 Plot 的 X limits 一致，避免 X-link 同步时被 ViewBox 钳制。

    Args:
        loader: 数据加载器（CSV: datalength, MDF: global_time_range）
        factor: 时间修正系数
        offset: 时间修正偏移

    Returns:
        (min_x, max_x, limits_xMin, limits_xMax) 或 None（loader 无效时）
        - min_x/max_x: 应用 factor/offset 后的数据范围（用于 vline bounds / viewRange）
        - limits_xMin/limits_xMax: 含 5% padding 的边界（用于 ViewBox limits）
    """
    if loader is None:
        return None

    if getattr(loader, "LOADER_TYPE", "") == "mdf" and hasattr(loader, "global_time_range"):
        raw_min, raw_max = loader.global_time_range
    elif getattr(loader, "datalength", 0) > 0:
        raw_min, raw_max = 1.0, float(loader.datalength)
    else:
        return None

    min_x = offset + factor * float(raw_min)
    max_x = offset + factor * float(raw_max)

    # 安全范围：min == max 时扩展
    if min_x == max_x:
        min_x -= 0.5 * (factor if factor != 0 else 1.0)
        max_x += 0.5 * (factor if factor != 0 else 1.0)

    data_span = max_x - min_x
    limits_xMin = min_x - DEFAULT_PADDING_VAL_X * data_span
    limits_xMax = max_x + DEFAULT_PADDING_VAL_X * data_span

    return (min_x, max_x, limits_xMin, limits_xMax)


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
