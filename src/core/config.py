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
from functools import wraps

from src.core.logger import get_logger

logger = get_logger(__name__)

DEFAULT_SHOW_X_AXIS_LABEL = False


def safe_callback(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            err_msg = str(e).lower()
            if "deleted" in err_msg or "wrapped" in err_msg or "c++ object" in err_msg:
                logger.debug(
                    "safe_callback: C++ 对象已销毁 in %s", func.__name__
                )
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
    except RuntimeError as e:
        err_msg = str(e).lower()
        if "deleted" in err_msg or "c++ object" in err_msg:
            logger.debug(
                "safe_qt_op: C++ 对象已销毁 → %s",
                getattr(func, "__name__", repr(func)),
            )
            return None
        # 非 deletion 的 RuntimeError 仍向上传播
        raise
    except AttributeError as e:
        # 保留 AttributeError 捕获，但记录调用名以便排查拼写错误
        logger.debug(
            "safe_qt_op 捕获 AttributeError: %s → %s",
            getattr(func, "__name__", repr(func)), e,
            exc_info=True,
        )
        return None

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
# 变量信息窗口（变量列表右键"变量信息"）
VAR_INFO_MAX_TABS = 50  # 单次提交的最大标签页数，超出截断并提示
VAR_INFO_ENUM_DISPLAY_LIMIT = 200  # 枚举文本表在信息页最多展示条目数
VAR_INFO_STATS_CACHE_MAX = 256  # 统计缓存上限（单条约150字节，共约38KB）
# 信息页「属性」列的最小宽度：该列可手动拖动（Interactive），下限用于
# 防止用户拖到几乎为零后标签全部被省略号截断、又找不到拖回来的把手。
# 注意它实际传给 QHeaderView.setMinimumSectionSize()，约束的是**所有列**
# 而非只有第 0 列。默认自动定宽实测为 130 px（按最长属性名算出），
# 所以这个值只在用户主动往窄拖时生效；调到 60 是为了给长文件路径多腾
# 出「值」列空间
VAR_INFO_COL0_MIN_WIDTH = 60
# 信息页数据行行高。PySide6 6.11 的 QTreeView 没有 setRowHeight（Qt5 的
# 公开槽，Qt6 已移除），只能由委托 sizeHint 给高度；又因视图开着
# uniformRowHeights，全表必然同值，逐行给不同高度会被拉平成最大值。
# 下限受复制按钮约束：按钮垂直居中，且绘制不裁剪到单元格 —— 行高小于
# 按钮边长时图标会溢出骑到相邻行上（实测 40 px 按钮配 18 px 行高，上下
# 共溢出 22 px、横跨约 3 行），比被裁掉更糟：会遮挡别的内容。注意这个
# 约束是“行高→按钮”方向的，改大按钮不会把行高顶高；上限受一屏行数约束：
# 行高越大越早出现垂直滚动条，而滚动条会从「值」列吃掉 18 px（实测
# 540 → 522）。macOS 系统默认实测 18 px（offscreen 为 17）；设 0 沿用默认
VAR_INFO_ROW_HEIGHT = 28
# 信息页「值」列右端的悬停复制按钮：图标边长与内边距。边长 16 px 是
# 点击热区可用的下限（再小就用不顺手），margin 同时用于把文本区与图标
# 隔开，避免长文件路径的末尾字符贴着图标
VAR_INFO_COPY_BTN_SIZE = 16
VAR_INFO_COPY_BTN_MARGIN = 4
# MDF 后台统计的分块点数。v1.1 实测（tmp/bench_chunk_size_tuning.py，
# 慢转换通道 174k 点）：1<<19 时单块持锁最高 65 ms，UI 线程 get_series
# 等锁 max 达 131.7 ms；调小至 1<<15 后单块 ≤6 ms，且整条统计总耗时
# 持平略优（45.6 → 31.2 ms），4.1M 点长通道总耗时不变（42 → 40 ms）。
# 该常量仅被 var_info._stats_mdf 使用，绘图路径 get_series 不经过它。
MDF_STATS_CHUNK_SIZE = 1 << 15  # 32768
# 变量信息的「归属信息」块（设备 / ECU / 测量组 / 所属函数…）。
# MDF 标准里没有函数层级块，这些行全部由通道名 / 注释 / SI 块**推断**而
# 得，因此默认开启但保留一行关停开关：置 False 后归属块整体不输出，
# 而「通道注释 / 文件注释」的解析与乱码修复不受影响（那是纯缺陷修复）。
MDF_ATTRIBUTION_ENABLED = True
# latin-1 误读 GBK 文本的回转修复（CANape 以 GBK 写入中文注释，asammdf
# 按 cp1252/latin-1 解码 → “82ºÅµ¥ÌåµçÑ¹”）。见 mdf_attribution.repair_text
# 的四条保守条件，正常中文与纯 ASCII 不受影响；如出现误回转可置 False 回退。
MDF_GBK_TEXT_REPAIR = True
# 归属信息 / 注释行的展示截断长度。全值仍由信息页的 tooltip 承担，
# 不截断会把「值」列撑得很宽、挤掉统计行的可见性。
VAR_INFO_ATTRIBUTION_MAX_LEN = 120
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
