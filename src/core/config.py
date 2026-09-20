"""全局配置常量和工具函数

包含项目共用的配置常量、安全回调装饰器和数值安全检测函数：
- widget_alive: 延迟回调摸控件前的 Qt C++ 对象存活判定
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

from shiboken6 import isValid as _shiboken_is_valid

from src.core.logger import get_logger

logger = get_logger(__name__)

DEFAULT_SHOW_X_AXIS_LABEL = False


def widget_alive(widget) -> bool:
    """Qt C++ 对象是否还在——延迟回调摸控件前的必要前置判断。

    ``QTimer.singleShot`` 持有的是普通 Python 方法/lambda，窗口或 tab 销毁不会
    取消它；届时残留的控件只剩 Python 包装器，按属性名取值照常（命中的是
    ``__dict__``），直到真调到 ``isVisible()`` / ``geometry()`` 才抛
    ``RuntimeError: Internal C++ object already deleted``。所以 ``not widget``
    和 ``hasattr(...)`` 这类弱守卫挡不住，必须先问一句。

    ``isValid`` 对非 Qt 对象（component 测试里的普通 Python 替身）返回 True，
    替身因此不会被误杀；但它对 ``None`` 同样返回 True，判空必须由前置的
    ``widget is not None`` 兜住。
    """
    return widget is not None and _shiboken_is_valid(widget)


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
VAR_INFO_STATS_CACHE_MAX = 512  # 统计缓存上限（单条约150字节，共约77KB）
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
# 状态栏上翻抽屉（P0-2）：宽度上限。实测（offscreen，14 pt 默认字体）520 px
# 下值列净宽 384 px，约等于 59 个拉丁字符 —— 够放常见工程路径，更长的绝对
# 路径按尾部省略（开头目录保留），完整值走行尾「复制」与 tooltip。再宽会
# 吃掉绘图区视线，再窄则路径中段都读不到
STATUS_DRAWER_WIDTH = 600
# x 轴抽屉的专用宽度上限（文件抽屉沿用上面的 520）：这扇抽屉的内容比文件抽屉
# 窄得多，共用 520 会在右侧白留一大块空。440 是扫出来的 —— 最紧的一行是页脚
# （提示文字 + 恢复默认 + 应用），实测 440 下 Fusion 余 21 px、cocoa 余 50 px；
# 再往下收到 420，Fusion 的页脚只剩 1 px（222/221），等于没留。
# 换原生 QPushButton 后档位行必须单独占一行也是这条链的一环：Fusion 给 push
# button ~80 px 最小宽，四枚 375 px 挤在频率框右侧会把自然宽顶到 600。
STATUS_DRAWER_WIDTH_AXIS = 440
# 抽屉与状态栏、窗口上沿之间的距离：与状态栏两端留白同一口径，让抽屉边缘
# 不与窗口圆角重叠
STATUS_DRAWER_EDGE_MARGIN = 8
# 抽屉可用高度的下限：状态栏以上放不到这个高度就不弹（弹出来只能看到一两行、
# 剩下的全靠滚动，不如不弹），改由调用方给一句可执行的提示。实测内容高：文件
# 信息抽屉 7 行 = offscreen/Fusion 170 px、cocoa 真机 160 px；x 轴基准抽屉 5 行
# （档位单独成行后）= Fusion 212 px、cocoa 206 px。140 是"再矮就没法看"的位置，
# 比两扇抽屉的自然高度都低一档
STATUS_DRAWER_MIN_HEIGHT = 140
# x 轴时间基准的系数范围（状态栏中段抽屉与顶部「时间修正」对话框**共用**）：
# 1e-6 s/样本 = 最高 1 MHz 采样率，1e6 s/样本 = 最低 1 µHz。
# 必须同源：两处各写一份下限时，抽屉能设到 1e-6，而对话框的 spinbox 会把
# 这个值夹回自己的下限（实测 0.0001），用户只是点开看一眼，点「确定」就把
# 基准改坏了 —— 范围不一致的代价不是"少个档位"，而是静默串改。
X_AXIS_FACTOR_MIN = 1e-6
X_AXIS_FACTOR_MAX = 1e6
# 采样率预设档位（Hz 与按钮文字）：现场试基准最常按这几个量级跳，
# 10 的幂次手打也容易错一位，所以给按钮而不是只留输入框。
# 档位由作者定（补 5 Hz、撤 1k/10k）；5 个按钮时预设行右边缘已到 449/484，
# 再加档只会继续挤压同一列里的预览行（见 status_drawer 的裁字问题）
# 按钮文字自带单位（作者定）：光秃秃的 "1 / 5 / 10 / 100" 在频率行里读不出
# 是 Hz 还是系数，而这一行没有别的单位线索。
X_AXIS_FREQUENCY_PRESETS = (
    (1.0, "1Hz"),
    (5.0, "5Hz"),
    (10.0, "10Hz"),
    (100.0, "100Hz"),
)
# 「文件路径」行送进剪贴板时的写法风格：native（当前平台，默认）| windows | posix。
# 背景（实测）：Windows 上从 Explorer 把文件拖进主窗口时，Qt 的
# QUrl.toLocalFile() 会把 UNC 网盘路径产出成 "//host/share/x.csv"（正斜杠）。
# 这种串粘回 Windows 会被 Shell 当 URL 交给浏览器（实测跳到 Edge），只有
# "\\host\\share" 才能跳转网盘；文件对话框给的是原生反斜杠，所以症状是
# “有时能用有时不能用”。入口已统一 normalize（见 paths.normalize_input_path），
# 本开关只决定展示/复制的写法；跨平台协作（Mac 上复制给 Windows 用）时手动选
PATH_COPY_STYLE = "native"
# 复制路径时是否包双引号（CMD / PowerShell 里含空格的路径必需）：
# auto（含空白或 &()#!^"'%<>,;=$`*? 才包）| always（对齐 Explorer「复制为路径」）
# | never（粘进 Excel 单元格 / 程序输入框时更干净）
PATH_COPY_QUOTE = "auto"
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

# Excel object 列数值兜底阈值：非空值中可解析为数值的比例须达到此值才转数值，
# 否则保留原文本（状态/枚举/备注列不得被静默销毁成 NaN）
EXCEL_NUMERIC_FALLBACK_RATIO = 0.9

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
        if isinstance(values, np.ndarray):
            # 快速路径：已经是 numpy 数组，避免冗余 float64 拷贝
            if values.dtype == np.float32:
                if values.size == 0:
                    return True, 0.0
                # float32 输入：只需检查是否有 inf/nan
                finite_mask = np.isfinite(values)
                if not finite_mask.any():
                    return False, None
                abs_max = float(np.max(np.abs(values[finite_mask])))
                # float32 值必然 <= FLOAT32_REPRESENTABLE_MAX，但保留统一比较
                return abs_max <= FLOAT32_REPRESENTABLE_MAX, abs_max
            elif values.dtype == np.float64:
                if values.size == 0:
                    return True, 0.0
                # float64 输入：检查 max abs 是否超出 float32 范围
                finite_mask = np.isfinite(values)
                if not finite_mask.any():
                    return False, None
                abs_max = float(np.max(np.abs(values[finite_mask])))
                return abs_max <= FLOAT32_REPRESENTABLE_MAX, abs_max
            else:
                # 其他 dtype（int 等），转 float64 检查
                arr = values.astype(np.float64)
        elif isinstance(values, pd.Series):
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
