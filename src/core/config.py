"""config"""
from __future__ import annotations
import sys
import os,logging,faulthandler,signal,threading,traceback
from typing import Any
import numpy as np
from PyQt6.QtCore import QtMsgType, qInstallMessageHandler
from PyQt6.QtWidgets import QApplication

DEBUG_LOG_ENABLED = False  # 临时排查日志开关
# X轴标签显示控制
DEFAULT_SHOW_X_AXIS_LABEL = False
_DEBUG_LOGGER = logging.getLogger("csv_plot_debug")
if DEBUG_LOG_ENABLED and not _DEBUG_LOGGER.handlers:
    _DEBUG_LOGGER.setLevel(logging.DEBUG)
    _log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv_plot_debug.log")
    _log_handler = logging.FileHandler(_log_path, encoding="utf-8")
    _log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _DEBUG_LOGGER.addHandler(_log_handler)
else:
    _DEBUG_LOGGER.addHandler(logging.NullHandler())

_FAULTHANDLER_FILE = None
_ORIGINAL_EXCEPTHOOK = sys.excepthook
_ORIGINAL_THREADING_EXCEPTHOOK = getattr(threading, "excepthook", None)
_QT_MESSAGE_HANDLER_INSTALLED = False


def debug_log(message: str, *args) -> None:
    """简单封装，方便随处开关调试日志"""
    if not DEBUG_LOG_ENABLED:
        return
    try:
        _DEBUG_LOGGER.debug(message, *args)
    except Exception:
        pass


def safe_callback(func):
    """
    装饰器：捕获回调中的异常，防止崩溃

    【稳定性优化】用于保护关键的信号回调函数，防止因对象已销毁等原因导致的崩溃。
    特别处理RuntimeError（C++对象已删除）和AttributeError。

    【开发模式】当DEBUG_LOG_ENABLED=True时，所有异常都会被打印到控制台，便于调试。
    """
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            # C++对象已删除
            err_msg = str(e).lower()
            if "deleted" in err_msg or "wrapped" in err_msg or "c++ object" in err_msg:
                debug_log("%s skipped: object deleted", func.__name__)
                return None
            raise
        except (AttributeError, TypeError) as e:
            # 属性访问错误或参数类型错误（对象可能部分销毁，或信号参数不匹配）
            if DEBUG_LOG_ENABLED:
                # 开发模式：打印详细错误信息到控制台
                print(f"[safe_callback] {func.__name__} error: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
            debug_log("%s error: %s", func.__name__, e)
            return None
        except Exception as e:
            if DEBUG_LOG_ENABLED:
                # 开发模式：打印详细错误信息到控制台
                print(f"[safe_callback] {func.__name__} unexpected error: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
            debug_log("%s unexpected error: %s", func.__name__, e)
            return None
    return wrapper


def _install_faulthandler() -> None:
    """启用 faulthandler 并记录 native crash。"""
    global _FAULTHANDLER_FILE
    if not DEBUG_LOG_ENABLED or _FAULTHANDLER_FILE is not None:
        return
    try:
        log_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(log_dir, "csv_plot_faulthandler.log")
        _FAULTHANDLER_FILE = open(path, "a", encoding="utf-8")
        faulthandler.enable(_FAULTHANDLER_FILE, all_threads=True)
        for sig in (signal.SIGSEGV, signal.SIGFPE, signal.SIGABRT, signal.SIGILL):
            try:
                faulthandler.register(sig, file=_FAULTHANDLER_FILE, all_threads=True, chain=True)
            except (ValueError, OSError):
                continue
        debug_log("Faulthandler enabled at %s", path)
    except Exception as exc:
        debug_log("Failed to enable faulthandler: %s", exc)


def _log_uncaught_exception(exc_type, exc_value, exc_traceback) -> None:
    formatted = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    debug_log("Uncaught exception:\n%s", formatted)
    if _ORIGINAL_EXCEPTHOOK:
        _ORIGINAL_EXCEPTHOOK(exc_type, exc_value, exc_traceback)


def _threading_exception_logger(args):
    formatted = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
    thread_name = getattr(args.thread, "name", "unknown")
    debug_log("Thread %s crashed:\n%s", thread_name, formatted)
    if _ORIGINAL_THREADING_EXCEPTHOOK:
        _ORIGINAL_THREADING_EXCEPTHOOK(args)


def _qt_message_handler(mode, context, message):
    level_map = {
        QtMsgType.QtDebugMsg: "DEBUG",
        QtMsgType.QtInfoMsg: "INFO",
        QtMsgType.QtWarningMsg: "WARNING",
        QtMsgType.QtCriticalMsg: "CRITICAL",
        QtMsgType.QtFatalMsg: "FATAL",
    }
    location = ""
    if context and context.file:
        location = f"{context.file}:{context.line}"
    elif context and context.category:
        location = context.category
    debug_log("QtMsg[%s] %s %s", level_map.get(mode, str(mode)), message, location)


def install_global_debug_hooks(app: QApplication) -> None:
    """一次性安装崩溃/日志钩子，便于定位 native 问题。"""
    if not DEBUG_LOG_ENABLED:
        return
    _install_faulthandler()
    if sys.excepthook is not _log_uncaught_exception:
        sys.excepthook = _log_uncaught_exception
    if hasattr(threading, "excepthook") and threading.excepthook is not _threading_exception_logger:
        threading.excepthook = _threading_exception_logger
    global _QT_MESSAGE_HANDLER_INSTALLED
    if not _QT_MESSAGE_HANDLER_INSTALLED:
        qInstallMessageHandler(_qt_message_handler)
        _QT_MESSAGE_HANDLER_INSTALLED = True
    try:
        app.aboutToQuit.connect(lambda: debug_log("QApplication.aboutToQuit emitted"))
    except Exception as exc:
        debug_log("Failed to connect aboutToQuit: %s", exc)


DEFAULT_PADDING_VAL_X = 0.05 # 默认x轴padding，单位为plot宽度   
DEFAULT_PADDING_VAL_Y = 0.1 # 默认y轴padding，单位为plot高度
FILE_SIZE_LIMIT_BACKGROUND_LOADING = 2  # 2MB：区分平均值文件(<100点)和连续测量文件(~10000点)
RATIO_RESET_PLOTS = 0.3 # 重置plot比例，超过此比例时，重置plot
FROZEN_VIEW_WIDTH_DEFAULT = 180 # 冻结视图宽度，默认值为180px
XRANGE_THRESHOLD_FOR_SYMBOLS = 100.0  # xRange宽度阈值（考虑factor后），小于此值显示symbols（细线+symbol），否则粗线无symbol
BLINK_PULSE = 200
FACTOR_SCROLL_ZOOM = 0.3
MIN_INDEX_LENGTH = 3 # 每个plot，至少显示MIN_INDEX_LENGTH个点
DEFAULT_LINE_WIDTH = 2 # 默认线宽
THICK_LINE_WIDTH = 2 # 粗线宽
THIN_LINE_WIDTH = 1 # 细线宽
UI_DEBOUNCE_DELAY_MS = 50 # UI事件防抖延迟时间
# 默认绘图布局配置
PLOT_ROW_MAX_DEFAULT = 4
PLOT_COL_MAX_DEFAULT = 3
PLOT_ROW_CURRENT_DEFAULT = 3
PLOT_COL_CURRENT_DEFAULT = 1

FLOAT32_SAFE_MAX = float(np.finfo(np.float32).max)

# 单位行自动检测阈值
UNIT_KEYWORD_RATIO_THRESHOLD = 0.2  # 单位关键字列比例超过此值，判定为单位行
VALID_NUMERIC_RATIO_THRESHOLD = 0.6  # 有效数值列比例超过此值，判定为数据行

# 单位关键字列表（子字符串匹配，用于自动检测标题行下方的单位行）
_UNIT_KEYWORDS = [
    'm', 's', 'g', 'A', 'K', 'mol', 'cd',
    'V', 'Ω', 'F', 'H', 'W', 'J', 'N', 'Nm', 'Pa', 'bar', 'm2', '/min', '/h', 'kWh', 'mm', '°CA',
    'L', 'm3',
    'ppm', 'ppb', '%',
    'rpm',
    '℃', '°F', '°C',
    '#/',
]


def _evaluate_float32_safety(values: Any) -> tuple[bool, float | None]:
    """
    判断数值是否能安全表示为 float32。

    参数:
        values: pandas Series、NumPy 数组或其他可迭代的数值序列。

    返回:
        tuple[bool, float | None]: (是否安全、绝对值最大值)
            当数据中不存在有限值时，绝对值最大值为 None。
    """
    if values is None:
        return False, None

    import pandas as pd

    try:
        if isinstance(values, pd.Series):
            arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
        else:
            try:
                arr = np.asarray(values, dtype=np.float64)
            except (ValueError, TypeError, OverflowError):
                arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)
    except Exception:
        return False, None

    if arr.size == 0:
        return True, 0.0

    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return False, None

    abs_max = float(np.max(np.abs(arr[finite_mask])))
    return abs_max <= FLOAT32_SAFE_MAX, abs_max


