"""崩溃兜底：把未处理异常写进日志目录。

打包版禁用了控制台（``scripts/build_win.py`` 的 ``--windows-console-mode=disable``），
而 PySide6 对 Qt 槽内的未捕获异常只走 ``sys.excepthook`` 打印到 stderr ——
用户端表现为「闪退、零线索」。这里在 ``main()`` 开头安装 ``sys`` / ``threading``
的 excepthook（Python 层 traceback）与 ``faulthandler``（C 层段错误 / 死锁）。
"""

from __future__ import annotations

import faulthandler
import os
import sys
import threading

from src.core.logger import LOG_FILE_NAME, get_logger

CRASH_LOGGER_NAME = "app.crash"
CRASH_LOG_NAME = "csv_plot_crash.log"
MAX_CRASH_LOG_BYTES = 1 * 1024 * 1024

# faulthandler 只持有 dup 后的 fd，文件对象必须模块级保活，否则被 GC 关掉 fd
_fault_handler_file = None
_previous_sys_hook = None
_previous_threading_hook = None
_installed = False


def crash_log_path() -> str:
    """崩溃转储文件路径（与主日志同目录，独立文件避免轮转后写到已改名的句柄）。"""
    return os.path.join(os.path.dirname(LOG_FILE_NAME), CRASH_LOG_NAME)


def _log_crash(message: str, exc_type, exc_value, exc_tb) -> None:
    try:
        # get_logger 会按需初始化 LogManager，文件 handler 此时已就位
        get_logger(CRASH_LOGGER_NAME).critical(
            message, exc_info=(exc_type, exc_value, exc_tb)
        )
    except Exception:
        # 兜底钩子自身绝不能抛异常，否则解释器只留下更少的线索
        pass


def _sys_excepthook(exc_type, exc_value, exc_tb):
    if exc_type is not SystemExit:
        _log_crash("未处理异常", exc_type, exc_value, exc_tb)
    previous = _previous_sys_hook
    if previous is not None:
        previous(exc_type, exc_value, exc_tb)


def _threading_excepthook(args):
    thread_name = getattr(args.thread, "name", "?")
    _log_crash(
        f"线程 '{thread_name}' 未处理异常",
        args.exc_type,
        args.exc_value,
        args.exc_traceback,
    )
    previous = _previous_threading_hook
    if previous is not None:
        previous(args)


def _enable_faulthandler() -> bool:
    path = crash_log_path()
    mode = "a"
    try:
        if os.path.getsize(path) > MAX_CRASH_LOG_BYTES:
            mode = "w"
    except OSError:
        pass
    try:
        file_obj = open(path, mode, encoding="utf-8")
    except OSError:
        return False
    try:
        faulthandler.enable(file=file_obj, all_threads=True)
    except Exception:
        file_obj.close()
        return False
    global _fault_handler_file
    if _fault_handler_file is not None:
        _fault_handler_file.close()
    _fault_handler_file = file_obj
    return True


def install_crash_logging() -> bool:
    """安装崩溃兜底，返回 faulthandler 是否可用；重复调用只生效一次。"""
    global _installed, _previous_sys_hook, _previous_threading_hook
    if _installed:
        return faulthandler.is_enabled()

    _previous_sys_hook = sys.excepthook
    _previous_threading_hook = threading.excepthook
    sys.excepthook = _sys_excepthook
    threading.excepthook = _threading_excepthook
    _installed = True
    return _enable_faulthandler()
