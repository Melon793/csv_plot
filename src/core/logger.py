"""Logger core module - LogManager singleton with async QueueHandler/QueueListener architecture"""

from __future__ import annotations
import threading
import logging
import logging.handlers
import atexit
import os
import sys
from queue import Queue

from PySide6.QtCore import QObject, Signal
from src.utils.platform_setup import is_frozen


LOG_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)-20s | %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"
LOG_FILE_MAX_BYTES = 5 * 1024 * 1024
LOG_FILE_BACKUP_COUNT = 3


def _get_log_dir() -> str:
    """获取日志文件目录，带多级回退保证可写性。

    回退链：
    1. 打包环境 → exe 所在目录
    2. 当前工作目录（开发环境）
    3. 用户应用数据目录（Windows: %APPDATA%/csv_plot/logs, 其他: ~/.csv_plot/logs）
    4. 系统临时目录（终极兜底）
    """
    import tempfile

    candidates: list[str] = []

    # 1. 打包环境：exe 所在目录
    if is_frozen():
        exe_dir = os.path.dirname(sys.executable)
        if exe_dir:
            candidates.append(exe_dir)

    # 2. 当前工作目录（开发环境下通常就是项目根目录）
    candidates.append(os.getcwd())

    # 3. 用户应用数据目录（跨平台安全，打包环境 CWD 不可写时回退到此）
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            candidates.append(os.path.join(appdata, "csv_plot", "logs"))
    else:
        candidates.append(os.path.join(os.path.expanduser("~"), ".csv_plot", "logs"))

    # 4. 终极兜底：系统临时目录
    candidates.append(os.path.join(tempfile.gettempdir(), "csv_plot_logs"))

    for d in candidates:
        try:
            os.makedirs(d, exist_ok=True)
            test_file = os.path.join(d, ".write_test")
            with open(test_file, "w") as f:
                f.write("")
            os.remove(test_file)
            return d
        except (OSError, PermissionError):
            continue

    # 理论上不会到这里（tempdir 一定可写），但保底返回 tempdir
    return tempfile.gettempdir()


LOG_FILE_NAME = os.path.join(_get_log_dir(), "csv_plot.log")


class QSignalLogHandler(logging.Handler, QObject):
    new_log = Signal(dict)

    def __init__(self, level=logging.NOTSET):
        logging.Handler.__init__(self, level=level)
        QObject.__init__(self)

    def emit(self, record: logging.LogRecord):
        try:
            entry = {
                "levelno": record.levelno,
                "levelname": record.levelname,
                "message": self.format(record),
                "name": record.name,
                "asctime": record.asctime,
            }
            self.new_log.emit(entry)
        except Exception:
            self.handleError(record)


class LogManager:
    _instance: LogManager | None = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls) -> LogManager:
        """线程安全的单例创建（双重检查锁 DCL）"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> LogManager:
        return cls()  # __new__ 已保证单例

    def __init__(self):
        if LogManager._initialized:
            return
        with LogManager._lock:
            if LogManager._initialized:
                return
            LogManager._initialized = True

        self._ui_handler = QSignalLogHandler()
        self._ui_handler.setLevel(logging.INFO)
        ui_formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        self._ui_handler.setFormatter(ui_formatter)

        try:
            self._file_handler = logging.handlers.RotatingFileHandler(
                LOG_FILE_NAME,
                maxBytes=LOG_FILE_MAX_BYTES,
                backupCount=LOG_FILE_BACKUP_COUNT,
                encoding="utf-8",
            )
            self._file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                "%(asctime)s [%(levelname)-8s] %(name)-20s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            self._file_handler.setFormatter(file_formatter)
        except (OSError, PermissionError):
            # 日志文件不可写时降级为 NullHandler，保证程序不崩溃
            self._file_handler = logging.NullHandler()
            self._file_handler.setLevel(logging.DEBUG)

        self._queue: Queue = Queue()
        self._queue_handler = logging.handlers.QueueHandler(self._queue)
        self._queue_listener = logging.handlers.QueueListener(
            self._queue, self._ui_handler, self._file_handler, respect_handler_level=True
        )
        self._queue_listener.start()

        root_logger = logging.getLogger()

        # 环境变量控制的调试日志开关（无需改代码即可开关）:
        #   CSV_PLOT_DEBUG=1          → 全局 DEBUG 级别（开启所有模块调试输出）
        #   CSV_PLOT_DEBUG_XLIMITS=1  → X轴 limits 变更追踪 + X-link 同步过程
        #   CSV_PLOT_DEBUG_CURSOR=1   → Cursor 锁定/同步状态追踪
        #   CSV_PLOT_DEBUG_PERF=1     → 仅性能诊断日志（[PERF] 前缀，见
        #                               docs/windows_smoothness_optimization.md §2.4/§7.3）
        if os.environ.get("CSV_PLOT_DEBUG"):
            root_logger.setLevel(logging.DEBUG)
        else:
            root_logger.setLevel(logging.INFO)

        root_logger.addHandler(self._queue_handler)

        if os.environ.get("CSV_PLOT_DEBUG_XLIMITS"):
            logging.getLogger("src.ui.widgets.axis_manager").setLevel(logging.DEBUG)
            logging.getLogger("src.ui.layout_manager").setLevel(logging.DEBUG)
        if os.environ.get("CSV_PLOT_DEBUG_CURSOR"):
            logging.getLogger("widget.cursor").setLevel(logging.DEBUG)
            logging.getLogger("src.ui.cursor_sync_manager").setLevel(logging.DEBUG)
            logging.getLogger("src.ui.file_loader_manager").setLevel(logging.DEBUG)
        if os.environ.get("CSV_PLOT_DEBUG_PERF"):
            # 性能热路径三个 logger：滚轮合并/游标同步/交互收尾的 [PERF] debug 日志
            logging.getLogger("widget.plot").setLevel(logging.DEBUG)
            logging.getLogger("src.ui.widgets.event_handler").setLevel(logging.DEBUG)
            logging.getLogger("src.ui.cursor_sync_manager").setLevel(logging.DEBUG)

        atexit.register(self._stop_listener)

    def _stop_listener(self):
        try:
            self._queue_listener.stop()
        except Exception:
            pass  # atexit 期间禁止日志输出，避免与 logging 子系统冲突

    def get_logger(self, name: str) -> logging.Logger:
        return logging.getLogger(name)

    def set_ui_log_level(self, level: int):
        self._ui_handler.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return LogManager.get_instance().get_logger(name)
