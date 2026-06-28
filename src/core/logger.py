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


LOG_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)-20s | %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"
LOG_FILE_MAX_BYTES = 5 * 1024 * 1024
LOG_FILE_BACKUP_COUNT = 3


def _get_log_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.getcwd()


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

        self._queue: Queue = Queue()
        self._queue_handler = logging.handlers.QueueHandler(self._queue)
        self._queue_listener = logging.handlers.QueueListener(
            self._queue, self._ui_handler, self._file_handler, respect_handler_level=True
        )
        self._queue_listener.start()

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)  # 临时 DEBUG 以捕获 X 轴诊断日志
        root_logger.addHandler(self._queue_handler)

        atexit.register(self._stop_listener)

    def _stop_listener(self):
        try:
            self._queue_listener.stop()
        except Exception:
            pass

    def get_logger(self, name: str) -> logging.Logger:
        return logging.getLogger(name)

    def set_ui_log_level(self, level: int):
        self._ui_handler.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return LogManager.get_instance().get_logger(name)
