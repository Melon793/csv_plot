"""P1-18：打包后无全局异常兜底的崩溃日志钩子测试。

覆盖三件事：未处理异常落进 CRITICAL 日志、钩子不吞掉默认行为、
faulthandler 目标文件不可写时安装本身不抛异常。
"""

from __future__ import annotations

import faulthandler
import logging
import sys
import threading
import traceback

import pytest

from src.core import crash_handler


class _Recorder(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """从「未安装」状态开始，且前一钩子指向静默 sink（不把 traceback 打到 stderr）。"""
    monkeypatch.setattr(crash_handler, "_installed", False)
    monkeypatch.setattr(crash_handler, "_previous_sys_hook", None)
    monkeypatch.setattr(crash_handler, "_previous_threading_hook", None)
    monkeypatch.setattr(crash_handler, "_fault_handler_file", None)
    monkeypatch.setattr(
        crash_handler, "LOG_FILE_NAME", str(tmp_path / "csv_plot.log")
    )
    monkeypatch.setattr(sys, "excepthook", lambda *a: None)
    monkeypatch.setattr(threading, "excepthook", lambda *a: None)

    old_faulthandler = faulthandler.is_enabled()
    yield tmp_path
    if old_faulthandler:
        faulthandler.enable()
    else:
        faulthandler.disable()


@pytest.fixture
def recorder():
    logger = logging.getLogger(crash_handler.CRASH_LOGGER_NAME)
    handler = _Recorder()
    logger.addHandler(handler)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)


def _raise_and_capture() -> tuple:
    try:
        raise ValueError("boom")
    except ValueError:
        return sys.exc_info()


class TestSysExcepthook:
    def test_critical_record_carries_traceback(self, isolated, recorder):
        crash_handler.install_crash_logging()
        exc_info = _raise_and_capture()

        sys.excepthook(*exc_info)

        assert len(recorder.records) == 1
        record = recorder.records[0]
        assert record.levelno == logging.CRITICAL
        assert record.name == crash_handler.CRASH_LOGGER_NAME
        assert "ValueError: boom" in "".join(
            traceback.format_exception(*record.exc_info)
        )

    def test_previous_hook_still_called(self, isolated):
        calls = []
        crash_handler.install_crash_logging()
        crash_handler._previous_sys_hook = lambda t, v, tb: calls.append((t, v, tb))
        exc_info = _raise_and_capture()

        sys.excepthook(*exc_info)

        assert calls == [exc_info]

    def test_no_previous_hook_does_not_raise(self, isolated, recorder):
        crash_handler.install_crash_logging()
        crash_handler._previous_sys_hook = None
        exc_info = _raise_and_capture()

        sys.excepthook(*exc_info)

        assert len(recorder.records) == 1

    def test_system_exit_not_logged(self, isolated, recorder):
        crash_handler.install_crash_logging()

        sys.excepthook(SystemExit, SystemExit(0), None)

        assert recorder.records == []

    def test_logging_failure_never_raises(self, isolated, monkeypatch):
        crash_handler.install_crash_logging()

        def _explode(*args, **kwargs):
            raise RuntimeError("logging is down")

        monkeypatch.setattr(crash_handler, "get_logger", _explode)
        exc_info = _raise_and_capture()

        sys.excepthook(*exc_info)  # 兜底钩子自身不得抛出


class TestThreadingExcepthook:
    @staticmethod
    def _args():
        exc_type, exc_value, exc_tb = _raise_and_capture()
        return threading.ExceptHookArgs(
            (exc_type, exc_value, exc_tb, threading.current_thread())
        )

    def test_thread_name_and_traceback_logged(self, isolated, recorder):
        crash_handler.install_crash_logging()
        args = self._args()

        threading.excepthook(args)

        assert len(recorder.records) == 1
        record = recorder.records[0]
        assert "MainThread" in record.getMessage()
        assert record.exc_info[1] is args.exc_value

    def test_previous_threading_hook_still_called(self, isolated):
        calls = []
        crash_handler.install_crash_logging()
        crash_handler._previous_threading_hook = calls.append

        threading.excepthook(self._args())

        assert len(calls) == 1


class TestFaulthandlerTarget:
    def test_dump_lands_in_crash_log(self, isolated):
        assert crash_handler.install_crash_logging() is True
        assert faulthandler.is_enabled()
        file_obj = crash_handler._fault_handler_file
        assert file_obj.name == crash_handler.crash_log_path()

        faulthandler.dump_traceback(file=file_obj)
        file_obj.flush()

        with open(crash_handler.crash_log_path(), encoding="utf-8") as f:
            content = f.read()
        assert "test_crash_handler.py" in content

    def test_unwritable_target_skips_faulthandler_only(self, isolated, recorder):
        # 崩溃日志路径被同名目录占位 → open 必定失败，但不影响 excepthook
        (isolated / crash_handler.CRASH_LOG_NAME).mkdir()

        assert crash_handler.install_crash_logging() is False
        assert crash_handler._fault_handler_file is None

        sys.excepthook(*_raise_and_capture())

        assert len(recorder.records) == 1

    def test_oversized_crash_log_is_truncated(self, isolated, monkeypatch):
        path = isolated / crash_handler.CRASH_LOG_NAME
        path.write_text("x" * 64, encoding="utf-8")
        monkeypatch.setattr(crash_handler, "MAX_CRASH_LOG_BYTES", 16)

        assert crash_handler.install_crash_logging() is True

        assert path.stat().st_size == 0

    def test_append_mode_keeps_previous_dumps(self, isolated):
        path = isolated / crash_handler.CRASH_LOG_NAME
        path.write_text("old\n", encoding="utf-8")

        assert crash_handler.install_crash_logging() is True
        file_obj = crash_handler._fault_handler_file
        faulthandler.dump_traceback(file=file_obj)
        file_obj.flush()

        with open(path, encoding="utf-8") as f:
            assert f.read().startswith("old\n")


class TestInstallContract:
    def test_repeated_install_keeps_original_previous_hook(self, isolated):
        original = sys.excepthook

        first = crash_handler.install_crash_logging()
        hook = sys.excepthook
        second = crash_handler.install_crash_logging()

        assert second is first
        assert sys.excepthook is hook
        assert crash_handler._previous_sys_hook is original
