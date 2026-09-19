"""P1-5：变量数值表关闭后必须真销毁，不再留下孤儿窗口

`closeEvent` 过去只 `hide()`：`_instance = None` 断了 Python 侧引用，C++ 骨架
（2×QTableView + tab widget + delegates）却因 parent 是主窗口而继续挂着，
反复开关线性泄漏。改为真 `deleteLater()` 后，在途延迟回调必须靠存活判定收尾。
"""

import sys
import time

import pandas as pd
import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QWidget
from shiboken6 import isValid

from src.ui.table_dialog import DataTableDialog


def pump(ms: int = 50) -> None:
    end = time.monotonic() + ms / 1000.0
    while time.monotonic() < end:
        QCoreApplication.processEvents()
        time.sleep(0.002)


def flush_deferred_deletes() -> None:
    """deleteLater 要等 DeferredDelete 投递才真正销毁 C++ 对象"""
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QCoreApplication.processEvents()


@pytest.fixture()
def clean_singleton():
    DataTableDialog._instance = None
    yield
    DataTableDialog._instance = None


@pytest.fixture()
def qt_errors(monkeypatch):
    """捕获 Qt 回调内的未捕获异常（否则测试会假绿）"""
    caught = []
    monkeypatch.setattr(
        sys, "excepthook", lambda t, v, tb: caught.append((t.__name__, str(v)))
    )
    return caught


def test_repeated_open_close_leaves_no_orphan_dialog(qapp, clean_singleton, app_settings):
    owner = QWidget()
    series = pd.Series([1.0, 2.0, 3.0], name="a")

    for round_no in range(1, 4):
        dlg = DataTableDialog.popup("a", series, parent=owner)
        dlg.set_skip_close_confirmation(True)
        dlg.close()
        flush_deferred_deletes()

        orphans = [d for d in owner.findChildren(DataTableDialog) if isValid(d)]
        assert not orphans, f"第 {round_no} 次开关后主窗口仍挂着 {len(orphans)} 个实例"
        assert not isValid(dlg)


def test_inflight_callbacks_after_close_are_skipped(qapp, clean_singleton, app_settings, qt_errors):
    """关闭时 100ms 的闪烁回调仍在途：销毁后必须静默跳过而不是打异常"""
    owner = QWidget()
    dlg = DataTableDialog.popup("a", pd.Series([1.0, 2.0, 3.0], name="a"), parent=owner)
    dlg.set_skip_close_confirmation(True)

    dlg.close()
    flush_deferred_deletes()
    pump(300)

    assert not isValid(dlg)
    assert not qt_errors, f"关闭后在途回调抛了异常：{qt_errors}"
