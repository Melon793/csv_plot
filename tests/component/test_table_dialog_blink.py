"""数据表列闪烁（_blink_column / _blink_step_*）component 测试（offscreen）。

回归目标：highlighted_cols 是 set，同一列在 pulse 窗口内被重复闪烁时第二次
add 被去重，但仍排入等量的 off 回调 —— 旧实现 `set.remove()` 在后到的回调
抛 KeyError（复现见 tmp/repro_blink_keyerror.py），现改为 discard。

替身策略：
- 直接构造 DataTableDialog，不走 popup/add_variables，避免污染类级 _instance
  单例；`_add_variable_to_table` 在无 loader 时可独立工作（units 与滚动恢复
  分支都有判空）。
- Qt 回调内的未捕获异常只经 sys.excepthook 打 stderr，pytest 不会感知（假绿
  风险），故显式挂钩记录后再断言。
"""

import sys
import time

import pandas as pd
import pytest

from PySide6.QtCore import QCoreApplication

from src.ui.table_dialog import CustomDelegate, DataTableDialog


def pump(ms: int = 50) -> None:
    """驱动事件循环，让 QTimer.singleShot 的延后回调落地。"""
    end = time.monotonic() + ms / 1000.0
    while time.monotonic() < end:
        QCoreApplication.processEvents()
        time.sleep(0.002)


@pytest.fixture()
def dialog(qapp):
    dlg = DataTableDialog()
    dlg._add_variable_to_table("a", pd.Series([1.0, 2.0, 3.0], name="a"))
    yield dlg
    dlg.hide()
    dlg.deleteLater()


@pytest.fixture()
def qt_errors(monkeypatch):
    """捕获 Qt 回调内的未捕获异常（否则测试会假绿）"""
    caught = []
    monkeypatch.setattr(
        sys, "excepthook", lambda t, v, tb: caught.append((t.__name__, str(v)))
    )
    return caught


# ---------- B1 单次闪烁：高亮落地后按脉冲结束清除 ----------

def test_single_blink_highlights_then_clears(dialog, qt_errors):
    delegate = dialog.delegate_main
    assert delegate.highlighted_cols == set()

    dialog._blink_column("a", pulse=300)
    assert delegate.highlighted_cols == {0}, "闪烁期间该列应处于高亮态"

    pump(900)
    assert delegate.highlighted_cols == set(), "脉冲结束后应取消高亮"
    assert not qt_errors, f"单次闪烁不应抛异常：{qt_errors}"


# ---------- B2 同列重叠闪烁：不再抛 KeyError ----------

def test_overlapping_blinks_same_column_do_not_crash(dialog, qt_errors):
    delegate = dialog.delegate_main

    dialog._blink_column("a", pulse=300)  # 第一次：add(0)
    dialog._blink_column("a", pulse=300)  # 第二次：add(0) 被 set 去重，但再排一个 off
    assert delegate.highlighted_cols == {0}

    pump(900)
    # 旧实现在此处抛 KeyError: 0（第二个 off 回调 remove 已被删的元素）
    assert delegate.highlighted_cols == set()
    assert not qt_errors, f"重叠闪烁不应抛异常：{qt_errors}"


# ---------- B3 off 步骤幂等：无高亮时重复调用安全 ----------

def test_blink_step_off_is_idempotent(dialog):
    delegate = CustomDelegate()
    view = dialog.main_view

    # 未 add 过直接 off（等价于回调重复投递），必须静默无操作
    dialog._blink_step_off(delegate, 0, view)
    dialog._blink_step_off(delegate, 0, view)
    assert delegate.highlighted_cols == set()

    dialog._blink_step_on(delegate, 0, view)
    dialog._blink_step_on(delegate, 0, view)  # 重复 add 安全
    assert delegate.highlighted_cols == {0}
    dialog._blink_step_off(delegate, 0, view)
    dialog._blink_step_off(delegate, 0, view)
    assert delegate.highlighted_cols == set()


# ---------- B4 冻结列走 frozen delegate ----------

def test_frozen_column_blinks_frozen_delegate(dialog, qt_errors):
    dialog.frozen_columns = ["a"]

    dialog._blink_column("a", pulse=300)
    assert dialog.delegate_frozen.highlighted_cols == {0}
    assert dialog.delegate_main.highlighted_cols == set(), "主区 delegate 不应被点亮"

    pump(900)
    assert dialog.delegate_frozen.highlighted_cols == set()
    assert not qt_errors, f"冻结列闪烁不应抛异常：{qt_errors}"
