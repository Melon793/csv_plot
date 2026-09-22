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


# 闪烁参数约定：off 回调由 `QTimer.singleShot(pulse, _off)` 调度，与 pulse 同刻。
# 生产默认 pulse=800ms；本文件显式传 **50ms**、pump 取 3× 余量（150ms）即可
# 稳定观测到清除（原为 pulse=300 / pump(900)，每个用例白等 0.75s）。
_BLINK_PULSE_MS = 50
_BLINK_WAIT_MS = 150


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

    dialog._blink_column("a", pulse=_BLINK_PULSE_MS)
    assert delegate.highlighted_cols == {0}, "闪烁期间该列应处于高亮态"

    pump(_BLINK_WAIT_MS)
    assert delegate.highlighted_cols == set(), "脉冲结束后应取消高亮"
    assert not qt_errors, f"单次闪烁不应抛异常：{qt_errors}"


# ---------- B2 同列重叠闪烁：不再抛 KeyError ----------

def test_overlapping_blinks_same_column_do_not_crash(dialog, qt_errors):
    delegate = dialog.delegate_main

    dialog._blink_column("a", pulse=_BLINK_PULSE_MS)  # 第一次：add(0)
    dialog._blink_column("a", pulse=_BLINK_PULSE_MS)  # 第二次：add(0) 被 set 去重，但再排一个 off
    assert delegate.highlighted_cols == {0}

    pump(_BLINK_WAIT_MS)
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

    dialog._blink_column("a", pulse=_BLINK_PULSE_MS)
    assert dialog.delegate_frozen.highlighted_cols == {0}
    assert dialog.delegate_main.highlighted_cols == set(), "主区 delegate 不应被点亮"

    pump(_BLINK_WAIT_MS)
    assert dialog.delegate_frozen.highlighted_cols == set()
    assert not qt_errors, f"冻结列闪烁不应抛异常：{qt_errors}"


# ---------- B5 tab 模式 off 回调：视图销毁 / 整页移除都要静默 ----------

def _make_tab_state(dialog, group_index=0):
    """造一个最小 tab 状态：真 QTableView + CustomDelegate + PandasTableModel"""
    from PySide6.QtWidgets import QTableView

    from src.ui.table_dialog import PandasTableModel, _GroupTabState

    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    view = QTableView(dialog)
    delegate = CustomDelegate()
    view.setItemDelegate(delegate)
    model = PandasTableModel(df, {})
    view.setModel(model)
    state = _GroupTabState(
        group_index=group_index, view=view, df=df, model=model, widget_index=0
    )
    dialog._group_tabs[group_index] = state
    return state, delegate


def _delete_view(view) -> None:
    """模拟 _remove_tab 的 view.deleteLater()：必须真把 C++ 对象删掉"""
    from PySide6.QtCore import QCoreApplication, QEvent

    view.deleteLater()
    QCoreApplication.sendPostedEvents(view, QEvent.Type.DeferredDelete)
    QCoreApplication.processEvents()


def test_tab_blink_off_callback_survives_destroyed_view(dialog, qt_errors):
    """回归 P1-6：闪烁窗口内 tab 视图被销毁，off 回调不得抛 RuntimeError"""
    state, delegate = _make_tab_state(dialog)

    dialog._blink_tab_column(state, "a", pulse=_BLINK_PULSE_MS)
    assert delegate.highlighted_cols == {0}

    _delete_view(state.view)
    pump(_BLINK_WAIT_MS)

    assert not qt_errors, f"tab 视图销毁后 off 回调抛了异常：{qt_errors}"


def test_tab_blink_off_callback_skips_removed_tab(dialog, qt_errors):
    """整页被单独移除（_remove_tab 不升 _gen）：按身份判定跳过，不碰旧 delegate"""
    state, delegate = _make_tab_state(dialog)

    dialog._blink_tab_column(state, "a", pulse=_BLINK_PULSE_MS)
    assert delegate.highlighted_cols == {0}

    dialog._group_tabs.pop(state.group_index)
    pump(_BLINK_WAIT_MS)

    assert delegate.highlighted_cols == {0}, "身份已失效的 tab 不应被 off 回调改写"
    assert not qt_errors, f"tab 移除后 off 回调抛了异常：{qt_errors}"


def test_tab_blink_off_callback_respects_generation_token(dialog, qt_errors):
    """_reset_tab_mode 升令牌：过期回调作废"""
    state, delegate = _make_tab_state(dialog)

    dialog._blink_tab_column(state, "a", pulse=_BLINK_PULSE_MS)
    dialog._gen += 1
    pump(_BLINK_WAIT_MS)

    assert delegate.highlighted_cols == {0}, "令牌已作废，off 回调不应执行"
    assert not qt_errors, f"过期回调抛了异常：{qt_errors}"

