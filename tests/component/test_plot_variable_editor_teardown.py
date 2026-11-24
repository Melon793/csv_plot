"""变量编辑器关闭收口：信号只断一次，且不误伤其他接收者。"""

import types
import warnings

import pytest
from PySide6.QtCore import QObject, Qt
from PySide6.QtTest import QTest

from src.core.settings import ConfigKey
from src.ui.plot_variable_editor import PlotVariableEditorDialog
from src.ui.widgets.variable_search_bar import VariableSearchBar
from tests.fixtures.waits import flush_deferred_deletes

_NAMES = ["a", "b", "c"]
_SETTINGS_KEYS = (
    ConfigKey.PLOT_VARIABLE_EDITOR_COLUMN_WIDTHS,
    ConfigKey.PLOT_VARIABLE_EDITOR_FIXED_COLS,
    ConfigKey.PLOT_VARIABLE_EDITOR_GEOMETRY,
)


@pytest.fixture()
def editor_factory(plot_factory, qapp, app_settings):
    for key in _SETTINGS_KEYS:
        app_settings.set_value(key, "")

    created = []

    def _make(pw=None):
        if pw is None:
            pw = plot_factory()
        host = pw.window()
        host.loader = types.SimpleNamespace(var_names=list(_NAMES), units={})
        host.data_validity = {}
        dlg = PlotVariableEditorDialog(pw)
        dlg.show()
        qapp.processEvents()
        assert dlg.search_bar is not None
        created.append((dlg, pw))
        return dlg, pw

    yield _make

    for dlg, _pw in created:
        dlg._width_save_timer.stop()
        if dlg.search_bar is not None:
            dlg.search_bar._debounce_timer.stop()
        if dlg.isVisible():
            dlg.close()
        dlg.deleteLater()
    flush_deferred_deletes()
    for key in _SETTINGS_KEYS:
        app_settings.set_value(key, "")
    app_settings.sync()


def _count_disconnects(monkeypatch, pw):
    """只统计目标信号实例的 disconnect 调用，仍转发给原实现。"""
    signal = pw.curves_changed
    original = type(signal).disconnect
    calls = []

    def counted(signal_instance, *args, **kwargs):
        if signal_instance is signal:
            calls.append(args)
        return original(signal_instance, *args, **kwargs)

    monkeypatch.setattr(type(signal), "disconnect", counted)
    return calls


def _assert_no_disconnect_warning(caught):
    failed = [
        str(item.message)
        for item in caught
        if "Failed to disconnect" in str(item.message)
    ]
    assert not failed, failed


def _assert_two_slots_disconnected_once(calls, dlg):
    assert len(calls) == 2, calls
    assert sum(args[0] == dlg._on_curves_changed_incremental for args in calls) == 1
    assert sum(args[0] == dlg.search_bar._on_curves_changed for args in calls) == 1


def test_window_close_disconnects_each_slot_once(
    editor_factory, qapp, monkeypatch
):
    """closeEvent → done 双入口只能产生两个目标断连，且无 libpyside 警告。"""
    dlg, pw = editor_factory()
    calls = _count_disconnects(monkeypatch, pw)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        dlg.close()
        qapp.processEvents()

    _assert_no_disconnect_warning(caught)
    _assert_two_slots_disconnected_once(calls, dlg)
    assert not dlg.isVisible()
    assert dlg._teardown_done


@pytest.mark.parametrize("close_kind", ["accept", "reject", "escape"])
def test_dialog_button_and_escape_paths_have_no_disconnect_warning(
    editor_factory, qapp, close_kind
):
    """确定/取消/Esc 均正常隐藏；Esc 从表格发送，避开搜索栏按键拦截。"""
    dlg, _pw = editor_factory()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        if close_kind == "escape":
            viewport = dlg.var_table.viewport()
            viewport.setFocus()
            assert viewport.hasFocus()
            QTest.keyClick(viewport, Qt.Key.Key_Escape)
        elif close_kind == "accept":
            dlg.accept()
        else:
            dlg.reject()
        qapp.processEvents()

    _assert_no_disconnect_warning(caught)
    assert not dlg.isVisible()
    assert dlg._teardown_done


def test_repeated_teardown_is_idempotent(editor_factory, monkeypatch):
    """直接重复收口也只断两个槽各一次。"""
    dlg, pw = editor_factory()
    calls = _count_disconnects(monkeypatch, pw)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        dlg._teardown()
        dlg._teardown()

    _assert_no_disconnect_warning(caught)
    _assert_two_slots_disconnected_once(calls, dlg)


def test_closed_editor_stops_responding_but_open_editor_responds(
    editor_factory, monkeypatch
):
    """关闭后两个回调停止；包装器在构造前安装，确保已连接槽被计数。"""
    editor_calls = 0
    search_calls = 0
    original_editor = PlotVariableEditorDialog._on_curves_changed_incremental
    original_search = VariableSearchBar._on_curves_changed

    def counted_editor(self):
        nonlocal editor_calls
        editor_calls += 1
        return original_editor(self)

    def counted_search(self):
        nonlocal search_calls
        search_calls += 1
        return original_search(self)

    monkeypatch.setattr(
        PlotVariableEditorDialog, "_on_curves_changed_incremental", counted_editor
    )
    monkeypatch.setattr(VariableSearchBar, "_on_curves_changed", counted_search)
    dlg, pw = editor_factory()

    pw.curves_changed.emit()
    assert (editor_calls, search_calls) == (1, 1)

    dlg.close()
    pw.curves_changed.emit()
    assert (editor_calls, search_calls) == (1, 1)


def test_close_does_not_disconnect_other_receivers(editor_factory):
    """不得用无参数 disconnect() 误断同一信号上的独立接收者。"""

    class Receiver(QObject):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def slot(self):
            self.calls += 1

    dlg, pw = editor_factory()
    receiver = Receiver()
    pw.curves_changed.connect(receiver.slot)

    dlg.close()
    pw.curves_changed.emit()

    assert receiver.calls == 1


def test_closing_one_editor_does_not_affect_another(
    editor_factory, monkeypatch
):
    """两个实例各自收口；关闭 A 后 B 的两个回调仍正常响应。"""
    calls = {}
    original_editor = PlotVariableEditorDialog._on_curves_changed_incremental
    original_search = VariableSearchBar._on_curves_changed

    def counted_editor(self):
        counts = calls.setdefault(id(self), [0, 0])
        counts[0] += 1
        return original_editor(self)

    def counted_search(self):
        counts = calls.setdefault(id(self.parent()), [0, 0])
        counts[1] += 1
        return original_search(self)

    monkeypatch.setattr(
        PlotVariableEditorDialog, "_on_curves_changed_incremental", counted_editor
    )
    monkeypatch.setattr(VariableSearchBar, "_on_curves_changed", counted_search)
    dlg_a, pw = editor_factory()
    dlg_b, _ = editor_factory(pw)

    try:
        pw.curves_changed.emit()
        assert calls[id(dlg_a)] == [1, 1]
        assert calls[id(dlg_b)] == [1, 1]

        dlg_a.close()
        pw.curves_changed.emit()
        assert calls[id(dlg_a)] == [1, 1]
        assert calls[id(dlg_b)] == [2, 2]

        dlg_b.close()
        pw.curves_changed.emit()
        assert calls[id(dlg_a)] == [1, 1]
        assert calls[id(dlg_b)] == [2, 2]
    finally:
        # 断言失败时也在类包装仍生效期间收口，避免清理阶段出现误报警告。
        dlg_a.close()
        dlg_b.close()
