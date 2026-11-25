"""R03/R09：清空范围在操作前明确，顶部全局清空默认取消。"""

from __future__ import annotations

import pytest

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QDragEnterEvent, QFontMetrics
from PySide6.QtWidgets import QApplication, QMessageBox

from src.ui.drag_drop import build_var_mimedata
from src.ui.drag_visuals import INDICATOR_FONT_PX, INDICATOR_INSET


def _two_pages_with_curve(mw, qtbot):
    first = mw.current_page
    second = mw.new_page()
    assert second is not None
    qtbot.waitUntil(lambda: len(second.plot_widgets) > 0, timeout=3000)
    assert second.plot_widgets[0].plot_widget.plot_variable("rpm")
    mw.activate_page(first)
    qtbot.waitUntil(lambda: mw.current_page is first, timeout=3000)
    assert first.plot_widgets[0].plot_widget.plot_variable("speed")
    return first, second


def _answer_current_page_clear(monkeypatch, answer):
    """记录确认窗参数；不依赖替身实现是否保存 kwargs。"""
    calls = []

    def question(parent, title, text, buttons=None, defaultButton=None):
        calls.append((title, text, buttons, defaultButton))
        return answer

    monkeypatch.setattr(QMessageBox, "question", staticmethod(question))
    return calls


def _plots(first, second):
    return first.plot_widgets[0].plot_widget, second.plot_widgets[0].plot_widget


@pytest.mark.parametrize(
    ("var_names", "expected_text"),
    [
        (["rpm"], "释放以替换当前子图"),
        ([], "空载荷，释放不会更改当前子图"),
    ],
)
def test_shift_hover_shows_current_subplot_scope_before_release(
    loaded_window, monkeypatch, qapp, var_names, expected_text
):
    """真实覆盖层在释放前可见，120ms 修饰键轮询也不得丢失范围说明。"""
    mw = loaded_window
    container = mw.current_page.plot_widgets[0]
    pw = container.plot_widget
    assert pw.plot_variable("speed")
    old_ci = pw.curves["speed"]
    # offscreen 没有真实鼠标位置；只屏蔽光标几何守卫，保留真实覆盖层绘制链。
    monkeypatch.setattr(pw, "_should_hide_drag_indicator", lambda _host: False)
    monkeypatch.setattr(mw, "_active_drag_container", container)
    monkeypatch.setattr(
        QApplication,
        "queryKeyboardModifiers",
        staticmethod(lambda: Qt.KeyboardModifier.ShiftModifier),
    )
    mime = build_var_mimedata(var_names)
    event = QDragEnterEvent(
        QPoint(20, 20),
        Qt.DropAction.CopyAction | Qt.DropAction.MoveAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.ShiftModifier,
    )

    pw.dragEnterEvent(event)
    qapp.processEvents()

    overlay = container._indicator
    assert event.isAccepted()
    assert overlay.isVisible()
    assert overlay._text == expected_text
    font = overlay.font()
    font.setPixelSize(INDICATOR_FONT_PX)
    font.setBold(True)
    available = (
        int(
            overlay.rect()
            .adjusted(
                INDICATOR_INSET, INDICATOR_INSET, -INDICATOR_INSET, -INDICATOR_INSET
            )
            .width()
        )
        - 16
    )
    assert (
        QFontMetrics(font).elidedText(
            overlay._text, Qt.TextElideMode.ElideRight, available
        )
        == overlay._text
    )
    assert pw.curves["speed"] is old_ci, "释放前不能改动曲线"

    # 直接执行真实 120ms 守卫，避免 offscreen/后台定时器时序造成假通过。
    pw._enforce_drag_indicator_visibility()
    qapp.processEvents()
    assert overlay._text == expected_text
    pw.dragLeaveEvent(event)


def test_top_clear_button_names_current_page_scope(loaded_window):
    button = loaded_window.clear_all_plots_btn

    assert button.text() == "清空当前页"
    assert "当前页" in button.toolTip()
    assert "确认" in button.toolTip()


def test_current_page_clear_cancel_preserves_every_page_and_has_no_announcement(
    loaded_window, dialog_stubs, monkeypatch, qtbot
):
    mw = loaded_window
    first, second = _two_pages_with_curve(mw, qtbot)
    first_pw, second_pw = _plots(first, second)
    before = (first_pw.curves["speed"], second_pw.curves["rpm"])
    calls = _answer_current_page_clear(monkeypatch, QMessageBox.StandardButton.No)
    broadcasts = []
    monkeypatch.setattr(
        mw, "_broadcast", lambda message, level="info": broadcasts.append(message)
    )

    qtbot.mouseClick(mw.clear_all_plots_btn, Qt.MouseButton.LeftButton)

    assert len(calls) == 1
    title, text, buttons, default = calls[0]
    assert "当前页" in title
    assert "1 条曲线" in text
    assert buttons & QMessageBox.StandardButton.Yes
    assert buttons & QMessageBox.StandardButton.No
    assert default == QMessageBox.StandardButton.No
    assert (first_pw.curves["speed"], second_pw.curves["rpm"]) == before
    assert not any("已清" in message for message in broadcasts)


def test_current_page_clear_confirm_clears_only_current_page_and_announces_scope(
    loaded_window, monkeypatch, qtbot
):
    mw = loaded_window
    first, second = _two_pages_with_curve(mw, qtbot)
    first_pw, second_pw = _plots(first, second)
    hidden_ci = second_pw.curves["rpm"]
    calls = _answer_current_page_clear(monkeypatch, QMessageBox.StandardButton.Yes)
    broadcasts = []
    monkeypatch.setattr(
        mw, "_broadcast", lambda message, level="info": broadcasts.append(message)
    )

    qtbot.mouseClick(mw.clear_all_plots_btn, Qt.MouseButton.LeftButton)

    assert len(calls) == 1
    assert first_pw.curves == {}
    assert second_pw.curves["rpm"] is hidden_ci
    assert broadcasts == ["已清空当前页 · 1 条曲线"]


def test_current_subplot_clear_does_not_clear_other_pages(
    loaded_window, monkeypatch, qtbot
):
    mw = loaded_window
    first, second = _two_pages_with_curve(mw, qtbot)
    first_pw, second_pw = _plots(first, second)
    hidden_ci = second_pw.curves["rpm"]
    broadcasts = []
    monkeypatch.setattr(
        mw, "_broadcast", lambda message, level="info": broadcasts.append(message)
    )

    first_pw.clear_current_plot()

    assert first_pw.curves == {}
    assert second_pw.curves["rpm"] is hidden_ci
    assert broadcasts == ["已清除当前子图 · 1 条曲线"]


def test_programmatic_global_clear_does_not_ask_for_confirmation(
    loaded_window, dialog_stubs, qtbot
):
    mw = loaded_window
    first, second = _two_pages_with_curve(mw, qtbot)

    mw.cursor_sync_manager.clear_all_plots()

    assert dialog_stubs["question"] == []
    assert all(pw.curves == {} for pw in _plots(first, second))


def test_programmatic_last_page_close_does_not_ask_for_confirmation(
    loaded_window, dialog_stubs
):
    """关最后一页的单页清空分支服务模板/回滚/测试，不能被用户确认框误伤。"""
    mw = loaded_window
    page = mw.current_page
    pw = page.plot_widgets[0].plot_widget
    assert pw.plot_variable("speed")

    mw.close_page(page, confirm=False)

    assert dialog_stubs["question"] == []
    assert pw.curves == {}


def _wait_page_activation(mw, page, qtbot):
    qtbot.waitUntil(
        lambda: mw.current_page is page
        and mw._pending_activation is None
        and not page._scene_dirty,
        timeout=3000,
    )


@pytest.mark.full_matrix
def test_current_page_clear_clears_every_subplot_and_preserves_shared_state(
    two_plots, monkeypatch, qtbot
):
    mw, first, left, right = two_plots
    second = mw.new_page()
    qtbot.waitUntil(lambda: len(second.plot_widgets) > 0, timeout=3000)
    hidden = second.plot_widgets[0].plot_widget
    assert hidden.plot_variable("rpm")
    hidden_ci = hidden.curves["rpm"]
    mw.activate_page(first)
    _wait_page_activation(mw, first, qtbot)

    cache = mw.value_cache
    sentinel = object()
    cache["speed"] = sentinel
    enum_maps = mw._enum_text_maps
    enum_maps["speed"] = {0: "零"}
    mw.saved_mark_range = (1.0, 2.0)
    assert left.plot_context.value_cache is cache
    calls = _answer_current_page_clear(monkeypatch, QMessageBox.StandardButton.Yes)
    broadcasts = []
    monkeypatch.setattr(
        mw, "_broadcast", lambda message, level="info": broadcasts.append(message)
    )

    qtbot.mouseClick(mw.clear_all_plots_btn, Qt.MouseButton.LeftButton)

    assert len(calls) == 1
    assert all(
        not container.plot_widget.curves
        and not container.plot_widget.invalid_curves
        for container in first.plot_widgets
    )
    assert hidden.curves["rpm"] is hidden_ci
    assert mw.value_cache is cache and cache["speed"] is sentinel
    assert left.plot_context.value_cache is cache
    assert mw._enum_text_maps is enum_maps and enum_maps["speed"] == {0: "零"}
    assert mw.saved_mark_range == (1.0, 2.0)
    assert broadcasts == ["已清空当前页 · 2 条曲线"]
    assert mw.current_page is first and len(mw.pages) == 2


def test_empty_current_page_clear_is_silent(loaded_window, monkeypatch, qtbot):
    mw = loaded_window
    pw = mw.current_page.plot_widgets[0].plot_widget
    pw.clear_plot_item(clear_invalid=True)
    mw.clear_status_message()
    calls = _answer_current_page_clear(monkeypatch, QMessageBox.StandardButton.Yes)
    broadcasts = []
    monkeypatch.setattr(
        mw, "_broadcast", lambda message, level="info": broadcasts.append(message)
    )

    qtbot.mouseClick(mw.clear_all_plots_btn, Qt.MouseButton.LeftButton)

    assert calls == []
    assert broadcasts == []
    assert mw._message_text == ""


def test_invalid_only_clear_confirms_and_keeps_x_axis_locked(
    loaded_window, monkeypatch, qtbot
):
    mw = loaded_window
    pw = mw.current_page.plot_widgets[0].plot_widget
    pw.clear_plot_item(clear_invalid=True)
    pw.demote_to_invalid("ghost", "gray")
    mw.set_x_axis_locked(True)
    calls = _answer_current_page_clear(monkeypatch, QMessageBox.StandardButton.Yes)
    broadcasts = []
    unlocks = []
    original_unlock = mw.set_x_axis_locked

    def record_unlock(locked, reason=""):
        unlocks.append((locked, reason))
        original_unlock(locked, reason)

    monkeypatch.setattr(mw, "set_x_axis_locked", record_unlock)
    monkeypatch.setattr(
        mw, "_broadcast", lambda message, level="info": broadcasts.append(message)
    )

    qtbot.mouseClick(mw.clear_all_plots_btn, Qt.MouseButton.LeftButton)

    assert len(calls) == 1
    assert "1 个失效占位" in calls[0][1]
    assert pw.curves == {} and pw.invalid_curves == {}
    assert unlocks == []
    assert mw._x_axis_locked is True
    assert broadcasts == ["已清空当前页 · 1 个失效占位"]


def test_mixed_clear_reports_both_counts_and_unlocks_once(
    loaded_window, monkeypatch, qtbot
):
    mw = loaded_window
    pw = mw.current_page.plot_widgets[0].plot_widget
    assert pw.plot_variable("speed")
    pw.demote_to_invalid("ghost", "gray")
    mw.set_x_axis_locked(True)
    calls = _answer_current_page_clear(monkeypatch, QMessageBox.StandardButton.Yes)
    unlocks = []
    original_unlock = mw.set_x_axis_locked

    def record_unlock(locked, reason=""):
        unlocks.append((locked, reason))
        original_unlock(locked, reason)

    monkeypatch.setattr(mw, "set_x_axis_locked", record_unlock)

    qtbot.mouseClick(mw.clear_all_plots_btn, Qt.MouseButton.LeftButton)

    assert len(calls) == 1
    assert "1 条曲线（另有 1 个失效占位）" in calls[0][1]
    assert unlocks == [(False, "绘图已清空")]
    assert pw.curves == {} and pw.invalid_curves == {}
    assert mw._x_axis_locked is False


def test_cancel_preserves_curves_placeholders_cache_and_pin(
    loaded_window, monkeypatch, qtbot
):
    mw = loaded_window
    pw = mw.current_page.plot_widgets[0].plot_widget
    assert pw.plot_variable("speed")
    pw.demote_to_invalid("ghost", "gray")
    ci = pw.curves["speed"]
    invalid = pw.invalid_curves["ghost"]
    cache = mw.value_cache
    sentinel = object()
    cache["speed"] = sentinel
    pw._cursor_manager.is_cursor_pinned = True
    pw._cursor_manager.pinned_x_value = 4.0
    calls = _answer_current_page_clear(monkeypatch, QMessageBox.StandardButton.No)
    broadcasts = []
    monkeypatch.setattr(
        mw, "_broadcast", lambda message, level="info": broadcasts.append(message)
    )

    qtbot.mouseClick(mw.clear_all_plots_btn, Qt.MouseButton.LeftButton)

    assert len(calls) == 1
    assert pw.curves["speed"] is ci
    assert pw.invalid_curves["ghost"] is invalid
    assert mw.value_cache is cache and cache["speed"] is sentinel
    assert pw._cursor_manager.is_cursor_pinned is True
    assert pw._cursor_manager.pinned_x_value == 4.0
    assert broadcasts == []


def test_confirmation_rechecks_contents_after_modal_switch(
    loaded_window, monkeypatch, qtbot, qapp
):
    mw = loaded_window
    first, second = _two_pages_with_curve(mw, qtbot)
    first_pw, second_pw = _plots(first, second)
    old_ci = first_pw.curves["speed"]
    hidden_ci = second_pw.curves["rpm"]

    def question(parent, title, text, buttons=None, defaultButton=None):
        mw.activate_page(second)
        qapp.processEvents()
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", staticmethod(question))

    assert mw.cursor_sync_manager.request_clear_current_page() is True

    assert mw.current_page is second
    assert first_pw.curves == {} and first_pw.invalid_curves == {}
    assert old_ci.curve.scene() is None, "隐藏目标页的旧曲线 item 必须先摘场景"
    assert first._scene_dirty is True
    assert second_pw.curves["rpm"] is hidden_ci
    mw.activate_page(first)
    _wait_page_activation(mw, first, qtbot)
    assert first_pw.curves == {}


@pytest.mark.parametrize("change", ["add", "clear"])
def test_confirmation_rechecks_count_before_clearing(
    loaded_window, monkeypatch, change
):
    mw = loaded_window
    pw = mw.current_page.plot_widgets[0].plot_widget
    assert pw.plot_variable("speed")
    broadcasts = []
    monkeypatch.setattr(
        mw, "_broadcast", lambda message, level="info": broadcasts.append(message)
    )

    def question(parent, title, text, buttons=None, defaultButton=None):
        if change == "add":
            assert pw.plot_variable("rpm")
        else:
            pw.clear_plot_item(clear_invalid=True)
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", staticmethod(question))

    cleared = mw.cursor_sync_manager.request_clear_current_page()

    if change == "add":
        assert cleared is True
        assert broadcasts == ["已清空当前页 · 2 条曲线"]
    else:
        assert cleared is False
        assert broadcasts == []


def test_current_page_clear_is_blocked_while_loading(
    loaded_window, monkeypatch, qtbot
):
    mw = loaded_window
    pw = mw.current_page.plot_widgets[0].plot_widget
    assert pw.plot_variable("speed")
    ci = pw.curves["speed"]
    calls = _answer_current_page_clear(monkeypatch, QMessageBox.StandardButton.Yes)
    broadcasts = []
    monkeypatch.setattr(
        mw, "_broadcast", lambda message, level="info": broadcasts.append(message)
    )

    class _RunningThread:
        def isRunning(self):
            return True

    mw._thread = _RunningThread()
    try:
        qtbot.mouseClick(mw.clear_all_plots_btn, Qt.MouseButton.LeftButton)
    finally:
        del mw._thread

    assert calls == []
    assert pw.curves["speed"] is ci
    assert broadcasts == ["正在加载数据，请稍候再清空当前页"]


def test_single_page_clear_does_not_close_page(loaded_window, monkeypatch, qtbot):
    mw = loaded_window
    page = mw.current_page
    pw = page.plot_widgets[0].plot_widget
    assert pw.plot_variable("speed")
    _answer_current_page_clear(monkeypatch, QMessageBox.StandardButton.Yes)

    def unexpected_close(*args, **kwargs):
        raise AssertionError("清空当前页不能触发关页")

    monkeypatch.setattr(mw, "close_page", unexpected_close)
    qtbot.mouseClick(mw.clear_all_plots_btn, Qt.MouseButton.LeftButton)

    assert mw.pages == [page]
    assert mw.current_page is page
    assert pw.curves == {}
