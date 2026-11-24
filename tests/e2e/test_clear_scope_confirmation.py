"""R03/R09：清空范围在操作前明确，顶部全局清空默认取消。"""

from __future__ import annotations

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QDragEnterEvent, QFontMetrics
from PySide6.QtWidgets import QMessageBox

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


def _answer_global_clear(monkeypatch, answer):
    """记录确认窗参数；不依赖替身实现是否保存 kwargs。"""
    calls = []

    def question(parent, title, text, buttons=None, defaultButton=None):
        calls.append((title, text, buttons, defaultButton))
        return answer

    monkeypatch.setattr(QMessageBox, "question", staticmethod(question))
    return calls


def _plots(first, second):
    return first.plot_widgets[0].plot_widget, second.plot_widgets[0].plot_widget


def test_shift_hover_shows_current_subplot_scope_before_release(
    loaded_window, monkeypatch, qapp
):
    """真实覆盖层在释放前可见且完整显示「当前子图」，不是执行后播报。"""
    mw = loaded_window
    pw = mw.current_page.plot_widgets[0].plot_widget
    assert pw.plot_variable("speed")
    old_ci = pw.curves["speed"]
    # offscreen 没有真实鼠标位置；只屏蔽光标几何守卫，保留真实覆盖层绘制链。
    monkeypatch.setattr(pw, "_should_hide_drag_indicator", lambda _host: False)
    mime = build_var_mimedata(["rpm"])
    event = QDragEnterEvent(
        QPoint(20, 20),
        Qt.DropAction.CopyAction | Qt.DropAction.MoveAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.ShiftModifier,
    )

    pw.dragEnterEvent(event)
    qapp.processEvents()

    overlay = mw.current_page.plot_widgets[0]._indicator
    assert event.isAccepted()
    assert overlay.isVisible()
    assert overlay._text == "释放以替换当前子图"
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
    pw.dragLeaveEvent(event)


def test_top_clear_button_names_global_scope(loaded_window):
    button = loaded_window.clear_all_plots_btn

    assert button.text() == "清空全部页"
    assert "全部页" in button.toolTip()
    assert "确认" in button.toolTip()


def test_global_clear_cancel_preserves_every_page_and_has_no_announcement(
    loaded_window, dialog_stubs, monkeypatch, qtbot
):
    mw = loaded_window
    first, second = _two_pages_with_curve(mw, qtbot)
    first_pw, second_pw = _plots(first, second)
    before = (first_pw.curves["speed"], second_pw.curves["rpm"])
    calls = _answer_global_clear(monkeypatch, QMessageBox.StandardButton.No)
    broadcasts = []
    monkeypatch.setattr(
        mw, "_broadcast", lambda message, level="info": broadcasts.append(message)
    )

    qtbot.mouseClick(mw.clear_all_plots_btn, Qt.MouseButton.LeftButton)

    assert len(calls) == 1
    title, text, buttons, default = calls[0]
    assert "全部页" in title
    assert "2 页" in text and "2 条曲线" in text
    assert buttons & QMessageBox.StandardButton.Yes
    assert buttons & QMessageBox.StandardButton.No
    assert default == QMessageBox.StandardButton.No
    assert (first_pw.curves["speed"], second_pw.curves["rpm"]) == before
    assert not any("已清除" in message for message in broadcasts)


def test_global_clear_confirm_clears_all_pages_and_announces_scope(
    loaded_window, monkeypatch, qtbot
):
    mw = loaded_window
    first, second = _two_pages_with_curve(mw, qtbot)
    first_pw, second_pw = _plots(first, second)
    calls = _answer_global_clear(monkeypatch, QMessageBox.StandardButton.Yes)
    broadcasts = []
    monkeypatch.setattr(
        mw, "_broadcast", lambda message, level="info": broadcasts.append(message)
    )

    qtbot.mouseClick(mw.clear_all_plots_btn, Qt.MouseButton.LeftButton)

    assert len(calls) == 1
    assert first_pw.curves == {} and second_pw.curves == {}
    assert broadcasts == ["已清除全部页 · 2 条曲线"]


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
