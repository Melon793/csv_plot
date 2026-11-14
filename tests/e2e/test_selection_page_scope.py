"""选择作用域 e2e：同页单持有、页级清空、跨页保留、非 plot 区不误清。"""

import pytest

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest

from src.core.data_types import SYMBOL_POLICY_OFF
from src.ui.widgets.multi_curve_manager import hit_test_curve
from tests.e2e.test_curve_selection_flow import _click_plot, _curve_pos, _pw
from tests.fixtures.waits import settle

CTRL = Qt.KeyboardModifier.ControlModifier


@pytest.fixture()
def plotted_window(loaded_window, qtbot, qapp):
    """首页画好 speed/rpm 的真实窗口（与 flow 夹具同几何前提）。"""
    mw = loaded_window
    mw.resize(1200, 800)
    qapp.processEvents()
    pw = mw.plot_widgets[0].plot_widget
    for name in ("speed", "rpm"):
        assert pw.plot_variable(name), name
    for ci in pw.curves.values():
        ci.symbol_policy = SYMBOL_POLICY_OFF
    pw._refresh_selection_projections()
    qapp.processEvents()
    qtbot.waitUntil(
        lambda: pw.view_box.sceneBoundingRect().height() > 100, timeout=5000
    )
    settle()
    return mw


def _any_blank(pw):
    """在绘图区内扫一个不命中任何曲线的 viewport 点。"""
    vb_rect = pw.view_box.sceneBoundingRect()
    for y in range(8, max(9, pw.viewport().height() - 8), 7):
        for x in range(8, max(9, pw.viewport().width() - 8), 13):
            pos = QPoint(x, y)
            scene = pw.mapToScene(pos)
            if not vb_rect.contains(scene):
                continue
            if hit_test_curve(pw, scene) is None:
                return pos
    pytest.fail("找不到空白点")


def _select(pw, name):
    _click_plot(pw, _curve_pos(pw, name))


@pytest.mark.full_matrix
def test_sibling_selection_replaces_page_holder_silently(two_plots):
    mw, page, left, right = two_plots
    _select(left, "speed")
    assert left._legend_selection == {"speed"}
    mw._message_label.setText("")

    _select(right, "rpm")

    assert right._legend_selection == {"rpm"}
    assert left._legend_selection == set(), "同页兄弟须被静默清空"
    assert page.has_any_selection() is True
    assert mw._message_label.text() == "已选中：rpm", "只播发起方"


@pytest.mark.full_matrix
def test_blank_click_clears_holder_on_other_plot(two_plots):
    mw, page, left, right = two_plots
    _select(left, "speed")
    assert page.has_any_selection() is True

    _click_plot(right, _any_blank(right), Qt.KeyboardModifier.NoModifier)

    assert left._legend_selection == set()
    assert page.has_any_selection() is False
    assert mw._message_label.text() == "已清除选择"


@pytest.mark.full_matrix
def test_ctrl_blank_on_other_plot_keeps_holder(two_plots):
    """空白处 Ctrl 单击不得清掉同页持有者。"""
    _mw, page, left, right = two_plots
    _select(left, "speed")

    _click_plot(right, _any_blank(right), CTRL)

    assert left._legend_selection == {"speed"}
    assert page.has_any_selection() is True


@pytest.mark.full_matrix
def test_non_plot_click_keeps_selection(two_plots):
    mw, _page, left, _right = two_plots
    _select(left, "speed")

    # 状态栏标签是非 plot 区域：焦点进出不得触发任何清选（无 focusOut 逻辑）
    QApplication.processEvents()
    mw._message_label.setFocus()
    mw._message_label.clearFocus()

    assert left._legend_selection == {"speed"}


def test_cross_page_selection_is_preserved(plotted_window, qtbot, qapp):
    mw = plotted_window
    first = mw.current_page
    first_pw = first.plot_widgets[0].plot_widget
    _select(first_pw, "speed")
    assert first_pw._legend_selection == {"speed"}

    second = mw.new_page()
    assert second is not None
    qtbot.waitUntil(lambda: len(second.plot_widgets) > 0, timeout=5000)
    second_pw = second.plot_widgets[0].plot_widget
    assert second_pw.plot_variable("rpm")
    qapp.processEvents()
    qtbot.waitUntil(
        lambda: second_pw.view_box.sceneBoundingRect().height() > 100, timeout=5000
    )
    settle()
    _select(second_pw, "rpm")

    # 第二页空白单击只清第二页
    _click_plot(second_pw, _any_blank(second_pw), Qt.KeyboardModifier.NoModifier)
    assert second_pw._legend_selection == set()

    mw.activate_page(first)
    qapp.processEvents()
    assert first_pw._legend_selection == {"speed"}, "跨页选择必须保留"
    assert first.has_any_selection() is True


def test_legend_blank_click_clears_page_holder(plotted_window, qapp):
    mw = plotted_window
    pw = _pw(mw)
    _select(pw, "speed")
    assert pw._legend_selection == {"speed"}
    browser = pw.legend_label

    pos = browser.viewport().rect().bottomRight() - QPoint(3, 3)
    QTest.mouseClick(
        browser.viewport(), Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier, pos,
    )
    qapp.processEvents()

    assert pw._legend_selection == set()
    assert mw._message_label.text() == "已清除选择"
