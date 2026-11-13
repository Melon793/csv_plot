"""P1 legend 多选：手势 / 高亮 / 重建修剪 / 批量菜单（offscreen）。

覆盖施工计划 §9.1 P1 用例：
- Ctrl 单击加选不切显隐（anchorClicked 抑制）、Shift 范围序、空白清选、
  裸单击不清选、占位不可选；
- ExtraSelections 高亮按 href 定位（截断安全）、setHtml 重放、clear 占位
  分支清空选择集、删除/reload 修剪；
- 拖选中集 → 批量 MIME；拖未选 → 单变量且选择集不动；
- 右键已选 → 批量菜单分发、未选 → 单变量菜单且不动选择集；
- 批量样式 / 删除确认 / 入暂存区（复制/剪切）。
"""

from urllib.parse import quote

import pytest

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QDropEvent, QMouseEvent
from PySide6.QtWidgets import QApplication, QMessageBox

from src.core.data_types import DRAW_MODE_STEP_POST, SYMBOL_POLICY_OFF
from src.ui.drag_drop import (
    ANCHOR_SCHEME,
    build_legend_var_mimedata,
    clear_active_legend_drag,
    is_active_legend_batch_drag,
    parse_var_names_from_mimedata,
    set_active_legend_drag,
)
from src.ui.widgets.curve_shelf import CurveShelfWidget
from tests.component.test_legend_context_menu import _fake_menu_env


# ---------------------------------------------------------------------------
# 手势 / 高亮辅助
# ---------------------------------------------------------------------------


def _anchor_pos(browser, var_name: str, scheme: str = ANCHOR_SCHEME):
    href = f"{scheme}:///{quote(var_name, safe='')}"
    for y in range(1, 48):
        for x in range(1, 400, 2):
            if browser.anchorAt(QPoint(x, y)) == href:
                return QPoint(x, y)
    pytest.fail(f"legend 中未找到 {href} 的锚点坐标")


def _blank_pos(browser):
    for y in range(1, 48):
        for x in range(1, 400, 4):
            if browser.anchorAt(QPoint(x, y)) == "":
                return QPoint(x, y)
    pytest.fail("legend 中未找到空白坐标")


def _mouse(event_type, pos, button, modifiers=Qt.KeyboardModifier.NoModifier):
    point = QPointF(pos)
    return QMouseEvent(
        event_type,
        point,
        point,
        QPointF(0.0, 0.0),
        button,
        button,
        modifiers,
    )


def _click(browser, pos, modifiers=Qt.KeyboardModifier.NoModifier):
    browser.mousePressEvent(
        _mouse(QEvent.Type.MouseButtonPress, pos, Qt.MouseButton.LeftButton, modifiers)
    )
    browser.mouseReleaseEvent(
        _mouse(QEvent.Type.MouseButtonRelease, pos, Qt.MouseButton.LeftButton, modifiers)
    )


def _right_click(browser, pos):
    browser.mousePressEvent(
        _mouse(QEvent.Type.MouseButtonPress, pos, Qt.MouseButton.RightButton)
    )
    browser.mouseReleaseEvent(
        _mouse(QEvent.Type.MouseButtonRelease, pos, Qt.MouseButton.RightButton)
    )


@pytest.fixture()
def three_curves(plot_factory):
    pw = plot_factory()
    for name in ("a", "b", "c"):
        assert pw.plot_variable(name)
    return pw


# ---------------------------------------------------------------------------
# 手势
# ---------------------------------------------------------------------------


def test_ctrl_click_toggles_selection_without_visibility(three_curves):
    pw = three_curves
    browser = pw.legend_label
    visible_before = {name: ci.visible for name, ci in pw.curves.items()}

    _click(browser, _anchor_pos(browser, "a"), Qt.KeyboardModifier.ControlModifier)
    assert pw._legend_selection == {"a"}
    _click(browser, _anchor_pos(browser, "b"), Qt.KeyboardModifier.ControlModifier)
    assert pw._legend_selection == {"a", "b"}
    _click(browser, _anchor_pos(browser, "a"), Qt.KeyboardModifier.ControlModifier)
    assert pw._legend_selection == {"b"}

    # Ctrl 点击被 release 拦截，anchorClicked 不发出 → 显隐不变
    assert {name: ci.visible for name, ci in pw.curves.items()} == visible_before
    assert len(browser.extraSelections()) > 0


def test_shift_click_selects_range_in_curve_order(three_curves):
    pw = three_curves
    browser = pw.legend_label

    _click(browser, _anchor_pos(browser, "a"), Qt.KeyboardModifier.ControlModifier)
    _click(browser, _anchor_pos(browser, "c"), Qt.KeyboardModifier.ShiftModifier)

    assert pw._legend_selection == {"a", "b", "c"}
    assert all(ci.visible for ci in pw.curves.values())


def test_shift_click_replaces_selection_and_keeps_anchor(three_curves):
    """Shift 范围选替换旧选择集；锚点保持，连续 Shift 以同一基准调整。"""
    pw = three_curves
    browser = pw.legend_label
    _click(browser, _anchor_pos(browser, "a"), Qt.KeyboardModifier.ControlModifier)
    _click(browser, _anchor_pos(browser, "c"), Qt.KeyboardModifier.ControlModifier)
    assert pw._legend_selection == {"a", "c"}

    _click(browser, _anchor_pos(browser, "b"), Qt.KeyboardModifier.ShiftModifier)
    assert pw._legend_selection == {"b", "c"}  # 自锚点 c 到 b，替换旧集合

    _click(browser, _anchor_pos(browser, "a"), Qt.KeyboardModifier.ShiftModifier)
    assert pw._legend_selection == {"a", "b", "c"}  # 锚点仍是 c


def test_blank_click_clears_selection(three_curves):
    pw = three_curves
    browser = pw.legend_label
    _click(browser, _anchor_pos(browser, "a"), Qt.KeyboardModifier.ControlModifier)
    assert pw._legend_selection

    _click(browser, _blank_pos(browser))

    assert pw._legend_selection == set()
    assert browser.extraSelections() == []


def test_plain_click_keeps_selection_and_toggles_visibility(three_curves):
    """D4：裸单击条目不清选择集；显隐行为保持既有。"""
    pw = three_curves
    browser = pw.legend_label
    _click(browser, _anchor_pos(browser, "a"), Qt.KeyboardModifier.ControlModifier)

    _click(browser, _anchor_pos(browser, "b"))

    assert pw._legend_selection == {"a"}
    assert pw.curves["b"].visible is False  # anchorClicked 正常切换


def test_placeholder_not_selectable(three_curves):
    pw = three_curves
    browser = pw.legend_label
    pw.demote_to_invalid("c", "gray", True)
    assert "c" in pw.invalid_curves

    _click(browser, _anchor_pos(browser, "c"), Qt.KeyboardModifier.ControlModifier)

    assert pw._legend_selection == set()


# ---------------------------------------------------------------------------
# 高亮重放 / 两分支 / 修剪
# ---------------------------------------------------------------------------


def test_highlight_replayed_after_legend_sethtml(three_curves):
    pw = three_curves
    browser = pw.legend_label
    _click(browser, _anchor_pos(browser, "a"), Qt.KeyboardModifier.ControlModifier)
    _click(browser, _anchor_pos(browser, "b"), Qt.KeyboardModifier.ControlModifier)
    assert browser.extraSelections()

    pw._multi_curve_manager.update_legend()  # 触发 setHtml 重建

    assert pw._legend_selection == {"a", "b"}
    assert len(browser.extraSelections()) > 0


def test_clear_branch_clears_selection(three_curves):
    pw = three_curves
    browser = pw.legend_label
    _click(browser, _anchor_pos(browser, "a"), Qt.KeyboardModifier.ControlModifier)
    assert pw._legend_selection

    pw.update_legend_label("channel name")  # 回归占位分支

    assert pw._legend_selection == set()
    assert pw._legend_sel_anchor is None
    assert browser.extraSelections() == []


def test_selection_pruned_when_curve_removed(three_curves):
    pw = three_curves
    browser = pw.legend_label
    _click(browser, _anchor_pos(browser, "a"), Qt.KeyboardModifier.ControlModifier)
    _click(browser, _anchor_pos(browser, "b"), Qt.KeyboardModifier.ControlModifier)

    pw.remove_variable_from_plot("a")

    assert pw._legend_selection == {"b"}


# ---------------------------------------------------------------------------
# 拖拽：选中集批量 / 未选单变量
# ---------------------------------------------------------------------------


class _FakeDrag:
    def __init__(self, parent):
        self.mime = None
        self.captured = {}
        _FakeDrag.last = self

    def setMimeData(self, mime):
        self.mime = mime

    def setPixmap(self, pixmap):
        self.pixmap = pixmap

    def setHotSpot(self, point):
        pass

    def exec(self, actions):
        self.captured["batch"] = is_active_legend_batch_drag()
        self.captured["names"] = parse_var_names_from_mimedata(self.mime)
        return Qt.DropAction.MoveAction


def test_drag_selected_set_uses_batch_mime(three_curves, monkeypatch):
    import src.ui.widgets.plot_ui_manager as pum

    pw = three_curves
    browser = pw.legend_label
    _click(browser, _anchor_pos(browser, "a"), Qt.KeyboardModifier.ControlModifier)
    _click(browser, _anchor_pos(browser, "b"), Qt.KeyboardModifier.ControlModifier)

    monkeypatch.setattr(pum, "QDrag", _FakeDrag)
    browser._drag_var_name = "a"
    browser._drag_press_pos = QPoint(1, 1)
    browser._start_var_drag()

    drag = _FakeDrag.last
    assert drag.captured["batch"] is True
    assert drag.captured["names"] == ["a", "b"]
    # exec 返回后注册表必须清空
    assert is_active_legend_batch_drag() is False


def test_batch_legend_drop_uses_skip_quiet(plot_factory, qapp, monkeypatch):
    """P1 选中集拖到目标图：skip + quiet（无确认窗），源端只删成功项。"""
    src = plot_factory()
    dst = plot_factory()
    for name in ("a", "b"):
        assert src.plot_variable(name)
    assert dst.plot_variable("a")  # 目标已含 a → 应静默跳过

    dialogs = []
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: dialogs.append("q"))
    )
    monkeypatch.setattr(
        QMessageBox, "information", staticmethod(lambda *a, **k: dialogs.append("i"))
    )

    mime = build_legend_var_mimedata(["a", "b"], src)
    _keep_alive.append(mime)
    event = QDropEvent(
        QPointF(5.0, 5.0),
        Qt.DropAction.CopyAction | Qt.DropAction.MoveAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    set_active_legend_drag(src, ["a", "b"], batch=True)
    try:
        dst.dropEvent(event)
    finally:
        clear_active_legend_drag()
    qapp.processEvents()

    assert "b" in dst.curves
    assert "a" in src.curves  # 跳过项不删源
    assert "b" not in src.curves  # 成功项按移动删除
    assert dialogs == []  # 批量 quiet：无模态


_keep_alive: list = []


def test_drag_unselected_single_keeps_selection(three_curves, monkeypatch):
    import src.ui.widgets.plot_ui_manager as pum

    pw = three_curves
    browser = pw.legend_label
    _click(browser, _anchor_pos(browser, "a"), Qt.KeyboardModifier.ControlModifier)
    _click(browser, _anchor_pos(browser, "b"), Qt.KeyboardModifier.ControlModifier)

    monkeypatch.setattr(pum, "QDrag", _FakeDrag)
    browser._drag_var_name = "c"
    browser._drag_press_pos = QPoint(1, 1)
    browser._start_var_drag()

    drag = _FakeDrag.last
    assert drag.captured["batch"] is False
    assert drag.captured["names"] == ["c"]
    assert pw._legend_selection == {"a", "b"}


# ---------------------------------------------------------------------------
# 右键分发（D5）
# ---------------------------------------------------------------------------


def test_right_click_selected_opens_batch_menu(three_curves, qapp):
    pw = three_curves
    browser = pw.legend_label
    _click(browser, _anchor_pos(browser, "a"), Qt.KeyboardModifier.ControlModifier)
    _click(browser, _anchor_pos(browser, "b"), Qt.KeyboardModifier.ControlModifier)

    calls = []
    browser._exec_batch_var_menu = lambda pos: calls.append(("batch", pos))
    browser._exec_var_menu = lambda name, pos: calls.append(("single", name, pos))

    _right_click(browser, _anchor_pos(browser, "a"))
    qapp.processEvents()

    assert calls and calls[0][0] == "batch"


def test_right_click_unselected_opens_single_menu(three_curves, qapp):
    pw = three_curves
    browser = pw.legend_label
    _click(browser, _anchor_pos(browser, "a"), Qt.KeyboardModifier.ControlModifier)
    _click(browser, _anchor_pos(browser, "b"), Qt.KeyboardModifier.ControlModifier)

    calls = []
    browser._exec_batch_var_menu = lambda pos: calls.append(("batch", pos))
    browser._exec_var_menu = lambda name, pos: calls.append(("single", name, pos))

    _right_click(browser, _anchor_pos(browser, "c"))
    qapp.processEvents()

    assert calls == [("single", "c", calls[0][2])]
    assert pw._legend_selection == {"a", "b"}  # D5：未选右键不动选择集


# ---------------------------------------------------------------------------
# 批量菜单动作
# ---------------------------------------------------------------------------


def _select_ab(browser):
    _click(browser, _anchor_pos(browser, "a"), Qt.KeyboardModifier.ControlModifier)
    _click(browser, _anchor_pos(browser, "b"), Qt.KeyboardModifier.ControlModifier)


class _FakeShelfDialog:
    def __init__(self, host):
        self.shelf = CurveShelfWidget(main_window=host)
        self.shown = 0

    def show(self):
        self.shown += 1

    def raise_(self):
        pass


def _attach_shelf_host(pw):
    host = pw.window()
    messages = []
    host._show_status_message = messages.append
    dialog = _FakeShelfDialog(host)
    host.get_curve_shelf = lambda: dialog
    return host, dialog, messages


def test_batch_menu_solo_and_show_all(three_curves, monkeypatch):
    pw = three_curves
    browser = pw.legend_label
    _select_ab(browser)

    _fake_menu_env(monkeypatch, choose_text="仅显示所选")
    browser._exec_batch_var_menu(QPoint(0, 0))
    assert pw.curves["a"].visible and pw.curves["b"].visible
    assert pw.curves["c"].visible is False

    _fake_menu_env(monkeypatch, choose_text="显示全部变量")
    browser._exec_batch_var_menu(QPoint(0, 0))
    assert all(ci.visible for ci in pw.curves.values())


def test_batch_menu_copy_and_clear(three_curves, monkeypatch):
    pw = three_curves
    browser = pw.legend_label
    _select_ab(browser)
    QApplication.clipboard().setText("SENTINEL")

    _fake_menu_env(monkeypatch, choose_text="复制变量名")
    browser._exec_batch_var_menu(QPoint(0, 0))
    assert QApplication.clipboard().text() == "a、b"

    _fake_menu_env(monkeypatch, choose_text="清除选择")
    browser._exec_batch_var_menu(QPoint(0, 0))
    assert pw._legend_selection == set()


def test_batch_menu_delete_confirm_once(three_curves, monkeypatch):
    pw = three_curves
    browser = pw.legend_label
    _select_ab(browser)
    questions = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(
            lambda *a, **k: questions.append(a) or QMessageBox.StandardButton.No
        ),
    )

    _fake_menu_env(monkeypatch, choose_text="删除变量（2）")
    browser._exec_batch_var_menu(QPoint(0, 0))
    assert len(questions) == 1  # N>1 一次汇总确认
    assert set(pw.curves) == {"a", "b", "c"}

    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes),
    )
    browser._exec_batch_var_menu(QPoint(0, 0))
    assert set(pw.curves) == {"c"}
    assert pw._legend_selection == set()


def test_batch_menu_shelf_copy(three_curves, monkeypatch):
    pw = three_curves
    browser = pw.legend_label
    _select_ab(browser)
    host, dialog, messages = _attach_shelf_host(pw)

    _fake_menu_env(monkeypatch, choose_text="加入暂存区（复制）")
    browser._exec_batch_var_menu(QPoint(0, 0))

    assert {"a", "b"} == {entry.var_name for entry in dialog.shelf.entries}
    assert set(pw.curves) == {"a", "b", "c"}
    assert dialog.shown == 1
    assert any("已复制 a、b" in message for message in messages)
    assert pw._legend_selection == {"a", "b"}  # 复制不清选择集


def test_batch_menu_shelf_cut_removes_curves(three_curves, monkeypatch, qapp):
    pw = three_curves
    browser = pw.legend_label
    _select_ab(browser)
    host, dialog, messages = _attach_shelf_host(pw)

    _fake_menu_env(monkeypatch, choose_text="剪切到暂存区")
    browser._exec_batch_var_menu(QPoint(0, 0))
    qapp.processEvents()  # 源端删除 singleShot(0)

    assert {"a", "b"} == {entry.var_name for entry in dialog.shelf.entries}
    assert set(pw.curves) == {"c"}
    assert any("已剪切 a、b" in message for message in messages)
    assert pw._legend_selection == set()


def test_batch_menu_shelf_rejected_while_loading(three_curves):
    pw = three_curves
    browser = pw.legend_label
    host, dialog, messages = _attach_shelf_host(pw)
    host.is_data_loading = lambda: True

    browser._send_selection_to_shelf(["a", "b"], cut=False)

    assert dialog.shelf.count() == 0
    assert "数据加载中，暂不可搬运变量" in messages


def test_batch_style_picks_applies_to_all_and_emits_once(three_curves):
    pw = three_curves
    browser = pw.legend_label
    _select_ab(browser)
    emitted = []
    pw.curves_changed.connect(lambda: emitted.append(1))

    browser._apply_batch_style_picks(["a", "b"], {"color": "#123456"})
    assert pw.curves["a"].color == "#123456"
    assert pw.curves["b"].color == "#123456"
    assert pw.curves["c"].color != "#123456"
    assert len(emitted) == 1

    emitted.clear()
    browser._apply_batch_style_picks(["a", "b"], {"draw_mode": DRAW_MODE_STEP_POST})
    assert pw.curves["a"].draw_mode == DRAW_MODE_STEP_POST
    assert pw.curves["b"].draw_mode == DRAW_MODE_STEP_POST
    assert len(emitted) == 1

    emitted.clear()
    browser._apply_batch_style_picks(["a"], {"symbol_policy": SYMBOL_POLICY_OFF})
    assert pw.curves["a"].symbol_policy == SYMBOL_POLICY_OFF
    assert len(emitted) == 1
