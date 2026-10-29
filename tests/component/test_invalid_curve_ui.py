"""失效曲线占位（方案 B2）UI 呈现 component 测试（offscreen）。

钉住 Commit 3 的 UI 契约：
- _build_legend_html：占位条目灰色 +（缺失）后缀 + 复用 ANCHOR_SCHEME，置尾部；
- _update_header_for_curves：纯占位页也渲染图例（判空含 invalid_curves，§6.1）；
- _on_legend_anchor_clicked：占位点击仅状态栏提示、不切换显隐（§12.1）；
- _start_var_drag：占位锚点不发起拖拽（§6.1）；
- _exec_invalid_menu：占位右键轻量菜单（删除占位 + 复制变量名），删除后 emit；
- 编辑器：占位行渲染、复选框切 visible_preference、增量刷新不误删占位行、
  清空所有连带清占位。
"""

from urllib.parse import quote

import pytest

from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from src.core.data_types import InvalidCurveRef
from src.ui.drag_drop import ANCHOR_SCHEME, parse_anchor_var_name
from src.ui.plot_variable_editor import PlotVariableEditorDialog


@pytest.fixture()
def silent_dialogs(monkeypatch):
    calls = {"question": []}
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda parent, title, text, *a, **k: (
            calls["question"].append((title, text)), QMessageBox.StandardButton.Yes)[1]
        ),
    )
    return calls


def _fake_menu_env(monkeypatch, choose_text=None):
    """patch plot_ui_manager.QMenu 为纯 Python 替身（同 test_legend_context_menu）"""
    menus = []

    class _FakeAction:
        def __init__(self, text):
            self.text = text
            self.enabled = True

        def setEnabled(self, enabled):
            self.enabled = bool(enabled)

    class _FakeMenu:
        def __init__(self, parent=None):
            self._acts = []
            menus.append(self)

        def addAction(self, text):
            act = _FakeAction(text)
            self._acts.append(act)
            return act

        def addSeparator(self):
            self._acts.append(_FakeAction(""))

        def actions(self):
            return list(self._acts)

        def exec(self, pos=None):
            if choose_text is None:
                return None
            for act in self._acts:
                if act.text == choose_text and act.enabled:
                    return act
            return None

    import src.ui.widgets.plot_ui_manager as pum
    monkeypatch.setattr(pum, "QMenu", _FakeMenu)
    return menus


# ---------- 图例渲染 ----------

def test_legend_html_renders_placeholder_entry(plot_factory):
    """占位条目：灰色 +（缺失）后缀 + ✕ 符号 + 复用 ANCHOR_SCHEME"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    pw.demote_to_invalid("b", "red", True)
    pw._update_header_for_curves()

    html = pw._multi_curve_manager._build_legend_html()
    assert "（缺失）" in html
    assert "✕" in html
    # 占位锚点仍用统一协议，可被解析回变量名
    href_b = f"{ANCHOR_SCHEME}:///{quote('b', safe='')}"
    assert href_b in html
    assert parse_anchor_var_name(href_b) == "b"


def test_legend_placeholder_placed_at_tail(plot_factory):
    """占位条目置于图例尾部（真实曲线之后）"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    assert pw.add_variable_to_plot("b")
    pw.demote_to_invalid("ghost", "gray", True)

    html = pw._multi_curve_manager._build_legend_html()
    pos_b = html.index(quote("b", safe=""))
    pos_ghost = html.index(quote("ghost", safe=""))
    assert pos_ghost > pos_b, "占位应排在真实曲线之后"


def test_pure_placeholder_page_renders_legend(plot_factory):
    """纯占位页（curves 空）：图例仍渲染占位，而非重置为 'channel name'（§6.1）"""
    pw = plot_factory()
    pw.invalid_curves["a"] = InvalidCurveRef(var_name="a", color="blue", visible_preference=True)

    pw._update_header_for_curves()

    html = pw.legend_label.toHtml()
    assert "（缺失）" in html
    assert "channel name" not in pw.legend_label.toPlainText()


# ---------- 点击提示（不切换显隐） ----------

def test_anchor_click_on_placeholder_shows_status_hint(plot_factory):
    """占位锚点点击：仅状态栏提示，不切换显隐（无曲线可切，§12.1）"""
    pw = plot_factory()
    pw.invalid_curves["ghost"] = InvalidCurveRef(var_name="ghost", color="gray")
    hints = []
    host = pw.window()
    host._show_status_message = lambda msg: hints.append(msg)

    from PySide6.QtCore import QUrl
    url = QUrl(f"{ANCHOR_SCHEME}:///{quote('ghost', safe='')}")
    pw._on_legend_anchor_clicked(url)

    assert hints and "ghost" in hints[0] and "占位" in hints[0]
    # 占位未被点击删除/复活
    assert "ghost" in pw.invalid_curves
    assert "ghost" not in pw.curves


def test_anchor_click_on_real_curve_still_toggles(plot_factory):
    """真实曲线锚点点击：照常切换显隐（回归）"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    assert pw.curves["a"].visible is True

    from PySide6.QtCore import QUrl
    url = QUrl(f"{ANCHOR_SCHEME}:///{quote('a', safe='')}")
    pw._on_legend_anchor_clicked(url)

    assert pw.curves["a"].visible is False


# ---------- 拖拽过滤 ----------

def test_start_var_drag_skips_placeholder(plot_factory, monkeypatch):
    """占位锚点不发起 legend 拖拽（§6.1）"""
    pw = plot_factory()
    pw.invalid_curves["ghost"] = InvalidCurveRef(var_name="ghost", color="gray")
    browser = pw.legend_label

    started = []
    monkeypatch.setattr(
        "src.ui.widgets.plot_ui_manager.QDrag",
        lambda *a, **k: started.append(1) or _RaiseOnExec(),
    )
    browser._drag_var_name = "ghost"
    browser._drag_press_pos = QPoint(0, 0)
    browser._start_var_drag()

    assert not started, "占位不应构造 QDrag"
    assert browser._drag_var_name is None  # 状态已清空


class _RaiseOnExec:
    def setMimeData(self, *a): pass
    def setPixmap(self, *a): pass
    def setHotSpot(self, *a): pass
    def exec(self, *a):
        raise AssertionError("占位拖拽不应执行 drag.exec")


def test_start_var_drag_proceeds_for_real_curve(plot_factory, monkeypatch):
    """真实曲线锚点照常发起拖拽（回归）"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    browser = pw.legend_label

    constructed = []

    class _FakeDrag:
        def __init__(self, *a):
            constructed.append(1)
        def setMimeData(self, *a): pass
        def setPixmap(self, *a): pass
        def setHotSpot(self, *a): pass
        def exec(self, *a):
            return None

    monkeypatch.setattr("src.ui.widgets.plot_ui_manager.QDrag", _FakeDrag)
    monkeypatch.setattr("src.ui.widgets.plot_ui_manager.set_active_legend_drag", lambda *a: None)
    monkeypatch.setattr("src.ui.widgets.plot_ui_manager.clear_active_legend_drag", lambda: None)
    browser._drag_var_name = "a"
    browser._drag_press_pos = QPoint(0, 0)
    browser._start_var_drag()

    assert constructed, "真实曲线应构造 QDrag"


# ---------- 占位右键菜单 ----------

def test_invalid_menu_remove_placeholder(plot_factory, monkeypatch):
    """占位右键「删除占位」：移除并 emit curves_changed（§6.2）"""
    pw = plot_factory()
    pw.invalid_curves["ghost"] = InvalidCurveRef(var_name="ghost", color="gray")
    emitted = []
    pw.curves_changed.connect(lambda: emitted.append(1))

    _fake_menu_env(monkeypatch, choose_text="删除占位")
    pw.legend_label._exec_invalid_menu("ghost", QPoint(0, 0))

    assert "ghost" not in pw.invalid_curves
    assert len(emitted) == 1


def test_invalid_menu_copy_name(plot_factory, monkeypatch):
    """占位右键「复制变量名」：写剪贴板、不删占位、不 emit"""
    pw = plot_factory()
    pw.invalid_curves["ghost"] = InvalidCurveRef(var_name="ghost", color="gray")
    emitted = []
    pw.curves_changed.connect(lambda: emitted.append(1))
    QApplication.clipboard().setText("SENTINEL")

    _fake_menu_env(monkeypatch, choose_text="复制变量名")
    pw.legend_label._exec_invalid_menu("ghost", QPoint(0, 0))

    assert QApplication.clipboard().text() == "ghost"
    assert "ghost" in pw.invalid_curves
    assert not emitted


def test_exec_var_menu_routes_placeholder_to_invalid_menu(plot_factory, monkeypatch):
    """_exec_var_menu 对占位变量分派到 _exec_invalid_menu（不给曲线菜单）"""
    pw = plot_factory()
    pw.invalid_curves["ghost"] = InvalidCurveRef(var_name="ghost", color="gray")

    called = []
    monkeypatch.setattr(
        pw.legend_label, "_exec_invalid_menu",
        lambda name, pos: called.append(name),
    )
    menus = _fake_menu_env(monkeypatch, choose_text=None)
    pw.legend_label._exec_var_menu("ghost", QPoint(0, 0))

    assert called == ["ghost"]
    assert not menus, "占位不应弹出曲线操作菜单"


# ---------- 编辑器 ----------

def _editor(pw, qapp):
    dlg = PlotVariableEditorDialog(pw)
    qapp.processEvents()
    return dlg


def test_editor_renders_placeholder_row(plot_factory, qapp):
    """编辑器表格渲染占位行：灰名 +（缺失），行标记 UserRole+1"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    pw.demote_to_invalid("ghost", "red", True)
    dlg = _editor(pw, qapp)

    names = [
        dlg.var_table.item(r, 1).data(Qt.ItemDataRole.UserRole)
        for r in range(dlg.var_table.rowCount())
    ]
    assert names == ["a", "ghost"]
    ghost_row = 1
    assert "（缺失）" in dlg.var_table.item(ghost_row, 1).text()
    assert dlg.var_table.item(ghost_row, 1).data(Qt.ItemDataRole.UserRole + 1) is True
    dlg.close()


def test_editor_checkbox_toggles_visible_preference(plot_factory, qapp):
    """占位行复选框切换 visible_preference（复活时生效）"""
    pw = plot_factory()
    pw.invalid_curves["ghost"] = InvalidCurveRef(var_name="ghost", color="red", visible_preference=True)
    dlg = _editor(pw, qapp)

    checkbox = dlg.var_table.cellWidget(0, 0)
    checkbox.setChecked(False)

    assert pw.invalid_curves["ghost"].visible_preference is False
    dlg.close()


def test_editor_incremental_refresh_keeps_placeholder_row(plot_factory, qapp):
    """增量刷新不误删占位行（§6.3）：占位存在时回退全量重建"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    pw.demote_to_invalid("ghost", "red", True)
    dlg = _editor(pw, qapp)
    assert dlg.var_table.rowCount() == 2

    # 触发一次 curves_changed（模拟外部增删）：占位行必须仍在
    pw.curves_changed.emit()
    qapp.processEvents()

    names = [
        dlg.var_table.item(r, 1).data(Qt.ItemDataRole.UserRole)
        for r in range(dlg.var_table.rowCount())
    ]
    assert "ghost" in names, f"占位行被误删: {names}"
    assert "a" in names
    dlg.close()


def test_editor_clear_all_clears_placeholders(plot_factory, qapp, silent_dialogs):
    """编辑器「清空所有」连带清占位并 emit（§6.3）"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    pw.demote_to_invalid("ghost", "red", True)
    dlg = _editor(pw, qapp)
    emitted = []
    pw.curves_changed.connect(lambda: emitted.append(1))

    dlg.clear_all_variables()

    assert list(pw.curves) == []
    assert list(pw.invalid_curves) == []
    assert dlg.var_table.rowCount() == 0
    assert emitted
    dlg.close()


def test_editor_delete_selected_removes_placeholder(plot_factory, qapp):
    """编辑器「删除选中」可删占位行（remove_variables_from_plot 两侧都查）"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    pw.demote_to_invalid("ghost", "red", True)
    dlg = _editor(pw, qapp)

    dlg.var_table.selectRow(1)  # ghost 占位行
    dlg._remove_selected_variable_impl()

    assert "ghost" not in pw.invalid_curves
    assert "a" in pw.curves
    dlg.close()
