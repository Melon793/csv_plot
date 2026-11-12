"""图例快速操作（快速改色 / 中键删除）component 测试（施工方案 §5）。

分步落地：
- Step 2：apply_curve_color 下沉 MultiCurveManager（编辑器改薄壳委托）；
- Step 3a：图例锚点拆分 color:/// + 调色板菜单；
- Step 3b：中键单击删除（含 view 层双击护栏）。

复用 plot_factory（tests/component/conftest.py）与右键菜单测试的
FakeMenu 替身模式（见 test_legend_context_menu.py 模块注释）。
"""

import re
import types

from urllib.parse import quote

import pandas as pd
import pytest

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt, QTimer, QUrl
from PySide6.QtGui import QColor, QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QColorDialog,
    QGridLayout,
    QToolTip,
)

from src.ui.drag_drop import COLOR_SCHEME, parse_legend_anchor
from src.ui.plot_variable_editor import PlotVariableEditorDialog
from src.ui.widgets.plot_ui_manager import PALETTE_COLUMNS, LegendTextBrowser
from tests.fixtures.waits import pump, settle

RED_NAME = QColor("red").name()  # "#ff0000"


# ---------------------------------------------------------------------------
# Step 2：apply_curve_color 下沉（刷新链与迁移前逐项一致，T14）
# ---------------------------------------------------------------------------

def test_apply_curve_color_updates_pen_and_refreshes(plot_factory, monkeypatch):
    pw = plot_factory()
    assert pw.plot_variable("a")
    ci = pw.curves["a"]
    old_width = ci.curve.opts.get("pen").widthF()
    ci.curve._cached_pen_key = "thin_blue"  # 预置缓存：改色后必须被清除

    legend_calls = []
    monkeypatch.setattr(
        pw, "update_legend_label", lambda html: legend_calls.append(html)
    )

    assert pw._multi_curve_manager.apply_curve_color("a", "red") is True

    assert ci.color == "red"
    pen = ci.curve.opts.get("pen")
    assert pen.color().name() == RED_NAME
    assert pen.widthF() == old_width, "改色必须保留原线宽"
    assert not hasattr(ci.curve, "_cached_pen_key"), "缓存 pen key 应被清除"
    assert legend_calls, "legend 应重建"


def test_apply_curve_color_updates_symbols(plot_factory):
    pw = plot_factory()
    assert pw.plot_variable("a")
    ci = pw.curves["a"]
    ci.curve._has_symbols = True

    assert pw._multi_curve_manager.apply_curve_color("a", "red") is True

    assert ci.curve.opts.get("symbolPen").color().name() == RED_NAME
    assert ci.curve.opts.get("symbolBrush").color().name() == RED_NAME


def test_apply_curve_color_missing_var_returns_false(plot_factory):
    pw = plot_factory()
    assert pw.plot_variable("a")
    assert pw._multi_curve_manager.apply_curve_color("ghost", "red") is False
    assert pw.curves["a"].color != "red"


def test_cursor_refresh_chain_runs(plot_factory):
    """游标刷新链：清游标项 + 重置节流时间戳（vline 可见时刷标签）。"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    pw._last_cursor_update_time = 123456.0

    cleared = []
    orig_clear = pw._clear_cursor_items
    pw._clear_cursor_items = lambda: (cleared.append(1), orig_clear())[0]

    assert pw._multi_curve_manager.apply_curve_color("a", "red") is True
    assert cleared == [1]
    assert pw._last_cursor_update_time == 0


# ---------------------------------------------------------------------------
# Step 2：编辑器薄壳委托（行为零变化，表格色块按返回值刷新）
# ---------------------------------------------------------------------------

@pytest.fixture()
def editor_env(plot_factory, qapp):
    """带宿主 loader 的编辑器对话框（对齐 test_plot_variable_editor_refresh 夹具）"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    host = pw.window()
    host.loader = types.SimpleNamespace(var_names=["a"], units={})
    host.data_validity = {}
    dlg = PlotVariableEditorDialog(pw)
    qapp.processEvents()
    yield dlg, pw
    dlg.deleteLater()


def test_editor_apply_color_delegates_to_manager(editor_env):
    dlg, pw = editor_env
    assert dlg._apply_color_to_curve("a", "red") is True
    assert pw.curves["a"].color == "red"
    # 不存在的变量：返回 False，无任何状态变化
    assert dlg._apply_color_to_curve("ghost", "red") is False
    assert "ghost" not in pw.curves


def test_editor_set_variable_color_updates_table_on_success(
    editor_env, monkeypatch
):
    dlg, pw = editor_env
    dlg.var_table.selectRow(0)
    var_item = dlg.var_table.item(0, 1)
    assert var_item is not None
    assert var_item.data(Qt.ItemDataRole.UserRole) == "a"

    monkeypatch.setattr(
        QColorDialog, "getColor", staticmethod(lambda *a, **k: QColor("#123456"))
    )
    dlg.set_variable_color()

    assert pw.curves["a"].color == "#123456"
    color_widget = dlg.var_table.cellWidget(0, 2)
    assert color_widget is not None
    assert "#123456" in color_widget.styleSheet()

    # 无效颜色（isValid 为 False）：颜色与表格样式均不变
    monkeypatch.setattr(
        QColorDialog, "getColor", staticmethod(lambda *a, **k: QColor())
    )
    dlg.set_variable_color()
    assert pw.curves["a"].color == "#123456"


# ---------------------------------------------------------------------------
# Step 3a：图例锚点拆分 + 调色板菜单
# ---------------------------------------------------------------------------

def _fake_menu_env(monkeypatch, choose_text=None, before_return=None):
    """patch plot_ui_manager.QMenu 为纯 Python 替身（右键/中键菜单用）。

    addAction 兼容 (text) 与 (action 对象) 两种签名，action 支持
    setEnabled/setCheckable/setChecked。调色板网格菜单不走本替身
    （见 _fake_palette_env），避免替身假绿。

    Returns:
        menus：每次弹出的菜单实例列表
    """
    menus = []

    class _FakeAction:
        """贴近 QAction 真实接口：text()/isChecked() 为方法，其余属性。"""

        def __init__(self, text):
            self._text = text
            self.enabled = True
            self.checkable = False
            self._checked = False

        def text(self):
            return self._text

        def setEnabled(self, enabled):
            self.enabled = bool(enabled)

        def setCheckable(self, checkable):
            self.checkable = bool(checkable)

        def setChecked(self, checked):
            self._checked = bool(checked)

        def isChecked(self):
            return self._checked

    class _FakeMenu:
        def __init__(self, parent=None):
            self._acts = []
            self._subs = []
            self.deleted = False
            menus.append(self)

        def addAction(self, *args):
            act = _FakeAction(args[-1])  # 末参恒为文案
            self._acts.append(act)
            return act

        def addSeparator(self):
            self._acts.append(_FakeAction(""))

        def addMenu(self, title):
            # 子菜单在父菜单 actions() 里占一个同名项；exec 可选中
            # 子菜单内动作（与真实 QMenu.exec 返回任意层级 action 一致）
            self._acts.append(_FakeAction(title))
            sub = _FakeMenu()
            self._subs.append(sub)
            return sub

        def actions(self):
            return list(self._acts)

        def deleteLater(self):
            self.deleted = True

        def close(self):
            pass

        def _iter_all_acts(self):
            for act in self._acts:
                yield act
            for sub in self._subs:
                yield from sub._iter_all_acts()

        def exec(self, pos=None):
            if before_return is not None:
                before_return()
            if choose_text is None:
                return None
            for act in self._iter_all_acts():
                if act.text() == choose_text and act.enabled:
                    return act
            return None

    import src.ui.widgets.plot_ui_manager as pum
    monkeypatch.setattr(pum, "QMenu", _FakeMenu)
    return menus


def _fake_palette_env(
    monkeypatch, pick_color=None, choose_more=False, before_return=None
):
    """patch ``LegendTextBrowser._build_color_palette_menu`` 为替身。

    行为用例只关心 ``exec`` 的分发结果（选色 / 更多颜色… / 取消），
    网格结构与真实点击由 T5a/T5b/T5d 用真实 builder 覆盖。

    Returns:
        menus：每次构建的菜单替身列表
    """
    menus = []

    class _FakeAction:
        def __init__(self, text):
            self._text = text
            self.enabled = True

        def text(self):
            return self._text

        def setEnabled(self, enabled):
            self.enabled = bool(enabled)

    class _FakeMenu:
        def __init__(self, parent=None):
            self.deleted = False
            self.closed = False
            self.on_pick = None
            self._more = None
            self._acts = []
            menus.append(self)

        def deleteLater(self):
            self.deleted = True

        def close(self):
            self.closed = True

        def exec(self, pos=None):
            if before_return is not None:
                before_return()
            if pick_color is not None:
                self.on_pick(pick_color)
                return None
            if choose_more:
                return self._more
            return None

    def _fake_build(self, current, on_pick):
        menu = _FakeMenu()
        menu.on_pick = on_pick
        menu._more = _FakeAction("更多颜色…")
        menu._acts.append(menu._more)
        return menu, menu._more

    monkeypatch.setattr(
        LegendTextBrowser, "_build_color_palette_menu", _fake_build
    )
    return menus


def _find_anchor_pos(browser, scheme, var_name):
    """扫描 legend viewport 找到指定 scheme 锚点命中坐标。"""
    href = f"{scheme}:///{quote(var_name, safe='')}"
    for y in range(1, 40, 2):
        for x in range(1, 200, 2):
            if browser.anchorAt(QPoint(x, y)) == href:
                return QPoint(x, y)
    pytest.fail(f"legend 中未找到 {scheme}:/// 锚点坐标：{var_name}")


def _mouse_event(event_type, pos, button, buttons):
    f = QPointF(pos)
    return QMouseEvent(
        event_type, f, f, QPointF(0.0, 0.0), button, buttons,
        Qt.KeyboardModifier.NoModifier,
    )


def _lmb(event_type, pos):
    return _mouse_event(event_type, pos,
                        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton)


def _lmb_move(pos):
    return _mouse_event(QEvent.Type.MouseMove, pos,
                        Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton)


def _rmb(event_type, pos):
    return _mouse_event(event_type, pos,
                        Qt.MouseButton.RightButton, Qt.MouseButton.RightButton)


def _mid(event_type, pos):
    return _mouse_event(event_type, pos,
                        Qt.MouseButton.MiddleButton, Qt.MouseButton.MiddleButton)


def test_t1_legend_html_anchor_pairs(plot_factory):
    """可见/隐藏条目各产出 color:/// + curve:/// 一对锚点，顺序稳定。"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    assert pw.plot_variable("b")
    pw.toggle_curve_visibility_by_name("b")  # b → 隐藏

    html = pw._multi_curve_manager._build_legend_html()
    parsed = [
        parse_legend_anchor(h)
        for h in re.findall(r"href='([^']+)'", html)
    ]
    assert parsed == [
        ("color", "a"), ("toggle", "a"),
        ("color", "b"), ("toggle", "b"),
    ]


def test_t2_invalid_placeholder_single_anchor(plot_factory):
    """失效占位条目保持单锚点 curve:///，无 color 锚点。"""
    from src.core.data_types import InvalidCurveRef

    pw = plot_factory()
    pw.invalid_curves["ghost"] = InvalidCurveRef(var_name="ghost", color="gray")

    html = pw._multi_curve_manager._build_legend_html()
    parsed = [
        parse_legend_anchor(h)
        for h in re.findall(r"href='([^']+)'", html)
    ]
    assert parsed == [("toggle", "ghost")]


def test_t3_color_block_click_does_not_toggle(plot_factory, monkeypatch, qapp):
    """色块单击（真实 press/release→anchorClicked 链路）：显隐不变、调度调色板。"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    pw.setFixedSize(430, 300)
    pw.show()
    qapp.processEvents()
    browser = pw.legend_label

    called = []
    monkeypatch.setattr(
        browser, "_exec_color_menu", lambda *a, **k: called.append(a)
    )
    visible_before = pw.curves["a"].visible
    pos = _find_anchor_pos(browser, COLOR_SCHEME, "a")

    QTest.mouseClick(
        browser.viewport(), Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier, pos,
    )
    qapp.processEvents()  # singleShot(0) 落地

    assert called, "色块点击应经由 anchorClicked 调度调色板菜单"
    assert pw.curves["a"].visible == visible_before, "点色块不得切显隐"


def test_t3b_name_click_still_toggles(plot_factory):
    """点名称：显隐切换保持不变。"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    pw._on_legend_anchor_clicked(QUrl(f"curve:///{quote('a', safe='')}"))
    assert pw.curves["a"].visible is False


def test_t4_palette_preset_applies(plot_factory, monkeypatch):
    """调色板选预设色：ci.color 更新 + legend 重建 + 菜单关闭/释放。"""
    pw = plot_factory()
    assert pw.plot_variable("a")  # 首条固定为色板首色
    menus = _fake_palette_env(monkeypatch, pick_color="red")

    legend_calls = []
    monkeypatch.setattr(
        pw, "update_legend_label", lambda html: legend_calls.append(html)
    )

    pw.legend_label._exec_color_menu("a", QPoint(0, 0))

    assert pw.curves["a"].color == "red"
    assert legend_calls
    assert menus[0].closed is True, "选色后应关闭菜单"
    assert menus[0].deleted is True, "QMenu 应经 deleteLater 释放"


def test_t5a_palette_grid_structure(plot_factory):
    """真实 builder：10 个色块按行优先 5×2 排布，当前色唯一双环标记。"""
    pw = plot_factory()
    assert pw.plot_variable("a")  # 首条固定为色板首色
    first_color = pw.curve_colors[0]
    menu, act_more = pw.legend_label._build_color_palette_menu(
        first_color, lambda name: None
    )
    try:
        grid_host = menu.actions()[0].defaultWidget()
        assert grid_host is not None, "网格应以 QWidgetAction 内嵌"
        grid = grid_host.layout()
        assert isinstance(grid, QGridLayout)

        swatches = [grid.itemAt(i).widget() for i in range(grid.count())]
        assert [s.color_name for s in swatches] == pw.curve_colors
        # 行优先：逐项校验 (row, col)，比只断行列数更能表达 6×2
        for index, swatch in enumerate(swatches):
            assert grid.getItemPosition(index)[:2] == (
                index // PALETTE_COLUMNS, index % PALETTE_COLUMNS
            ), f"{swatch.color_name} 网格位置错误"
        assert grid.columnCount() == PALETTE_COLUMNS
        assert grid.rowCount() == 2

        assert act_more.text() == "更多颜色…"

        selected = [s for s in swatches if s.is_selected()]
        assert [s.color_name for s in selected] == [first_color]
        for swatch in swatches:
            assert swatch.toolTip(), "每个色块应有中文 tooltip"
            assert re.search(r"[\u4e00-\u9fff]", swatch.toolTip())
        assert selected[0].toolTip().endswith("（当前）")
    finally:
        menu.deleteLater()


def test_t5b_swatch_click_and_keyboard(plot_factory, qapp):
    """真实 builder：鼠标点击与 Enter 键都能选中色块（菜单已 popup）。"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    picked = []
    second_color = pw.curve_colors[1]
    menu, _ = pw.legend_label._build_color_palette_menu(
        pw.curve_colors[0], picked.append
    )
    grid_host = menu.actions()[0].defaultWidget()
    grid = grid_host.layout()
    second_swatch = next(
        grid.itemAt(i).widget()
        for i in range(grid.count())
        if grid.itemAt(i).widget().color_name == second_color
    )

    menu.popup(QPoint(50, 50))
    qapp.processEvents()
    try:
        QTest.mouseClick(second_swatch, Qt.MouseButton.LeftButton)
        qapp.processEvents()
        assert picked == [second_color]

        QTest.keyClick(second_swatch, Qt.Key.Key_Return)
        qapp.processEvents()
        assert picked == [second_color, second_color]
    finally:
        menu.close()
        menu.deleteLater()


def test_t5c_cancel_is_noop(plot_factory, monkeypatch):
    """取消（exec 返回 None，等价 Esc/点菜单外）：不改色、不弹对话框。"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    _fake_palette_env(monkeypatch)  # 既不选色也不选「更多颜色…」

    dialog_calls = []
    monkeypatch.setattr(
        QColorDialog,
        "getColor",
        staticmethod(lambda *a, **k: dialog_calls.append(1) or QColor("#123456")),
    )

    pw.legend_label._exec_color_menu("a", QPoint(0, 0))

    assert pw.curves["a"].color == pw.curve_colors[0]
    assert not dialog_calls


def test_t5d_real_exec_click_applies(plot_factory, monkeypatch):
    """真实 QMenu.exec + 点击色块：exec 正常返回并应用颜色（关闭/回收路径）。"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    built = []
    orig_build = LegendTextBrowser._build_color_palette_menu

    def _build_and_click(self, current, on_pick):
        menu, act_more = orig_build(self, current, on_pick)
        built.append(menu)

        def _click_second():
            if not menu.isVisible():
                return  # exec 提前返回、菜单已释放时不得再摸控件
            grid = menu.actions()[0].defaultWidget().layout()
            for i in range(grid.count()):
                swatch = grid.itemAt(i).widget()
                if swatch.color_name == pw.curve_colors[1]:
                    QTest.mouseClick(swatch, Qt.MouseButton.LeftButton)
                    return

        QTimer.singleShot(0, _click_second)
        return menu, act_more

    monkeypatch.setattr(
        LegendTextBrowser, "_build_color_palette_menu", _build_and_click
    )

    pw.legend_label._exec_color_menu("a", QPoint(20, 20))

    assert pw.curves["a"].color == pw.curve_colors[1]
    assert built, "真实菜单应被构建"


def test_t6_more_colors_dialog_applies(plot_factory, monkeypatch):
    """「更多颜色…」→ QColorDialog 返回合法色 → 应用。"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    _fake_palette_env(monkeypatch, choose_more=True)

    captured = {}

    def _fake_get_color(initial=None, parent=None, title="", *a, **k):
        captured.update(initial=initial.name(), title=title, parent=parent)
        return QColor("#123456")

    monkeypatch.setattr(QColorDialog, "getColor", staticmethod(_fake_get_color))
    pw.legend_label._exec_color_menu("a", QPoint(0, 0))

    assert pw.curves["a"].color == "#123456"
    assert captured["title"] == "选择颜色"
    assert captured["parent"] is pw


def test_t6b_more_colors_invalid_noop(plot_factory, monkeypatch):
    """「更多颜色…」→ 无效色（取消）：颜色不变。"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    _fake_palette_env(monkeypatch, choose_more=True)
    monkeypatch.setattr(
        QColorDialog, "getColor", staticmethod(lambda *a, **k: QColor())
    )
    pw.legend_label._exec_color_menu("a", QPoint(0, 0))
    assert pw.curves["a"].color == pw.curve_colors[0]


def test_t7_stale_curve_guard_after_menu(plot_factory, monkeypatch):
    """菜单打开期间曲线被删（reload/编辑器）：分发复查后不应用、不崩溃。"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    menus = _fake_palette_env(
        monkeypatch, pick_color="red", before_return=lambda: pw.curves.pop("a")
    )
    pw.legend_label._exec_color_menu("a", QPoint(0, 0))

    assert menus, "菜单应弹出"
    assert "a" not in pw.curves  # 二次复查拦截，无崩溃即通过


def test_t7b_stale_curve_guard_after_dialog(plot_factory, monkeypatch):
    """对话框打开期间曲线被删：同样不应用（二次复查口径）。"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    _fake_palette_env(monkeypatch, choose_more=True)

    def _fake_get_color(*a, **k):
        pw.curves.pop("a", None)  # 模拟对话框阻塞期间被 reload/删除
        return QColor("#123456")

    monkeypatch.setattr(QColorDialog, "getColor", staticmethod(_fake_get_color))
    pw.legend_label._exec_color_menu("a", QPoint(0, 0))
    assert "a" not in pw.curves


def test_t7c_menu_released_and_dialog_skipped_when_curve_deleted(
    plot_factory, monkeypatch
):
    """菜单打开期间变量被删：菜单仍释放，且不再弹颜色对话框（#6）。"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    menus = _fake_palette_env(
        monkeypatch, choose_more=True,
        before_return=lambda: pw.curves.pop("a"),
    )
    dialog_calls = []
    monkeypatch.setattr(
        QColorDialog, "getColor",
        staticmethod(lambda *a, **k: dialog_calls.append(1) or QColor("#123456")),
    )

    pw.legend_label._exec_color_menu("a", QPoint(0, 0))

    assert menus[0].deleted is True, "QMenu 应经 deleteLater 释放"
    assert not dialog_calls, "变量已删时不应再弹颜色对话框"


def test_t12_drag_from_color_block(plot_factory, monkeypatch, qapp):
    """色块上按住拖出 ≥ startDragDistance：宽松解析生效，正常发起拖拽。"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    browser = pw.legend_label
    pos = _find_anchor_pos(browser, COLOR_SCHEME, "a")

    started = []
    monkeypatch.setattr(browser, "_start_var_drag", lambda: started.append(1))

    browser.mousePressEvent(_lmb(QEvent.Type.MouseButtonPress, pos))
    moved = pos + QPoint(QApplication.startDragDistance() + 5, 0)
    browser.mouseMoveEvent(_lmb_move(moved))
    assert started, "色块上拖拽应正常发起"


def test_t12b_drag_still_works_from_name(plot_factory, monkeypatch, qapp):
    """名称上拖拽回归：宽松解析后仍正常发起。"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    browser = pw.legend_label
    pos = _find_anchor_pos(browser, "curve", "a")

    started = []
    monkeypatch.setattr(browser, "_start_var_drag", lambda: started.append(1))

    browser.mousePressEvent(_lmb(QEvent.Type.MouseButtonPress, pos))
    moved = pos + QPoint(QApplication.startDragDistance() + 5, 0)
    browser.mouseMoveEvent(_lmb_move(moved))
    assert started


def test_t13_right_click_on_color_block_opens_menu(plot_factory, monkeypatch, qapp):
    """色块上右键 press/release：宽松解析生效，弹出与名称右键相同菜单。"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    browser = pw.legend_label
    pos = _find_anchor_pos(browser, COLOR_SCHEME, "a")

    menus = _fake_menu_env(monkeypatch, choose_text="复制变量名")
    QApplication.clipboard().setText("SENTINEL")
    browser.mousePressEvent(_rmb(QEvent.Type.MouseButtonPress, pos))
    browser.mouseReleaseEvent(_rmb(QEvent.Type.MouseButtonRelease, pos))
    qapp.processEvents()

    assert QApplication.clipboard().text() == "a"
    acts = [a.text() for a in menus[0].actions() if a.text()]
    assert acts[0] == "仅显示此变量"
    assert acts[-1] == "绘图变量编辑器"


# ---------------------------------------------------------------------------
# Step 3b：中键单击删除（含 view 层双击护栏）
# ---------------------------------------------------------------------------

def test_t8_middle_click_deletes_curve(plot_factory, monkeypatch, qapp):
    """中键 release 在锚点上：删除曲线 + mark_stats 刷新 + 状态栏播报。"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    assert pw.plot_variable("b")
    browser = pw.legend_label
    pos = _find_anchor_pos(browser, "curve", "a")  # 宽松解析：中键名称锚点命中

    refresh = []
    monkeypatch.setattr(
        pw.window().layout_manager,
        "request_mark_stats_refresh",
        lambda *a, **k: refresh.append(1),
    )
    msgs = []
    pw.window()._show_status_message = lambda msg: msgs.append(msg)

    browser.mousePressEvent(_mid(QEvent.Type.MouseButtonPress, pos))
    browser.mouseReleaseEvent(_mid(QEvent.Type.MouseButtonRelease, pos))
    qapp.processEvents()  # singleShot(0) 落地

    assert "a" not in pw.curves
    assert "b" in pw.curves
    assert refresh, "request_mark_stats_refresh 应被调用"
    assert msgs == ["已移除 a"]


def test_t9_middle_click_deletes_placeholder(plot_factory, qapp):
    """中键在失效占位上：删除占位 + curves_changed emit。"""
    from src.core.data_types import InvalidCurveRef

    pw = plot_factory()
    pw.invalid_curves["ghost"] = InvalidCurveRef(var_name="ghost", color="gray")
    pw._update_header_for_curves()  # 渲染占位条目
    browser = pw.legend_label
    pos = _find_anchor_pos(browser, "curve", "ghost")

    fired = []
    pw.curves_changed.connect(lambda: fired.append(1))
    browser.mousePressEvent(_mid(QEvent.Type.MouseButtonPress, pos))
    browser.mouseReleaseEvent(_mid(QEvent.Type.MouseButtonRelease, pos))
    qapp.processEvents()

    assert "ghost" not in pw.invalid_curves
    assert fired == [1]


def test_t10_middle_drag_cancels_delete(plot_factory, monkeypatch, qapp):
    """中键位移 ≥ startDragDistance：不删除。"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    browser = pw.legend_label
    pos = _find_anchor_pos(browser, "curve", "a")

    called = []
    monkeypatch.setattr(
        browser, "_exec_middle_delete", lambda *a, **k: called.append(a)
    )

    far = pos + QPoint(QApplication.startDragDistance() + 10, 0)
    browser.mousePressEvent(_mid(QEvent.Type.MouseButtonPress, pos))
    browser.mouseReleaseEvent(_mid(QEvent.Type.MouseButtonRelease, far))
    qapp.processEvents()

    assert not called
    assert "a" in pw.curves


def test_t10b_middle_double_click_no_double_delete(plot_factory, qapp):
    """中键连击两次：第二次 release 因二次复查归属静默 no-op（幂等）。"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    assert pw.plot_variable("b")
    browser = pw.legend_label
    pos = _find_anchor_pos(browser, "curve", "a")
    msgs = []
    pw.window()._show_status_message = lambda msg: msgs.append(msg)

    # 双击事件序列的两次 press/release 连发（删除在 singleShot 中落地，
    # 第二次调度转回事件循环时变量已删 → 静默）
    browser.mousePressEvent(_mid(QEvent.Type.MouseButtonPress, pos))
    browser.mouseReleaseEvent(_mid(QEvent.Type.MouseButtonRelease, pos))
    browser.mousePressEvent(_mid(QEvent.Type.MouseButtonPress, pos))
    browser.mouseReleaseEvent(_mid(QEvent.Type.MouseButtonRelease, pos))
    qapp.processEvents()

    assert "a" not in pw.curves
    assert "b" in pw.curves
    assert msgs == ["已移除 a"], "第二次删除应被二次复查拦截"


def test_t11_middle_click_blank_no_action(plot_factory, monkeypatch, qapp):
    """中键在无锚点空白处：无动作。"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    browser = pw.legend_label

    blank = None
    for y in range(1, 40):
        for x in range(1, 200, 4):
            if browser.anchorAt(QPoint(x, y)) == "":
                blank = QPoint(x, y)
                break
        if blank is not None:
            break
    assert blank is not None

    called = []
    monkeypatch.setattr(
        browser, "_exec_middle_delete", lambda *a, **k: called.append(a)
    )
    browser.mousePressEvent(_mid(QEvent.Type.MouseButtonPress, blank))
    browser.mouseReleaseEvent(_mid(QEvent.Type.MouseButtonRelease, blank))
    qapp.processEvents()

    assert not called
    assert "a" in pw.curves


def _inject_middle_double_click(pw, vp_pos, qapp):
    """经真实链路（QTest→viewport→view→scene→proxy）注入中键双击序列。"""
    QTest.mousePress(
        pw.viewport(), Qt.MouseButton.MiddleButton,
        Qt.KeyboardModifier.NoModifier, vp_pos,
    )
    QTest.mouseRelease(
        pw.viewport(), Qt.MouseButton.MiddleButton,
        Qt.KeyboardModifier.NoModifier, vp_pos,
    )
    QTest.mouseDClick(
        pw.viewport(), Qt.MouseButton.MiddleButton,
        Qt.KeyboardModifier.NoModifier, vp_pos,
    )
    QTest.mouseRelease(
        pw.viewport(), Qt.MouseButton.MiddleButton,
        Qt.KeyboardModifier.NoModifier, vp_pos,
    )
    qapp.processEvents()


def test_t0b_legend_middle_dblclick_does_not_clear_plot(plot_factory, qapp):
    """T0 探针回归：图例锚点上中键双击不清空本格（view 层护栏生效）。

    语义上双击 = 两次单击删除 = 删除一次（幂等）；关键是不触发
    clear_current_plot。
    """
    pw = plot_factory()
    assert pw.plot_variable("a")
    assert pw.plot_variable("b")
    pw.show()
    qapp.processEvents()

    browser = pw.legend_label
    proxy = browser.graphicsProxyWidget()
    local = _find_anchor_pos(browser, "curve", "a")
    vp_pos = pw.mapFromScene(proxy.mapToScene(QPointF(local)))

    clear_calls = []
    pw.clear_current_plot = lambda: clear_calls.append(1)

    _inject_middle_double_click(pw, vp_pos, qapp)

    assert not clear_calls, "图例上中键双击不得触发 clear_current_plot"
    assert set(pw.curves) == {"b"}, "双击 = 删除一次（幂等），b 保留"


def test_t0d_guard_uses_press_hit_when_geometry_shrinks(plot_factory, qapp):
    """护栏取「按下命中 ∪ 当前位置命中」：几何失效时仍不清空（#1 回归）。

    直接构造「按下在图例内、dblclick 时刻当前位置在图例区外」的状态，
    固定 view 层判定逻辑本身（不依赖布局能否复现变矮）。
    """
    pw = plot_factory()
    assert pw.plot_variable("a")
    pw.show()
    qapp.processEvents()

    center = pw.view_box.mapToScene(pw.view_box.boundingRect().center())
    vp = pw.mapFromScene(center)
    assert not pw._is_pos_over_legend(vp)  # 前置：当前位置在图例区外

    pw._mid_press_over_legend = True  # 模拟「按下在图例内、其后图例变矮」
    clear_calls = []
    pw.clear_current_plot = lambda: clear_calls.append(1)

    ev = QMouseEvent(
        QEvent.Type.MouseButtonDblClick,
        QPointF(vp), QPointF(vp),
        Qt.MouseButton.MiddleButton, Qt.MouseButton.MiddleButton,
        Qt.KeyboardModifier.NoModifier,
    )
    pw.mouseDoubleClickEvent(ev)
    qapp.processEvents()

    assert not clear_calls, "按下命中图例时应吞掉双击，不得清空"
    assert pw._mid_press_over_legend is False, "手势状态应被消费"


def test_t0e_multiline_legend_middle_dblclick_does_not_clear(plot_factory, qapp, monkeypatch):
    """多行图例真实序列：删除落地使图例变矮后，同一坐标的 dblclick 不清空（#1）。

    前置条件由实测构造（tmp/2026-11-12/probe_multiline_legend.py）：
    5 条 20 字符长名 → 图例 3 行；删末尾条目后降到 2 行，末尾条目
    原坐标落到矩形外。
    """
    names = [f"chn{i:02d}_" + "x" * 20 for i in range(1, 12)]
    df = pd.DataFrame({n: [1.0, 2.0, 3.0] for n in names})
    pw = plot_factory(df)
    # 固定窄宽度：确保长名条目折行成多行（宽度随父布局变化时结论不稳）
    pw.setFixedSize(430, 420)
    pw.show()
    qapp.processEvents()

    browser = pw.legend_label
    proxy = browser.graphicsProxyWidget()

    # 自适应定位「把图例撑出新行」的那一条：删掉它高度必然回退
    # （不依赖具体字体宽度/窗口尺寸，避免前置条件随环境漂移）
    prev_h = browser.height()
    target = None
    plotted = []
    for n in names:
        assert pw.plot_variable(n)
        qapp.processEvents()
        h = browser.height()
        if h > prev_h:
            target = n
            prev_h = h
            break
        plotted.append(n)
    assert target is not None, "未能构造出跨行条目（图例始终单行）"
    assert plotted, "跨行条目应是第 2 条及以后"

    last = target
    href = f"curve:///{quote(last, safe='')}"
    local = None
    for y in range(browser.height() - 1, -1, -1):
        for x in range(0, browser.width(), 3):
            if browser.anchorAt(QPoint(x, y)) == href:
                local = QPoint(x, y)
                break
        if local is not None:
            break
    assert local is not None, "未定位到末尾条目锚点"
    vp_pos = pw.mapFromScene(proxy.mapToScene(QPointF(local)))

    h_before = browser.height()
    assert pw._is_pos_over_legend(vp_pos)

    clear_calls = []
    pw.clear_current_plot = lambda: clear_calls.append(1)

    # 真实单击：press → release → 删除在 singleShot 中落地
    QTest.mousePress(
        pw.viewport(), Qt.MouseButton.MiddleButton,
        Qt.KeyboardModifier.NoModifier, vp_pos,
    )
    QTest.mouseRelease(
        pw.viewport(), Qt.MouseButton.MiddleButton,
        Qt.KeyboardModifier.NoModifier, vp_pos,
    )
    qapp.processEvents()

    assert last not in pw.curves
    settle(30)  # 文档重排 + 图例高度联动在事件循环内完成
    assert browser.height() < h_before, (
        f"前置条件：删除后图例应变矮（h={browser.height()}，"
        f"before={h_before}，width={browser.width()}）"
    )

    # 模拟「dblclick 时当前位置已不在图例内」：删除改变了 header/plot 布局，
    # 实际映射结果随环境漂移（本机实测仍可能命中），故显式固定该前置，
    # 让断言只检验修复点本身——press 瞬间的命中必须留存下来并生效。
    monkeypatch.setattr(pw, "_is_pos_over_legend", lambda *a, **k: False)

    # 双击第二拍：同一坐标
    QTest.mouseDClick(
        pw.viewport(), Qt.MouseButton.MiddleButton,
        Qt.KeyboardModifier.NoModifier, vp_pos,
    )
    qapp.processEvents()

    assert not clear_calls, "图例上中键双击不得清空整格"
    assert pw._mid_press_over_legend is False, "按下状态应被双击消费"
    assert set(pw.curves) == set(plotted)


def test_t0c_plot_area_middle_dblclick_still_clears(plot_factory, qapp):
    """护栏不误伤：绘图区中键双击仍清空本格。"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    pw.show()
    qapp.processEvents()

    center_scene = pw.view_box.mapToScene(pw.view_box.boundingRect().center())
    vp_pos = pw.mapFromScene(center_scene)

    clear_calls = []
    pw.clear_current_plot = lambda: clear_calls.append(1)

    _inject_middle_double_click(pw, vp_pos, qapp)

    assert clear_calls == [1], "绘图区中键双击清空行为保持不变"


# ---------------------------------------------------------------------------
# Step 4：tooltip 手势引导（§2.8）
# ---------------------------------------------------------------------------

def _hover_move(pos):
    return _mouse_event(QEvent.Type.MouseMove, pos,
                        Qt.MouseButton.NoButton, Qt.MouseButton.NoButton)


def _spy_tooltip(monkeypatch):
    """替换 QToolTip 类级静态方法（showText/hideText 为类访问，直接生效）"""
    shown, hidden = [], []

    def _fake_show(pos, text, *a, **k):
        shown.append((pos, text))

    monkeypatch.setattr(QToolTip, "showText", staticmethod(_fake_show))
    monkeypatch.setattr(QToolTip, "hideText", staticmethod(lambda *a, **k: hidden.append(1)))
    return shown, hidden


def test_tooltip_by_anchor_type(plot_factory, monkeypatch):
    """hover 色块/名称锚点并停留：文案随锚点类型变化。"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    browser = pw.legend_label
    browser._tooltip_timer.setInterval(10)  # 测试加速，不真等 1.2s
    shown, _ = _spy_tooltip(monkeypatch)

    color_pos = _find_anchor_pos(browser, COLOR_SCHEME, "a")
    name_pos = _find_anchor_pos(browser, "curve", "a")

    browser.mouseMoveEvent(_hover_move(color_pos))
    pump(60)
    assert shown[-1][1] == browser._TOOLTIP_COLOR
    browser.mouseMoveEvent(_hover_move(name_pos))
    pump(60)
    assert shown[-1][1] == browser._TOOLTIP_TOGGLE


def test_tooltip_delayed_until_dwell(plot_factory, monkeypatch):
    """未停留到延迟时长不显示；到点才显示；同锚点内移动不重复弹出。"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    browser = pw.legend_label
    browser._tooltip_timer.setInterval(50)
    shown, _ = _spy_tooltip(monkeypatch)
    name_pos = _find_anchor_pos(browser, "curve", "a")

    browser.mouseMoveEvent(_hover_move(name_pos))
    pump(20)
    assert not shown, "未到延迟时长不应显示"
    assert browser._tooltip_timer.isActive(), "应处于延迟计时中"

    pump(120)
    assert shown and shown[-1][1] == browser._TOOLTIP_TOGGLE

    count = len(shown)
    browser.mouseMoveEvent(_hover_move(name_pos + QPoint(1, 0)))
    pump(80)
    assert len(shown) == count, "同一锚点内移动不应重复弹出"


def test_tooltip_placeholder_and_blank(plot_factory, monkeypatch):
    """hover 失效占位：占位文案；hover 空白：隐藏 tooltip。"""
    from src.core.data_types import InvalidCurveRef

    pw = plot_factory()
    assert pw.plot_variable("a")
    pw.invalid_curves["ghost"] = InvalidCurveRef(var_name="ghost", color="gray")
    pw._update_header_for_curves()
    browser = pw.legend_label
    browser._tooltip_timer.setInterval(10)
    shown, hidden = _spy_tooltip(monkeypatch)

    ghost_pos = _find_anchor_pos(browser, "curve", "ghost")
    browser.mouseMoveEvent(_hover_move(ghost_pos))
    pump(60)
    assert shown[-1][1] == browser._TOOLTIP_INVALID

    blank = None
    for y in range(1, 40):
        for x in range(1, 200, 4):
            if browser.anchorAt(QPoint(x, y)) == "":
                blank = QPoint(x, y)
                break
        if blank is not None:
            break
    assert blank is not None
    browser.mouseMoveEvent(_hover_move(blank))
    assert hidden, "空白处应隐藏 tooltip"


def test_tooltip_hidden_on_press_and_leave(plot_factory, monkeypatch):
    """任何按下 / 鼠标离开：隐藏 tooltip 并取消未到点的计时。"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    browser = pw.legend_label
    _, hidden = _spy_tooltip(monkeypatch)

    name_pos = _find_anchor_pos(browser, "curve", "a")
    browser.mouseMoveEvent(_hover_move(name_pos))
    assert browser._tooltip_timer.isActive()
    hidden.clear()

    browser.mousePressEvent(_lmb(QEvent.Type.MouseButtonPress, name_pos))
    assert hidden, "按下应隐藏 tooltip"
    assert not browser._tooltip_timer.isActive(), "按下应取消延迟计时"

    browser.leaveEvent(QEvent(QEvent.Type.Leave))
    assert len(hidden) == 2, "离开应再次隐藏 tooltip"
