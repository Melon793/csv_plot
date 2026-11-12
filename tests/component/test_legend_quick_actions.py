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

import pytest

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt, QUrl
from PySide6.QtGui import QColor, QMouseEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QColorDialog, QToolTip

from src.ui.drag_drop import COLOR_SCHEME, parse_legend_anchor
from src.ui.plot_variable_editor import PlotVariableEditorDialog

RED_NAME = QColor("red").name()  # "#ff0000"


# ---------------------------------------------------------------------------
# Step 2：apply_curve_color 下沉（刷新链与迁移前逐项一致，T14）
# ---------------------------------------------------------------------------

def test_apply_curve_color_updates_pen_and_refreshes(plot_factory):
    pw = plot_factory()
    assert pw.plot_variable("a")
    ci = pw.curves["a"]
    old_width = ci.curve.opts.get("pen").widthF()
    ci.curve._cached_pen_key = "thin_blue"  # 预置缓存：改色后必须被清除

    legend_calls = []
    orig_update = pw.update_legend_label
    pw.update_legend_label = lambda html: legend_calls.append(html)

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
    """patch plot_ui_manager.QMenu 为纯 Python 替身（替身扩展版）。

    相比 test_legend_context_menu 的版本：addAction 兼容 (text) 与
    (icon, text) 两种签名（调色板项带 QIcon 色块），action 支持
    setCheckable/setChecked。图标视觉在替身下不做断言（评审必修项 2），
    只断文案序列 + checked 状态。

    Returns:
        menus：每次弹出的菜单实例列表
    """
    menus = []

    class _FakeAction:
        """贴近 QAction 真实接口：text()/isChecked() 为方法，其余属性。

        生产代码 _exec_color_menu 会调用 chosen.text()，替身必须同签名。
        """

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
            menus.append(self)

        def addAction(self, *args):
            act = _FakeAction(args[-1])  # 末参恒为文案
            self._acts.append(act)
            return act

        def addSeparator(self):
            self._acts.append(_FakeAction(""))

        def actions(self):
            return list(self._acts)

        def exec(self, pos=None):
            if before_return is not None:
                before_return()
            if choose_text is None:
                return None
            for act in self._acts:
                if act.text() == choose_text and act.enabled:
                    return act
            return None

    import src.ui.widgets.plot_ui_manager as pum
    monkeypatch.setattr(pum, "QMenu", _FakeMenu)
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
    """点色块：显隐不变、_exec_color_menu 被调度（行为变更回归）。"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    browser = pw.legend_label

    called = []
    monkeypatch.setattr(
        browser, "_exec_color_menu", lambda *a, **k: called.append(a)
    )
    visible_before = pw.curves["a"].visible

    pw._on_legend_anchor_clicked(QUrl(f"{COLOR_SCHEME}:///{quote('a', safe='')}"))
    qapp.processEvents()  # singleShot(0) 落地

    assert called, "色块点击应调度调色板菜单"
    assert pw.curves["a"].visible == visible_before, "点色块不得切显隐"


def test_t3b_name_click_still_toggles(plot_factory):
    """点名称：显隐切换保持不变。"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    pw._on_legend_anchor_clicked(QUrl(f"curve:///{quote('a', safe='')}"))
    assert pw.curves["a"].visible is False


def test_t4_palette_preset_applies(plot_factory, monkeypatch):
    """调色板选预设色：ci.color 更新 + legend 重建。"""
    pw = plot_factory()
    assert pw.plot_variable("a")  # 首条固定 blue
    _fake_menu_env(monkeypatch, choose_text="red")

    legend_calls = []
    orig_update = pw.update_legend_label
    pw.update_legend_label = lambda html: legend_calls.append(html)

    pw.legend_label._exec_color_menu("a", QPoint(0, 0))

    assert pw.curves["a"].color == "red"
    assert legend_calls


def test_t5_current_color_checked(plot_factory, monkeypatch):
    """菜单结构：十预设色 + 「更多颜色…」；当前色 checked，其余未选。"""
    pw = plot_factory()
    assert pw.plot_variable("a")  # 首条固定 blue
    menus = _fake_menu_env(monkeypatch)

    pw.legend_label._exec_color_menu("a", QPoint(0, 0))

    acts = [a for a in menus[0].actions() if a.text()]  # 滤掉分隔线占位
    assert [a.text() for a in acts] == pw.curve_colors + ["更多颜色…"]
    checked = [a.text() for a in acts if a.isChecked()]
    assert checked == ["blue"]
    assert not any(a.checkable and not a.isChecked() for a in acts)


def test_t6_more_colors_dialog_applies(plot_factory, monkeypatch):
    """「更多颜色…」→ QColorDialog 返回合法色 → 应用。"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    _fake_menu_env(monkeypatch, choose_text="更多颜色…")

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
    _fake_menu_env(monkeypatch, choose_text="更多颜色…")
    monkeypatch.setattr(
        QColorDialog, "getColor", staticmethod(lambda *a, **k: QColor())
    )
    pw.legend_label._exec_color_menu("a", QPoint(0, 0))
    assert pw.curves["a"].color == "blue"


def test_t7_stale_curve_guard_after_menu(plot_factory, monkeypatch):
    """菜单打开期间曲线被删（reload/编辑器）：分发复查后不应用、不崩溃。"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    menus = _fake_menu_env(
        monkeypatch, choose_text="red", before_return=lambda: pw.curves.pop("a")
    )
    pw.legend_label._exec_color_menu("a", QPoint(0, 0))

    assert menus, "菜单应弹出"
    assert "a" not in pw.curves  # 二次复查拦截，无崩溃即通过


def test_t7b_stale_curve_guard_after_dialog(plot_factory, monkeypatch):
    """对话框打开期间曲线被删：同样不应用。"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    _fake_menu_env(
        monkeypatch, choose_text="更多颜色…",
        before_return=lambda: pw.curves.pop("a"),
    )
    monkeypatch.setattr(
        QColorDialog, "getColor",
        staticmethod(lambda *a, **k: QColor("#123456")),
    )
    pw.legend_label._exec_color_menu("a", QPoint(0, 0))
    assert "a" not in pw.curves


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
    """hover 色块/名称锚点：文案随锚点类型变化。"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    browser = pw.legend_label
    shown, _ = _spy_tooltip(monkeypatch)

    color_pos = _find_anchor_pos(browser, COLOR_SCHEME, "a")
    name_pos = _find_anchor_pos(browser, "curve", "a")

    browser.mouseMoveEvent(_hover_move(color_pos))
    assert shown[-1][1] == browser._TOOLTIP_COLOR
    browser.mouseMoveEvent(_hover_move(name_pos))
    assert shown[-1][1] == browser._TOOLTIP_TOGGLE


def test_tooltip_placeholder_and_blank(plot_factory, monkeypatch):
    """hover 失效占位：占位文案；hover 空白：隐藏 tooltip。"""
    from src.core.data_types import InvalidCurveRef

    pw = plot_factory()
    assert pw.plot_variable("a")
    pw.invalid_curves["ghost"] = InvalidCurveRef(var_name="ghost", color="gray")
    pw._update_header_for_curves()
    browser = pw.legend_label
    shown, hidden = _spy_tooltip(monkeypatch)

    ghost_pos = _find_anchor_pos(browser, "curve", "ghost")
    browser.mouseMoveEvent(_hover_move(ghost_pos))
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
    """任何按下 / 鼠标离开：隐藏 tooltip。"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    browser = pw.legend_label
    _, hidden = _spy_tooltip(monkeypatch)

    name_pos = _find_anchor_pos(browser, "curve", "a")
    browser.mouseMoveEvent(_hover_move(name_pos))
    hidden.clear()

    browser.mousePressEvent(_lmb(QEvent.Type.MouseButtonPress, name_pos))
    assert hidden, "按下应隐藏 tooltip"

    browser.leaveEvent(QEvent(QEvent.Type.Leave))
    assert len(hidden) == 2, "离开应再次隐藏 tooltip"
