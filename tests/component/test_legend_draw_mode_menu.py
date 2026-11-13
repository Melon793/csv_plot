"""legend 右键菜单「样式」面板组件测试（offscreen）。

右键菜单原「绘制方式」「数据点」两个子菜单合并为单一「样式」入口，
内嵌与左键色块快捷面板共享的 _build_style_panel_widget（形态三：
色板 + 两组图标分段控件）。应用统一走
MultiCurveManager.apply_curve_draw_mode / apply_curve_symbol_policy，
菜单关闭后经 _apply_curve_panel_picks 集中执行。

替身策略与 test_legend_context_menu 相同：patch plot_ui_manager.QMenu
为 QObject 化 FakeMenu（支持 addMenu 子菜单与 deleteLater；子菜单会
作为 QWidgetAction 的 parent，须为 QObject），面板本体为真实
widget，offscreen 下用 QTest 点击分段按钮驱动。
"""

from PySide6.QtCore import QObject, QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLabel, QPushButton, QWidgetAction

import pytest

from src.core.data_types import (
    DRAW_MODE_AUTO,
    DRAW_MODE_STEP_POST,
    SYMBOL_POLICY_AUTO,
    SYMBOL_POLICY_OFF,
)
from src.ui.widgets.plot_ui_manager import _SegmentedControl


def _fake_menu_env(monkeypatch, choose_text=None, before_return=None):
    """patch plot_ui_manager.QMenu 为支持子菜单的 QObject 替身。"""
    menus = []

    class _FakeAction:
        def __init__(self, text):
            self.text = text
            self.enabled = True
            self.checked = False

        def setEnabled(self, enabled):
            self.enabled = bool(enabled)

        def setCheckable(self, _checkable):
            pass

        def setChecked(self, checked):
            self.checked = bool(checked)

    class _FakeMenu(QObject):
        """QObject 子类：「样式」子菜单作为 QWidgetAction 的 parent
        （真实 QMenu 语义），替身须可作 QObject 父节点。"""

        def __init__(self, parent=None):
            super().__init__()
            self._acts = []
            self._subs = []
            menus.append(self)

        def addAction(self, *args):
            act = _FakeAction(args[-1])
            self._acts.append(act)
            return act

        def addSeparator(self):
            self._acts.append(_FakeAction(""))

        def addMenu(self, title):
            self._acts.append(_FakeAction(title))
            sub = _FakeMenu()
            self._subs.append(sub)
            return sub

        def actions(self):
            return list(self._acts)

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
                if act.text == choose_text and act.enabled:
                    return act
            return None

        def close(self):
            pass

        def deleteLater(self):
            pass

    import src.ui.widgets.plot_ui_manager as pum
    monkeypatch.setattr(pum, "QMenu", _FakeMenu)
    return menus


@pytest.fixture()
def legend_env(plot_factory, qapp):
    pw = plot_factory()
    assert pw.plot_variable("a")
    msgs = []
    pw.window()._show_status_message = lambda msg: msgs.append(msg)
    return pw, msgs


def _style_panel_of(menus):
    """从替身菜单列表中取出「样式」子菜单内嵌的面板 widget。"""
    sub = menus[1]
    style_act = next(
        a.text for a in sub.actions() if isinstance(a.text, QWidgetAction)
    )
    panel = style_act.defaultWidget()
    assert panel is not None, "样式子菜单应以 QWidgetAction 内嵌面板"
    return panel


def _seg_buttons(panel, index):
    """取第 index 组分段控件的按钮 dict（label → QPushButton）。"""
    segs = panel.findChildren(_SegmentedControl)
    assert len(segs) == 2
    return {b.text(): b for b in segs[index].findChildren(QPushButton)}


def _exec_menu_with_click(
    monkeypatch, qapp, pw, seg_index, label, before_click=None
):
    """弹右键菜单并在 exec 期间（菜单打开时）点击指定分段按钮。

    替身 exec 同步返回，点击必须经 before_return 在 exec 内完成，
    才能命中「picked 记录 → exec 返回 → 关闭后统一执行」链路。
    """
    state = {"menus": None}

    def _click():
        if before_click is not None:
            before_click()
        panel = _style_panel_of(state["menus"])
        QTest.mouseClick(
            _seg_buttons(panel, seg_index)[label], Qt.MouseButton.LeftButton
        )
        qapp.processEvents()

    state["menus"] = _fake_menu_env(monkeypatch, before_return=_click)
    pw.legend_label._exec_var_menu("a", QPoint(0, 0))
    return state["menus"]


def test_menu_has_single_style_entry_with_panel(legend_env, monkeypatch):
    """结构：单一「样式」入口（无绘制方式/数据点子菜单），面板两组分段
    各 3 段且当前值（auto）选中。"""
    pw, _msgs = legend_env
    menus = _fake_menu_env(monkeypatch)
    pw.legend_label._exec_var_menu("a", QPoint(0, 0))

    main = menus[0]
    texts = [a.text for a in main.actions()]
    assert "样式" in texts
    assert "绘制方式" not in texts and "数据点" not in texts

    panel = _style_panel_of(menus)
    draw_btns = _seg_buttons(panel, 0)
    assert set(draw_btns) == {"直线", "阶梯后", "阶梯前"}
    assert [t for t, b in draw_btns.items() if b.isChecked()] == ["直线"]
    symbol_btns = _seg_buttons(panel, 1)
    assert set(symbol_btns) == {"自动", "显示", "隐藏"}
    assert [t for t, b in symbol_btns.items() if b.isChecked()] == ["自动"]


def test_click_segment_switches_draw_mode(legend_env, monkeypatch, qapp):
    """点击「阶梯后」分段 → picked 记录 → 关闭后统一应用 + 状态栏回执。"""
    pw, msgs = legend_env
    _exec_menu_with_click(monkeypatch, qapp, pw, 0, "阶梯后")

    ci = pw.curves["a"]
    assert ci.draw_mode == DRAW_MODE_STEP_POST
    assert ci.curve.opts["stepMode"] == "right"
    assert msgs and "绘制方式" in msgs[-1]


def test_click_segment_pins_symbol_policy(legend_env, monkeypatch, qapp):
    """点击「隐藏」分段 → symbol_policy=off + 状态栏回执。"""
    pw, msgs = legend_env
    _exec_menu_with_click(monkeypatch, qapp, pw, 1, "隐藏")

    ci = pw.curves["a"]
    assert ci.symbol_policy == SYMBOL_POLICY_OFF
    assert not getattr(ci.curve, "_has_symbols", False)
    assert msgs and "数据点" in msgs[-1]


def test_checked_state_reflects_current_mode(legend_env, monkeypatch):
    """已切 step-post 后再弹菜单：分段选中跟随当前值。"""
    pw, _msgs = legend_env
    pw._multi_curve_manager.apply_curve_draw_mode("a", DRAW_MODE_STEP_POST)
    menus = _fake_menu_env(monkeypatch)
    pw.legend_label._exec_var_menu("a", QPoint(0, 0))

    panel = _style_panel_of(menus)
    draw_btns = _seg_buttons(panel, 0)
    assert [t for t, b in draw_btns.items() if b.isChecked()] == ["阶梯后"]


def test_enum_auto_step_annotation(legend_env, monkeypatch):
    """枚举通道自动阶梯：面板绘制方式分组标签带「（枚举通道自动）」标注。"""
    pw, _msgs = legend_env
    pw._multi_curve_manager.apply_curve_draw_mode("a", DRAW_MODE_STEP_POST)
    pw.curves["a"].draw_mode_source = "auto"
    menus = _fake_menu_env(monkeypatch)
    pw.legend_label._exec_var_menu("a", QPoint(0, 0))

    panel = _style_panel_of(menus)
    caps = [lb.text() for lb in panel.findChildren(QLabel)]
    assert any("绘制方式" in t and "枚举通道自动" in t for t in caps)


def test_cancel_without_pick_is_noop(legend_env, monkeypatch):
    """取消（exec 返回 None 且未点面板）：状态与曲线集不变。"""
    pw, msgs = legend_env
    menus = _fake_menu_env(monkeypatch)
    pw.legend_label._exec_var_menu("a", QPoint(0, 0))

    assert menus
    assert not msgs
    assert pw.curves["a"].draw_mode == DRAW_MODE_AUTO
    assert pw.curves["a"].symbol_policy == SYMBOL_POLICY_AUTO


def test_stale_curve_guard_blocks_apply(legend_env, monkeypatch, qapp):
    """菜单打开期间曲线被删：分发前双查拦截，状态与曲线集不变。"""

    def _remove_curve():
        pw.curves.pop("a")

    pw, msgs = legend_env
    menus = _exec_menu_with_click(
        monkeypatch, qapp, pw, 0, "阶梯后", before_click=_remove_curve
    )

    assert menus
    assert not msgs  # 未播报 → 动作未执行
    assert "a" not in pw.curves


def test_style_actions_silent_without_status_host(
    plot_factory, qapp, monkeypatch
):
    """宿主无 _show_status_message（旧测试替身/极端场景）：静默成功。"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    # 不挂 _show_status_message，走 getattr None 分支
    _exec_menu_with_click(monkeypatch, qapp, pw, 0, "阶梯后")
    assert pw.curves["a"].draw_mode == DRAW_MODE_STEP_POST
