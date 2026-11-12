"""legend 右键菜单「绘制方式 / 数据点」子菜单组件测试（offscreen）。

方案 A：图例右键菜单新增单曲线粒度样式入口，与变量编辑器同源
（MultiCurveManager.apply_curve_draw_mode / apply_curve_symbol_policy）。

替身策略与 test_legend_context_menu 相同：patch plot_ui_manager.QMenu
为纯 Python FakeMenu（支持 addMenu 子菜单与 deleteLater），offscreen 下
不得进入原生模态循环。
"""

from PySide6.QtCore import QPoint

import pytest

from src.core.data_types import (
    DRAW_MODE_STEP_POST,
    SYMBOL_POLICY_OFF,
)


def _fake_menu_env(monkeypatch, choose_text=None, before_return=None):
    """patch plot_ui_manager.QMenu 为支持子菜单的纯 Python 替身。"""
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

    class _FakeMenu:
        def __init__(self, parent=None):
            self._acts = []
            self._subs = []
            menus.append(self)

        def addAction(self, text):
            act = _FakeAction(text)
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


def test_menu_contains_style_submenus_with_checked_current(legend_env, monkeypatch):
    """结构：绘制方式/数据点子菜单各 3 项，当前值（auto）checked。"""
    pw, _msgs = legend_env
    menus = _fake_menu_env(monkeypatch)
    pw.legend_label._exec_var_menu("a", QPoint(0, 0))

    main = menus[0]
    texts = [a.text for a in main.actions()]
    assert "绘制方式" in texts and "数据点" in texts
    # 两个子菜单（menus[1]=绘制方式, menus[2]=数据点）
    draw_menu = menus[1]
    assert [a.text for a in draw_menu.actions()] == ["直线", "阶梯(后)", "阶梯(前)"]
    checked = [a.text for a in draw_menu.actions() if a.checked]
    assert checked == ["直线"]  # 初值 auto → 直线 checked
    symbol_menu = menus[2]
    assert [a.text for a in symbol_menu.actions()] == ["自动", "始终显示", "始终隐藏"]
    checked = [a.text for a in symbol_menu.actions() if a.checked]
    assert checked == ["自动"]


def test_choose_step_post_switches_draw_mode(legend_env, monkeypatch):
    """选中「阶梯(后)」→ ci.draw_mode / stepMode / 状态栏回执。"""
    pw, msgs = legend_env
    _fake_menu_env(monkeypatch, choose_text="阶梯(后)")
    pw.legend_label._exec_var_menu("a", QPoint(0, 0))

    ci = pw.curves["a"]
    assert ci.draw_mode == DRAW_MODE_STEP_POST
    assert ci.curve.opts["stepMode"] == "left"
    assert msgs and "绘制方式" in msgs[-1]


def test_choose_symbol_off_pins_policy(legend_env, monkeypatch):
    """选中「始终隐藏」→ symbol_policy=off + 状态栏回执。"""
    pw, msgs = legend_env
    _fake_menu_env(monkeypatch, choose_text="始终隐藏")
    pw.legend_label._exec_var_menu("a", QPoint(0, 0))

    ci = pw.curves["a"]
    assert ci.symbol_policy == SYMBOL_POLICY_OFF
    assert not getattr(ci.curve, "_has_symbols", False)
    assert msgs and "数据点" in msgs[-1]


def test_checked_state_reflects_current_mode(legend_env, monkeypatch):
    """已切 step-post 后再弹菜单：checked 跟随当前值。"""
    pw, _msgs = legend_env
    pw._multi_curve_manager.apply_curve_draw_mode("a", DRAW_MODE_STEP_POST)
    menus = _fake_menu_env(monkeypatch)
    pw.legend_label._exec_var_menu("a", QPoint(0, 0))

    draw_menu = menus[1]
    checked = [a.text for a in draw_menu.actions() if a.checked]
    assert checked == ["阶梯(后)"]


def test_stale_curve_guard_blocks_apply(legend_env, monkeypatch):
    """菜单打开期间曲线被删：分发前双查拦截，状态与曲线集不变。"""
    pw, msgs = legend_env

    def _remove_curve():
        pw.curves.pop("a")

    menus = _fake_menu_env(
        monkeypatch, choose_text="阶梯(后)", before_return=_remove_curve
    )
    pw.legend_label._exec_var_menu("a", QPoint(0, 0))

    assert menus
    assert not msgs  # 未播报 → 动作未执行
    assert "a" not in pw.curves


def test_style_actions_silent_without_status_host(plot_factory, qapp, monkeypatch):
    """宿主无 _show_status_message（旧测试替身/极端场景）：静默成功。"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    _fake_menu_env(monkeypatch, choose_text="阶梯(后)")
    # 不挂 _show_status_message，走 getattr None 分支
    pw.legend_label._exec_var_menu("a", QPoint(0, 0))
    assert pw.curves["a"].draw_mode == DRAW_MODE_STEP_POST
