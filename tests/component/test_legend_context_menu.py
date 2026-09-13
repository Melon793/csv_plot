"""legend 右键菜单 component 测试（offscreen）。

设计 v1.2 方案 B：右键经 mousePress/mouseRelease 驱动 _exec_var_menu。
QGraphicsProxyWidget 的 contextMenuEvent 转发链路已被 Qt 源码 !hasFocus()
早退判死（tmp/proxy_context_menu_probe_v4~v7.py），入口不依赖它。

替身策略：patch plot_ui_manager.QMenu 模块符号为纯 Python FakeMenu
（PySide6 实例调用 menu.exec() 不走被替换的类属性，直接 patch
QMenu.exec 会在 offscreen 下进入原生模态循环导致卡死，见
tmp/probe_qmenu_exec_patch.py）；QMessageBox.warning /
VariableInfoDialog.popup 为"类访问"调用，monkeypatch 直接生效。
"""

from urllib.parse import quote

import pytest

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QMessageBox

from src.ui.drag_drop import ANCHOR_SCHEME


# ---------- helpers ----------

def _fake_menu_env(monkeypatch, choose_text=None, before_return=None):
    """patch plot_ui_manager.QMenu 为纯 Python 替身。

    Args:
        choose_text: exec 时要选中的 action 文案；None 表示菜单直接关闭
        before_return: exec 返回前执行的副作用（如模拟菜单打开期间
            曲线被删除，见 T8）

    Returns:
        menus：每次弹出的菜单实例列表（可查 actions() 文案序列）
    """
    menus = []

    class _FakeAction:
        def __init__(self, text):
            self.text = text
            self.enabled = True

        def setEnabled(self, enabled):
            # 真实 QMenu 上置灰项不可选中，exec 同步尊重该标志
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
            if before_return is not None:
                before_return()
            if choose_text is None:
                return None
            for act in self._acts:
                if act.text == choose_text and act.enabled:
                    return act
            return None

    import src.ui.widgets.plot_ui_manager as pum
    monkeypatch.setattr(pum, "QMenu", _FakeMenu)
    return menus


def _find_anchor_point(browser, var_name):
    """扫描 legend viewport 找到锚点命中坐标（右键 press/release 用）。

    命中逻辑与生产左键点击同源（anchorAt），此处只定位坐标供事件构造。
    """
    href = f"{ANCHOR_SCHEME}:///{quote(var_name, safe='')}"
    for y in range(1, 25, 2):
        for x in range(1, 120, 2):
            if browser.anchorAt(QPoint(x, y)) == href:
                return QPoint(x, y)
    pytest.fail(f"legend 中未找到变量 {var_name} 的锚点坐标")


def _find_blank_point(browser):
    """扫描找一个非锚点坐标（anchorAt 返回空串）"""
    for y in range(1, 40):
        for x in range(1, 120, 4):
            if browser.anchorAt(QPoint(x, y)) == "":
                return QPoint(x, y)
    pytest.fail("legend 中未找到非锚点空白坐标")


def _rmb(event_type, pos):
    """构造右键 QMouseEvent（globalPos 占位：入口用 viewport().mapToGlobal 重算）"""
    f = QPointF(pos)
    return QMouseEvent(
        event_type,
        f,
        f,
        QPointF(0.0, 0.0),
        Qt.MouseButton.RightButton,
        Qt.MouseButton.RightButton,
        Qt.KeyboardModifier.NoModifier,
    )


# ---------- T1 删除变量 ----------

def test_t1_remove_variable(plot_factory, monkeypatch):
    pw = plot_factory()
    assert pw.plot_variable("a")
    assert pw.plot_variable("b")
    browser = pw.legend_label

    emitted = []
    pw.curves_changed.connect(lambda: emitted.append(1))
    refresh = []
    monkeypatch.setattr(
        pw.window().layout_manager,
        "request_mark_stats_refresh",
        lambda *a, **k: refresh.append(1),
    )

    _fake_menu_env(monkeypatch, choose_text="删除变量")
    browser._exec_var_menu("a", QPoint(0, 0))

    assert "a" not in pw.curves
    assert "b" in pw.curves
    assert len(emitted) == 1
    assert refresh, "request_mark_stats_refresh 应被调用"


# ---------- T2 复制变量名 ----------

def test_t2_copy_variable_name(plot_factory, monkeypatch):
    pw = plot_factory()
    assert pw.plot_variable("a")
    assert pw.plot_variable("b")

    emitted = []
    pw.curves_changed.connect(lambda: emitted.append(1))
    QApplication.clipboard().setText("SENTINEL")

    _fake_menu_env(monkeypatch, choose_text="复制变量名")
    pw.legend_label._exec_var_menu("a", QPoint(0, 0))

    assert QApplication.clipboard().text() == "a"
    assert set(pw.curves) == {"a", "b"}
    assert not emitted


# ---------- T3 变量信息（有 loader） ----------

def test_t3_variable_info_popup(plot_factory, monkeypatch):
    pw = plot_factory()
    assert pw.plot_variable("a")
    host = pw.window()
    host.loader = object()  # FakeHost 默认无 loader，注入非 None

    import src.ui.dialogs.variable_info_dialog as vid
    popups = []
    monkeypatch.setattr(
        vid.VariableInfoDialog,
        "popup",
        classmethod(lambda cls, names, parent=None: popups.append((names, parent))),
    )

    _fake_menu_env(monkeypatch, choose_text="变量信息")
    pw.legend_label._exec_var_menu("a", QPoint(0, 0))

    assert popups and popups[0][0] == ["a"] and popups[0][1] is host


# ---------- T4 变量信息（无 loader） ----------

def test_t4_variable_info_no_loader(plot_factory, monkeypatch):
    pw = plot_factory()
    assert pw.plot_variable("a")

    warnings = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        staticmethod(lambda *a, **k: warnings.append(a)),
    )
    import src.ui.dialogs.variable_info_dialog as vid
    popups = []
    monkeypatch.setattr(
        vid.VariableInfoDialog,
        "popup",
        classmethod(lambda cls, names, parent=None: popups.append(names)),
    )

    _fake_menu_env(monkeypatch, choose_text="变量信息")
    pw.legend_label._exec_var_menu("a", QPoint(0, 0))  # FakeHost 无 loader

    assert warnings, "应弹出「尚未加载数据文件」警告"
    assert not popups


# ---------- T5 solo：仅目标可见 + 单次刷新 ----------

def test_t5_solo_single_refresh(plot_factory, monkeypatch):
    pw = plot_factory()
    for n in ("a", "b", "c"):
        assert pw.plot_variable(n)

    legend_calls = []
    orig = pw.update_legend_label

    def spy(html):
        legend_calls.append(html)
        return orig(html)

    monkeypatch.setattr(pw, "update_legend_label", spy)

    _fake_menu_env(monkeypatch, choose_text="仅显示此变量")
    pw.legend_label._exec_var_menu("a", QPoint(0, 0))

    ci = pw.curves
    assert ci["a"].visible is True and ci["a"].curve.isVisible()
    assert ci["b"].visible is False and not ci["b"].curve.isVisible()
    assert ci["c"].visible is False and not ci["c"].curve.isVisible()
    assert len(legend_calls) == 1, "solo 批量切换应只刷新一次 legend"


# ---------- T6 显隐两项并列：solo → 显示全部 ----------

def test_t6_solo_then_show_all(plot_factory, monkeypatch):
    pw = plot_factory()
    for n in ("a", "b", "c"):
        assert pw.plot_variable(n)

    _fake_menu_env(monkeypatch, choose_text="仅显示此变量")
    pw.legend_label._exec_var_menu("a", QPoint(0, 0))
    assert pw.curves["a"].visible is True
    assert pw.curves["b"].visible is False

    # 两项常驻：互斥文案不再存在，靠「显示全部变量」恢复
    _fake_menu_env(monkeypatch, choose_text="显示全部变量")
    pw.legend_label._exec_var_menu("a", QPoint(0, 0))

    assert all(ci.visible for ci in pw.curves.values())
    assert all(ci.curve.isVisible() for ci in pw.curves.values())


# ---------- T7 置灰谓词真值表 ----------

def test_t7_visibility_predicates(plot_factory):
    pw = plot_factory()
    for n in ("a", "b", "c"):
        assert pw.plot_variable(n)

    # 全可见：solo 生效（需隐藏其他），显示全部为空操作
    assert pw.can_solo("a") is True
    assert pw.can_show_all() is False

    pw.solo_curve_visibility("a")
    # solo a 生效中：solo a 置灰，solo b 可切换目标，显示全部可点
    assert pw.can_solo("a") is False
    assert pw.can_solo("b") is True
    assert pw.can_show_all() is True

    # 陈旧变量名：无曲线可 solo → 置灰
    assert pw.can_solo("nope") is False

    # 单曲线且可见：两个动作都是空操作 → 都置灰
    pw2 = plot_factory()
    assert pw2.plot_variable("a")
    assert pw2.can_solo("a") is False
    assert pw2.can_show_all() is False

    # 单曲线被左键隐藏：两项重新可点
    pw2.toggle_curve_visibility_by_name("a")
    assert pw2.can_solo("a") is True
    assert pw2.can_show_all() is True


# ---------- T7b 菜单结构：两项并列 + 分隔线位置 ----------

def test_t7b_menu_layout_and_separators(plot_factory, monkeypatch):
    pw = plot_factory()
    for n in ("a", "b"):
        assert pw.plot_variable(n)

    menus = _fake_menu_env(monkeypatch)
    pw.legend_label._exec_var_menu("a", QPoint(0, 0))

    acts = menus[0].actions()
    # "" 为分隔线占位：仅显隐组与操作组之间一条，复制变量名/变量信息无分隔线
    assert [a.text for a in acts] == [
        "仅显示此变量",
        "显示全部变量",
        "",
        "删除变量",
        "复制变量名",
        "变量信息",
    ]
    # 全可见态：solo 可点、显示全部置灰
    assert acts[0].enabled is True
    assert acts[1].enabled is False


# ---------- T7c 置灰项不可触发 ----------

def test_t7c_disabled_item_not_triggerable(plot_factory, monkeypatch):
    pw = plot_factory()
    for n in ("a", "b"):
        assert pw.plot_variable(n)
    assert pw.solo_curve_visibility("a") is True

    _fake_menu_env(monkeypatch, choose_text="仅显示此变量")
    pw.legend_label._exec_var_menu("a", QPoint(0, 0))

    # 已处于 solo a 态 → 该项置灰，选中后可见性无任何变化
    assert pw.curves["a"].visible is True
    assert pw.curves["b"].visible is False
    assert pw.curves["b"].curve.isVisible() is False


# ---------- T8 陈旧防御：菜单打开期间曲线被删 ----------

def test_t8_stale_curve_guard(plot_factory, monkeypatch):
    pw = plot_factory()
    assert pw.plot_variable("a")
    assert pw.plot_variable("b")

    QApplication.clipboard().setText("SENTINEL")

    def _remove_curve():
        pw.curves.pop("a")  # 模拟菜单打开期间被 reload/编辑器删除

    menus = _fake_menu_env(
        monkeypatch, choose_text="复制变量名", before_return=_remove_curve
    )
    pw.legend_label._exec_var_menu("a", QPoint(0, 0))

    # 菜单弹过，但动作被分发前的二次校验拦截
    assert menus
    assert QApplication.clipboard().text() == "SENTINEL"


# ---------- T9 入口贯通：右键 press/release 驱动 ----------

def test_t9_entry_through_press_release(plot_factory, monkeypatch, qapp):
    pw = plot_factory()
    assert pw.plot_variable("a")
    browser = pw.legend_label

    pos = _find_anchor_point(browser, "a")
    _fake_menu_env(monkeypatch, choose_text="复制变量名")

    QApplication.clipboard().setText("SENTINEL")
    browser.mousePressEvent(_rmb(QEvent.Type.MouseButtonPress, pos))
    browser.mouseReleaseEvent(_rmb(QEvent.Type.MouseButtonRelease, pos))
    qapp.processEvents()  # 让 singleShot(0) 落地

    assert QApplication.clipboard().text() == "a"


# ---------- T10 手势取消：按住右键拖出阈值后松开 ----------

def test_t10_right_drag_cancels_menu(plot_factory, monkeypatch, qapp):
    pw = plot_factory()
    assert pw.plot_variable("a")
    browser = pw.legend_label
    pos = _find_anchor_point(browser, "a")

    called = []
    monkeypatch.setattr(browser, "_exec_var_menu", lambda *a, **k: called.append(a))
    QApplication.clipboard().setText("SENTINEL")

    far = pos + QPoint(QApplication.startDragDistance() + 10, 0)
    browser.mousePressEvent(_rmb(QEvent.Type.MouseButtonPress, pos))
    browser.mouseReleaseEvent(_rmb(QEvent.Type.MouseButtonRelease, far))
    qapp.processEvents()

    assert not called
    assert QApplication.clipboard().text() == "SENTINEL"


# ---------- T11 非锚点右键：不弹菜单 ----------

def test_t11_right_click_blank_no_menu(plot_factory, monkeypatch, qapp):
    pw = plot_factory()
    assert pw.plot_variable("a")
    browser = pw.legend_label

    called = []
    monkeypatch.setattr(browser, "_exec_var_menu", lambda *a, **k: called.append(a))

    blank = _find_blank_point(browser)
    browser.mousePressEvent(_rmb(QEvent.Type.MouseButtonPress, blank))
    browser.mouseReleaseEvent(_rmb(QEvent.Type.MouseButtonRelease, blank))
    qapp.processEvents()

    assert not called
