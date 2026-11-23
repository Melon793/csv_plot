"""占位「替换变量…」component 测试（身份匹配 C4 / D7/D13 / MJ-8 四边界）。

菜单替身照 test_invalid_curve_ui._fake_menu_env 口径；选择对话框用
monkeypatch pick 替身（offscreen 模态框会挂起，陷阱 #4），对话框本体
的过滤/选中逻辑单独直测（不 exec）。
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
from PySide6.QtCore import QPoint

from src.core.data_types import DRAW_MODE_AUTO, InvalidCurveRef
from src.ui.dialogs.variable_picker import VariablePickerDialog


def _fake_menu_env(monkeypatch, choose_text=None):
    menus = []

    class _FakeAction:
        def __init__(self, text):
            self.text = text

    class _FakeMenu:
        def __init__(self, parent=None):
            self._acts = []
            menus.append(self)

        def addAction(self, text):
            act = _FakeAction(text)
            self._acts.append(act)
            return act

        def exec(self, *a):
            if choose_text is None:
                return None
            return next(x for x in self._acts if x.text == choose_text)

        def deleteLater(self):
            pass

    monkeypatch.setattr("src.ui.widgets.plot_ui_manager.QMenu", _FakeMenu)
    return menus


def _setup(plot_factory, monkeypatch, chosen, df=None, var_names=None):
    """pw + 宿主 loader/播报替身 + pick 替身。返回 (pw, broadcasts, emitted)。"""
    pw = plot_factory(df)
    host = pw.window()
    names = var_names if var_names is not None else list(pw.data.columns)
    host.loader = SimpleNamespace(var_names=names)
    broadcasts: list = []
    host._broadcast = lambda text, level="info": broadcasts.append((text, level))
    emitted: list = []
    pw.curves_changed.connect(lambda: emitted.append(1))
    monkeypatch.setattr(
        VariablePickerDialog, "pick",
        classmethod(lambda cls, names, parent=None, **kw: chosen),
    )
    return pw, broadcasts, emitted


def _ref(**kw):
    base = dict(var_name="ghost", color="red", visible_preference=True)
    base.update(kw)
    return InvalidCurveRef(**base)


class TestReplaceFlow:
    def test_happy_path_carries_color_visibility_settings(
        self, plot_factory, monkeypatch
    ):
        """D13②③④⑤：add 成功后删占位，颜色/可见性/显示名/换算携带，回执一条。"""
        pw, broadcasts, emitted = _setup(plot_factory, monkeypatch, "b")
        pw.invalid_curves["ghost"] = _ref(
            visible_preference=False, display_name="自定义", factor=2.0, offset=-1.0
        )
        _fake_menu_env(monkeypatch, choose_text="替换变量…")
        pw.legend_label._exec_invalid_menu("ghost", QPoint(0, 0))

        assert "ghost" not in pw.invalid_curves
        ci = pw.curves["b"]
        assert ci.color == "red"
        assert ci.visible is False
        assert ci.display_name == "自定义"
        assert ci.factor == 2.0
        assert ci.offset == -1.0
        # add/可见性/收尾各驱动一次增量刷新，至少一次（§4.5 发射口径）
        assert emitted
        assert broadcasts[-1][0] == "已替换占位 ghost → b"

    def test_reject_duplicate_in_curves(self, plot_factory, monkeypatch):
        """D13①：新名已在本格 curves → 拒绝，不删任何资产。"""
        pw, broadcasts, emitted = _setup(plot_factory, monkeypatch, "a")
        assert pw.plot_variable("a")
        emitted.clear()  # 首绘自身的 emit 不算替换路径的
        pw.invalid_curves["ghost"] = _ref()
        _fake_menu_env(monkeypatch, choose_text="替换变量…")
        pw.legend_label._exec_invalid_menu("ghost", QPoint(0, 0))

        assert "ghost" in pw.invalid_curves
        assert set(pw.curves) == {"a"}
        assert not emitted
        assert "已在本格绘图中" in broadcasts[-1][0]
        assert broadcasts[-1][1] == "warn"

    def test_reject_hits_another_placeholder(self, plot_factory, monkeypatch):
        """D13①：新名恰为另一占位 → 拒绝，两个占位都保留。"""
        pw, broadcasts, emitted = _setup(plot_factory, monkeypatch, "ghost2")
        pw.invalid_curves["ghost"] = _ref()
        pw.invalid_curves["ghost2"] = InvalidCurveRef(var_name="ghost2", color="blue")
        _fake_menu_env(monkeypatch, choose_text="替换变量…")
        pw.legend_label._exec_invalid_menu("ghost", QPoint(0, 0))

        assert set(pw.invalid_curves) == {"ghost", "ghost2"}
        assert not pw.curves
        assert not emitted
        assert "已是本格占位" in broadcasts[-1][0]

    def test_same_name_revives_with_preferences(self, plot_factory, monkeypatch):
        """新名恰为占位自身：同名复活，add 内部原子消费占位并带回偏好。"""
        pw, broadcasts, emitted = _setup(plot_factory, monkeypatch, "a")
        pw.invalid_curves["a"] = InvalidCurveRef(
            var_name="a", color="green", visible_preference=False,
            display_name="复活名", factor=3.0,
        )
        _fake_menu_env(monkeypatch, choose_text="替换变量…")
        pw.legend_label._exec_invalid_menu("a", QPoint(0, 0))

        assert "a" not in pw.invalid_curves
        ci = pw.curves["a"]
        assert ci.color == "green"
        assert ci.visible is False
        assert ci.display_name == "复活名"
        assert ci.factor == 3.0
        assert broadcasts[-1][0] == "已替换占位 a → a"

    def test_failure_keeps_placeholder(self, plot_factory, monkeypatch):
        """D13②：add 失败（文本列不可绘）→ 回滚不删占位。"""
        df = pd.DataFrame({"a": [1.0, 2.0], "txt": ["x", "y"]})
        pw, broadcasts, emitted = _setup(
            plot_factory, monkeypatch, "txt", df=df, var_names=["a", "txt"]
        )
        pw.invalid_curves["ghost"] = _ref()
        _fake_menu_env(monkeypatch, choose_text="替换变量…")
        pw.legend_label._exec_invalid_menu("ghost", QPoint(0, 0))

        assert "ghost" in pw.invalid_curves
        assert "txt" not in pw.curves
        assert not emitted
        assert "替换失败" in broadcasts[-1][0]

    def test_enum_channel_replacement(self, plot_factory, monkeypatch, mdf_loader):
        """MJ-8 边界：枚举通道可替换，自动阶梯绘制方式生效。"""
        try:
            pw = plot_factory()
            pw.plot_context.loader = mdf_loader
            host = pw.window()
            host.loader = mdf_loader
            broadcasts: list = []
            host._broadcast = lambda text, level="info": broadcasts.append((text, level))
            monkeypatch.setattr(
                VariablePickerDialog, "pick",
                classmethod(lambda cls, names, parent=None, **kw: "State"),
            )
            pw.invalid_curves["ghost"] = _ref()
            _fake_menu_env(monkeypatch, choose_text="替换变量…")
            pw.legend_label._exec_invalid_menu("ghost", QPoint(0, 0))

            assert "ghost" not in pw.invalid_curves
            ci = pw.curves["State"]
            assert ci.draw_mode != DRAW_MODE_AUTO
            assert broadcasts[-1][0] == "已替换占位 ghost → State"
        finally:
            mdf_loader.close()

    def test_cancel_keeps_placeholder(self, plot_factory, monkeypatch):
        pw, broadcasts, emitted = _setup(plot_factory, monkeypatch, None)
        pw.invalid_curves["ghost"] = _ref()
        _fake_menu_env(monkeypatch, choose_text="替换变量…")
        pw.legend_label._exec_invalid_menu("ghost", QPoint(0, 0))

        assert "ghost" in pw.invalid_curves
        assert not pw.curves
        assert not broadcasts

    def test_no_loader_broadcasts_hint(self, plot_factory, monkeypatch):
        pw = plot_factory()
        host = pw.window()
        broadcasts: list = []
        host._broadcast = lambda text, level="info": broadcasts.append((text, level))
        pw.invalid_curves["ghost"] = _ref()
        _fake_menu_env(monkeypatch, choose_text="替换变量…")
        pw.legend_label._exec_invalid_menu("ghost", QPoint(0, 0))

        assert "ghost" in pw.invalid_curves
        assert "尚未加载数据文件" in broadcasts[-1][0]

    def test_menu_has_three_actions(self, plot_factory, monkeypatch):
        pw = plot_factory()
        pw.invalid_curves["ghost"] = _ref()
        menus = _fake_menu_env(monkeypatch, choose_text=None)
        pw.legend_label._exec_invalid_menu("ghost", QPoint(0, 0))
        texts = [a.text for a in menus[0]._acts]
        assert texts == ["删除占位", "复制变量名", "替换变量…"]


class TestPickerDialog:
    def test_filter_substring_and_wildcard(self, qapp):
        dlg = VariablePickerDialog(["EngineSpeed", "BrakePressure", "WheelSpeedFL"])
        try:
            dlg._search.setText("speed")
            visible = [
                dlg._list.item(i).text()
                for i in range(dlg._list.count())
                if not dlg._list.item(i).isHidden()
            ]
            assert visible == ["EngineSpeed", "WheelSpeedFL"]
            dlg._search.setText("wheel*fl")
            assert dlg.selected_name() == "WheelSpeedFL"
            dlg._search.setText("不存在")
            assert dlg.selected_name() is None
        finally:
            dlg.deleteLater()

    def test_empty_keyword_shows_all(self, qapp):
        dlg = VariablePickerDialog(["a", "b"])
        try:
            assert dlg.selected_name() == "a"  # 默认选中首行
            dlg._search.setText("")
            visible = sum(
                not dlg._list.item(i).isHidden() for i in range(dlg._list.count())
            )
            assert visible == 2
        finally:
            dlg.deleteLater()

    def test_accept_without_selection_stays_open(self, qapp):
        dlg = VariablePickerDialog(["a"])
        try:
            dlg._list.setCurrentItem(None)
            dlg._on_accept()
            assert dlg.result() != VariablePickerDialog.DialogCode.Accepted
            dlg._list.setCurrentRow(0)
            dlg._on_accept()
            assert dlg.result() == VariablePickerDialog.DialogCode.Accepted
            assert dlg.selected_name() == "a"
        finally:
            dlg.deleteLater()