"""失效曲线占位（方案 B2）数据层 component 测试（offscreen）。

钉住 Commit 1 的数据层契约：
- demote_to_invalid：从 curves 迁移到 invalid_curves，同名不重复；
- 三个复活入口（add_variable_to_plot / plot_variable 首绘 / add_variables_to_plot
  批量）命中占位时以占位色复活、恢复可见性偏好、从占位侧移除；
- clear_plot_item 的 clear_invalid 两种取值（False 保留、True 连带清）；
- 不变量：同一变量绝不同时在 curves 与 invalid_curves。
"""

import pytest

from PySide6.QtWidgets import QMessageBox

from src.core.data_types import InvalidCurveRef


@pytest.fixture()
def silent_dialogs(monkeypatch):
    """替换 QMessageBox 静态方法为记录器，避免 offscreen 模态阻塞（陷阱 #4）"""
    calls = {"warning": [], "information": [], "question": []}
    monkeypatch.setattr(
        QMessageBox, "warning",
        staticmethod(lambda parent, title, text, *a, **k: calls["warning"].append((title, text)) or QMessageBox.StandardButton.Ok),
    )
    monkeypatch.setattr(
        QMessageBox, "information",
        staticmethod(lambda parent, title, text, *a, **k: calls["information"].append((title, text)) or QMessageBox.StandardButton.Ok),
    )
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda parent, title, text, *a, **k: calls["question"].append((title, text)) or QMessageBox.StandardButton.No),
    )
    return calls


def _assert_disjoint(pw):
    """不变量：同一变量名绝不同时出现在两侧"""
    overlap = set(pw.curves) & set(pw.invalid_curves)
    assert not overlap, f"curves 与 invalid_curves 重叠: {overlap}"


# ---------- demote 迁移 ----------

def test_demote_moves_curve_to_invalid(plot_factory, silent_dialogs):
    """demote：变量从 curves 移除、进 invalid_curves，保留颜色/可见性"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    pw.curves["a"].color = "red"
    pw.curves["a"].visible = False

    pw.demote_to_invalid("a", "red", False)

    assert "a" not in pw.curves
    assert "a" in pw.invalid_curves
    ref = pw.invalid_curves["a"]
    assert isinstance(ref, InvalidCurveRef)
    assert ref.color == "red"
    assert ref.visible_preference is False
    _assert_disjoint(pw)


def test_demote_absent_variable_creates_placeholder(plot_factory, silent_dialogs):
    """demote 一个不在 curves 的变量（reload 已清空场景）：仅落占位"""
    pw = plot_factory()
    pw.demote_to_invalid("ghost", "green", True)

    assert "ghost" not in pw.curves
    assert pw.invalid_curves["ghost"].color == "green"
    _assert_disjoint(pw)


def test_demote_default_color_gray_when_empty(plot_factory, silent_dialogs):
    """demote 传空色：落固定灰色兜底"""
    pw = plot_factory()
    pw.demote_to_invalid("x", "", True)
    assert pw.invalid_curves["x"].color == "gray"


# ---------- 复活入口 1：add_variable_to_plot ----------

def test_revive_via_add_variable_to_plot(plot_factory, silent_dialogs):
    """add_variable_to_plot 命中占位：同色复活、恢复可见性、占位移除"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    pw.demote_to_invalid("a", "purple", True)
    assert "a" in pw.invalid_curves

    assert pw.add_variable_to_plot("a")

    assert "a" in pw.curves
    assert "a" not in pw.invalid_curves
    assert pw.curves["a"].color == "purple"
    assert pw.curves["a"].visible is True
    _assert_disjoint(pw)


def test_revive_via_add_variable_restores_hidden_preference(plot_factory, silent_dialogs):
    """复活时可见性偏好 False → 曲线以隐藏态复活"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    pw.demote_to_invalid("a", "orange", False)

    assert pw.add_variable_to_plot("a")

    assert pw.curves["a"].visible is False
    _assert_disjoint(pw)


# ---------- 复活入口 2：plot_variable 首绘 ----------

def test_revive_via_plot_variable_first_render(plot_factory, silent_dialogs):
    """curves 为空、仅剩占位时手动添加 = 首绘分支复活（同色）"""
    pw = plot_factory()
    # 直接构造占位（不经过 curves），模拟"上一轮 reload 全部降级"
    pw.invalid_curves["a"] = InvalidCurveRef(var_name="a", color="brown", visible_preference=True)

    assert pw.plot_variable("a")

    assert "a" in pw.curves
    assert "a" not in pw.invalid_curves
    assert pw.curves["a"].color == "brown"
    _assert_disjoint(pw)


def test_plot_variable_first_render_default_blue_without_placeholder(plot_factory, silent_dialogs):
    """无占位的首绘仍是蓝色（向后兼容）"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    assert pw.curves["a"].color == "blue"


def test_plot_variable_preferred_color_param(plot_factory, silent_dialogs):
    """plot_variable 新增 preferred_color 形参生效（模板复活着色依赖）"""
    pw = plot_factory()
    assert pw.plot_variable("a", preferred_color="cyan")
    assert pw.curves["a"].color == "cyan"


# ---------- 复活入口 3：add_variables_to_plot 批量 ----------

def test_revive_via_batch_add(plot_factory, silent_dialogs):
    """批量拖入命中占位：逐条查占位取色复活"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    pw.demote_to_invalid("a", "olive", True)
    # b 是新变量，a 是复活
    pw.add_variables_to_plot(["a", "b"])

    assert "a" in pw.curves and "b" in pw.curves
    assert "a" not in pw.invalid_curves
    assert pw.curves["a"].color == "olive"
    _assert_disjoint(pw)


# ---------- clear_plot_item 的 clear_invalid ----------

def test_clear_plot_item_keeps_invalid_by_default(plot_factory, silent_dialogs):
    """clear_plot_item() 缺省不动占位（reload 跨清空存活）"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    pw.demote_to_invalid("a", "red", True)

    pw.clear_plot_item()

    assert list(pw.curves) == []
    assert "a" in pw.invalid_curves


def test_clear_plot_item_clears_invalid_when_requested(plot_factory, silent_dialogs):
    """clear_plot_item(clear_invalid=True) 连带清占位（用户显式清空/模板还原）"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    pw.demote_to_invalid("a", "red", True)

    pw.clear_plot_item(clear_invalid=True)

    assert list(pw.curves) == []
    assert list(pw.invalid_curves) == []


def test_clear_current_plot_clears_invalid(plot_factory, silent_dialogs):
    """用户入口 clear_current_plot 连带清占位，无孤儿资产"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    pw.demote_to_invalid("a", "red", True)

    pw.clear_current_plot()

    assert list(pw.invalid_curves) == []


# ---------- 不变量：连续 demote/复活 ----------

def test_demote_revive_cycle_stays_disjoint(plot_factory, silent_dialogs):
    """多轮 demote→复活循环后两侧始终不重叠"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    for color in ("red", "green", "blue"):
        pw.demote_to_invalid("a", color, True)
        _assert_disjoint(pw)
        assert pw.add_variable_to_plot("a")
        _assert_disjoint(pw)
        assert pw.curves["a"].color == color
