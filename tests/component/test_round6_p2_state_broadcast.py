"""第 6 轮批次 C（P2 精选）：单变量 legend 播报 parity / 菜单入区 updated
计数 / preset 应用失败字段回滚。

P2-33：单变量 legend 移动/复制成功原静默（仅 degraded 出声），与批量路径
同口径收口到 broadcast_transfer。
P2-34：菜单批量入区同名覆盖更新原按「新增」播报，漏 updated 计数。
C-3：apply_curve_line_preset 先写字段后试应用，异常时残留钉住值使调用方
同值 no-op 守卫吞掉重试。
"""
from __future__ import annotations

import types

import pandas as pd
import pytest

from PySide6.QtCore import Qt

from src.ui import drag_drop
from src.ui.drag_drop import build_legend_var_mimedata

_KEEP_ALIVE = []  # QMimeData 模块级保活（历史 dropEvent 段错误坑）


@pytest.fixture()
def three_curves(plot_factory, qapp):
    df = pd.DataFrame({n: [float(i), float(i + 1)] for i, n in enumerate("abc")})
    pw = plot_factory(df)
    pw.resize(500, 300)
    pw.window().show()
    pw.show()
    for n in "abc":
        assert pw.plot_variable(n)
    return pw


def _fake_drop_event(mime):
    return types.SimpleNamespace(
        mimeData=lambda: mime,
        keyboardModifiers=lambda: Qt.KeyboardModifier.NoModifier,
        ignore=lambda: None,
        accept=lambda: None,
        acceptProposedAction=lambda: None,
    )


def test_single_legend_drop_broadcasts_move(three_curves, plot_factory, qapp, monkeypatch):
    source = three_curves
    target = plot_factory(pd.DataFrame({"a": [1.0, 2.0], "z": [3.0, 4.0]}))
    target.resize(500, 300)
    target.window().show()
    target.show()
    messages = []
    monkeypatch.setattr(
        "src.ui.widgets.plot_widget.broadcast_transfer",
        lambda host, action, names=(), **kw: messages.append(
            (action, list(names), kw)
        ),
    )
    mime = build_legend_var_mimedata(["a"], source)
    _KEEP_ALIVE.append(mime)
    drag_drop.set_active_legend_drag(source, ["a"], batch=False)
    try:
        target._handle_legend_drop(_fake_drop_event(mime), ["a"])
    finally:
        drag_drop.clear_active_legend_drag()
    qapp.processEvents()  # 源端延迟删除
    assert "a" in target.curves
    assert messages, "单变量移动成功必须播报（修复前静默）"
    action, names, kw = messages[-1]
    assert action == "move" and names == ["a"]
    assert kw.get("degraded") is False


def test_single_legend_drop_degraded_note(plot_factory, qapp, monkeypatch):
    """源已销毁 + 移动语义：降级注记并入统一播报（不再单独 broadcast_text）。"""
    df = pd.DataFrame({"a": [1.0, 2.0], "z": [3.0, 4.0]})
    target = plot_factory(df)
    target.resize(500, 300)
    target.window().show()
    target.show()
    assert target.plot_variable("a")  # z 在数据域内但未绘制，drop 可落地
    messages = []
    monkeypatch.setattr(
        "src.ui.widgets.plot_widget.broadcast_transfer",
        lambda host, action, names=(), **kw: messages.append(
            (action, list(names), kw)
        ),
    )
    mime = build_legend_var_mimedata(["z"], None)
    _KEEP_ALIVE.append(mime)
    # 不注册拖拽上下文 → source=None → degraded
    target._handle_legend_drop(_fake_drop_event(mime), ["z"])
    assert messages, "降级路径必须播报"
    action, names, kw = messages[-1]
    assert kw.get("degraded") is True


def test_menu_shelf_intake_reports_updated(three_curves, qapp):
    from tests.component.test_legend_selection import _attach_shelf_host

    pw = three_curves
    browser = pw.legend_label
    _host, dialog, messages = _attach_shelf_host(pw)
    browser._send_selection_to_shelf(["a"], cut=False)
    assert any("已复制 a" in m for m in messages)
    messages.clear()
    browser._send_selection_to_shelf(["a"], cut=False)
    assert any("已更新 1 条同名暂存" in m for m in messages), (
        f"同名覆盖更新须与拖入区同规格播报：{messages}"
    )


def test_single_legend_move_ask_merge_stays_silent(three_curves, plot_factory, qapp, monkeypatch):
    """评审 B1：目标已有同名 → ask 确认合并（merged-only，源被删）不得
    错报「未搬运」，恢复旧静默（结果守卫与出区播报同口径）。"""
    source = three_curves
    target = plot_factory(pd.DataFrame({"a": [9.0, 8.0]}))
    target.resize(500, 300)
    target.window().show()
    target.show()
    assert target.plot_variable("a")  # 目标已有同名
    messages = []
    monkeypatch.setattr(
        "src.ui.widgets.plot_widget.broadcast_transfer",
        lambda host, action, names=(), **kw: messages.append(
            (action, list(names), kw)
        ),
    )
    from PySide6.QtWidgets import QMessageBox
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes),
    )
    mime = build_legend_var_mimedata(["a"], source)
    _KEEP_ALIVE.append(mime)
    drag_drop.set_active_legend_drag(source, ["a"], batch=False)
    try:
        target._handle_legend_drop(_fake_drop_event(mime), ["a"])
    finally:
        drag_drop.clear_active_legend_drag()
    qapp.processEvents()  # 源端延迟删除
    assert "a" not in source.curves, "合并后源曲线应被删除"
    assert not messages, f"merged-only 不得播报（尤其不得错报未搬运）：{messages}"


def test_preset_apply_failure_rolls_back_field(three_curves, monkeypatch):
    pw = three_curves
    ci = pw.curves["a"]
    assert ci.line_preset == "auto"

    def boom(*a, **k):
        raise RuntimeError("view destroyed")

    monkeypatch.setattr(pw, "_apply_curve_style", boom)
    assert pw._multi_curve_manager.apply_curve_line_preset("a", "bold") is False
    assert ci.line_preset == "auto", "应用失败必须回滚字段（防 no-op 守卫吞重试）"
    # 回滚后重试可正常应用
    monkeypatch.undo()
    assert pw._multi_curve_manager.apply_curve_line_preset("a", "bold") is True
    assert ci.line_preset == "bold"