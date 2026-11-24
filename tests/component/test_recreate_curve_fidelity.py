"""第 6 轮 P1-C：_recreate_curve 重建链保真（symbol_policy / visible / preset）。

reload 安全网（_safe_check_and_rebuild_curve）与显隐 toggle 的曲线离场景分支
都走 _recreate_curve；add_variable_to_plot 新建 CurveInfo 回落 auto/可见，
用户钉住的数据点策略与隐藏态不得静默重置（probe E4 实测 policy→auto、
visible→True）。与 reload 接力（saved_state）同族保真契约。
"""
from __future__ import annotations

import pandas as pd

from src.core.data_types import (
    LINE_PRESET_AUTO,
    LINE_PRESET_BOLD,
    SYMBOL_POLICY_AUTO,
    SYMBOL_POLICY_OFF,
    SYMBOL_POLICY_ON,
)


def _make_pw(plot_factory):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    pw = plot_factory(df)
    pw.resize(500, 300)
    pw.window().show()
    pw.show()
    assert pw.plot_variable("a")
    return pw


def _invalidate(pw, name="a"):
    """模拟失效：曲线离场景（reload 安全网判据 scene() is None）。"""
    ci = pw.curves[name]
    if ci.curve is not None:
        pw.plot_item.removeItem(ci.curve)


def test_recreate_removes_old_scene_item(plot_factory, qapp):
    """旧项仍在场景时也必须重建，且删除新项后不得留下旧曲线。"""
    pw = _make_pw(plot_factory)
    old_ci = pw.curves["a"]
    old_curve = old_ci.curve
    pw._multi_curve_manager._recreate_curve("a")
    new_ci = pw.curves["a"]
    assert new_ci is not old_ci
    assert new_ci.curve is not old_curve
    assert old_curve not in pw.plot_item.scene().items()
    assert sum(
        callable(getattr(item, "name", None)) and item.name() == "a"
        for item in pw.plot_item.scene().items()
    ) == 1
    assert pw.remove_variable_from_plot("a") is True
    assert "a" not in pw.curves
    assert not any(
        callable(getattr(item, "name", None)) and item.name() == "a"
        for item in pw.plot_item.scene().items()
    )


def test_recreate_failure_restores_old_curve(plot_factory, qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    pw = _make_pw(plot_factory)
    old_ci = pw.curves["a"]
    old_curve = old_ci.curve
    new_curve = None

    def fail_after_replace():
        nonlocal new_curve
        new_curve = pw.curves["a"].curve
        assert new_curve is not old_curve
        raise RuntimeError("rebuild failed")

    monkeypatch.setattr(
        pw._multi_curve_manager, "_update_header_for_curves", fail_after_replace
    )
    monkeypatch.setattr(
        QMessageBox, "critical", staticmethod(lambda *args, **kwargs: None)
    )
    pw._multi_curve_manager._recreate_curve("a")
    assert pw.curves["a"] is old_ci
    assert old_curve in pw.plot_item.scene().items()
    assert new_curve not in pw.plot_item.scene().items()


def test_recreate_keeps_pinned_symbol_policy(plot_factory, qapp):
    pw = _make_pw(plot_factory)
    old_ci = pw.curves["a"]
    old_ci.symbol_policy = SYMBOL_POLICY_ON
    _invalidate(pw)
    pw._multi_curve_manager._recreate_curve("a")
    new_ci = pw.curves["a"]
    assert new_ci is not old_ci, "重建必须真发生（add 失败被吞时旧 ci 保留会空转）"
    assert new_ci.symbol_policy == SYMBOL_POLICY_ON


def test_recreate_keeps_hidden_visible(plot_factory, qapp):
    pw = _make_pw(plot_factory)
    ci = pw.curves["a"]
    ci.visible = False
    if ci.curve is not None:
        ci.curve.setVisible(False)
    _invalidate(pw)
    pw._multi_curve_manager._recreate_curve("a")
    new_ci = pw.curves["a"]
    assert new_ci is not ci
    assert new_ci.visible is False, "隐藏曲线重建后不得变可见"
    if new_ci.curve is not None:
        assert new_ci.curve.isVisible() is False
    # 评审 B-1：legend 投影必须对齐隐藏态（□ 而非 ■）
    html = pw.legend_label.toHtml()
    assert "□" in html, f"legend 应显示隐藏标记 □，实际={html[:200]}"


def test_recreate_emits_curves_changed_after_restore(plot_factory, qapp):
    """评审 B-1/NB-5：恢复后补发 curves_changed，且最后一次事件必须观察到
    恢复后的值（钉时序，防「两次 emit 都在恢复前」的错误实现）。"""
    pw = _make_pw(plot_factory)
    pw.curves["a"].symbol_policy = SYMBOL_POLICY_ON
    _invalidate(pw)
    events = []
    pw.curves_changed.connect(
        lambda: events.append(pw.curves["a"].symbol_policy)
    )
    pw._multi_curve_manager._recreate_curve("a")
    assert len(events) >= 2, f"add 一次 + 恢复后补发一次，实际={len(events)}"
    assert events[-1] == SYMBOL_POLICY_ON, f"最后一次 emit 须观察到恢复值：{events}"


def test_recreate_mode_only_also_refreshes_projection(plot_factory, qapp):
    """评审 NB-1：仅 draw_mode 钉住（preset/policy=auto、可见）也须收尾刷新。"""
    from src.core.data_types import DRAW_MODE_SOURCE_USER, DRAW_MODE_STEP_POST

    pw = _make_pw(plot_factory)
    ci = pw.curves["a"]
    ci.draw_mode = DRAW_MODE_STEP_POST
    ci.draw_mode_source = DRAW_MODE_SOURCE_USER
    _invalidate(pw)
    events = []
    pw.curves_changed.connect(lambda: events.append(pw.curves["a"].draw_mode))
    pw._multi_curve_manager._recreate_curve("a")
    new_ci = pw.curves["a"]
    assert new_ci is not ci
    assert new_ci.draw_mode == DRAW_MODE_STEP_POST
    assert new_ci.draw_mode_source == DRAW_MODE_SOURCE_USER
    assert events and events[-1] == DRAW_MODE_STEP_POST, (
        f"mode-only 形态也须收尾 emit 且观察到恢复值：{events}"
    )


def test_recreate_keeps_preset_and_policy_together(plot_factory, qapp):
    pw = _make_pw(plot_factory)
    ci = pw.curves["a"]
    ci.line_preset = LINE_PRESET_BOLD
    ci.symbol_policy = SYMBOL_POLICY_OFF
    _invalidate(pw)
    pw._multi_curve_manager._recreate_curve("a")
    new_ci = pw.curves["a"]
    assert new_ci is not ci
    assert new_ci.line_preset == LINE_PRESET_BOLD
    assert new_ci.symbol_policy == SYMBOL_POLICY_OFF
    # 评审 NB-2：渲染层真重算——policy OFF 时 3 点稀疏数据不得显 symbol
    if new_ci.curve is not None:
        assert new_ci.curve.opts.get("symbol") is None


def test_recreate_auto_policy_not_over_pinned(plot_factory, qapp):
    """正面护栏：auto 策略重建后仍 auto（不得误钉 user 状态）。"""
    pw = _make_pw(plot_factory)
    assert pw.curves["a"].symbol_policy == SYMBOL_POLICY_AUTO
    _invalidate(pw)
    pw._multi_curve_manager._recreate_curve("a")
    assert pw.curves["a"].symbol_policy == SYMBOL_POLICY_AUTO
    assert pw.curves["a"].line_preset == LINE_PRESET_AUTO
    assert pw.curves["a"].visible is True