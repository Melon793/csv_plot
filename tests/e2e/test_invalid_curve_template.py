"""失效占位的模板 / auto-save / 导出往返 e2e（方案 B2 Commit 4，真实 MainWindow）。

钉住 §8 契约：
- 导出：占位变量名写入 PlotConfig.invalid_curves；纯占位格子不被 _export_page 跳过；
- 应用：模板中不存在的变量降级为占位（固定灰色）；模板自带 invalid_curves 忠实还原；
- 往返：导出→应用→再导出，占位资产不丢不重；
- 兼容：旧模板（无 invalid_curves 键）零回归；
- apply_config 完整还原：既有占位随旧曲线一并清除（clear_invalid=True），不残留。
"""

from __future__ import annotations

import json

from src.core.config import INVALID_CURVE_COLOR
from src.core.plot_config import PlotSessionConfig


def _pw(page, index=0):
    return page.plot_widgets[index].plot_widget


def test_export_writes_placeholder_names(loaded_window, qtbot):
    """占位变量名进 invalid_curves，与真实曲线并列导出（§8.2）"""
    mw = loaded_window
    first = mw.current_page
    assert _pw(first).plot_variable("speed")
    _pw(first).demote_to_invalid("ghost", "red", True)

    config = mw.plot_config_manager.export_current_config(mw)
    plots = config.pages[0].plots

    assert len(plots) == 1
    assert plots[0].curves == ["speed"]
    assert plots[0].invalid_curves == ["ghost"]


def test_export_does_not_skip_pure_placeholder_cell(loaded_window, qtbot):
    """纯占位格子（无真实曲线）不被 _export_page 跳过（§8.2/R16）

    旧实现判空只看 get_curve_names()，纯占位格子会被静默丢弃 → 资产丢失。
    """
    mw = loaded_window
    first = mw.current_page
    # 只放占位、不放真实曲线
    _pw(first).demote_to_invalid("lonely", "blue", False)
    assert list(_pw(first).curves) == []

    config = mw.plot_config_manager.export_current_config(mw)
    plots = config.pages[0].plots

    assert len(plots) == 1, "纯占位格子被静默丢弃"
    assert plots[0].curves == []
    assert plots[0].invalid_curves == ["lonely"]


def test_export_roundtrip_through_json(loaded_window, qtbot):
    """导出 → JSON 往返 → invalid_curves 保真（存储层序列化路径）"""
    mw = loaded_window
    first = mw.current_page
    assert _pw(first).plot_variable("speed")
    _pw(first).demote_to_invalid("ghost", "red", True)

    config = mw.plot_config_manager.export_current_config(mw)
    restored = PlotSessionConfig.from_dict(json.loads(json.dumps(config.to_dict())))

    assert restored.pages[0].plots[0].curves == ["speed"]
    assert restored.pages[0].plots[0].invalid_curves == ["ghost"]


def test_apply_demotes_missing_variable_to_gray_placeholder(loaded_window, qtbot):
    """应用模板：不存在的变量降级为占位，落固定灰色（§8.3/§12.2）"""
    mw = loaded_window
    payload = {
        "layout_rows": 1, "layout_cols": 1,
        "plots": [{"curves": ["speed", "no_such_var"]}],
    }
    config = PlotSessionConfig.from_dict(payload)

    assert mw.plot_config_manager.apply_config(mw, config) is True
    qtbot.wait(50)

    pw0 = _pw(mw.current_page)
    assert "speed" in pw0.curves
    assert "no_such_var" in pw0.invalid_curves
    assert pw0.invalid_curves["no_such_var"].color == INVALID_CURVE_COLOR
    assert not (set(pw0.curves) & set(pw0.invalid_curves))


def test_apply_restores_template_own_placeholders(loaded_window, qtbot):
    """模板自带 invalid_curves：忠实还原为占位（固定灰色）"""
    mw = loaded_window
    payload = {
        "layout_rows": 1, "layout_cols": 1,
        "plots": [{"curves": ["speed"], "invalid_curves": ["saved_ghost"]}],
    }
    config = PlotSessionConfig.from_dict(payload)

    assert mw.plot_config_manager.apply_config(mw, config) is True
    qtbot.wait(50)

    pw0 = _pw(mw.current_page)
    assert "speed" in pw0.curves
    assert "saved_ghost" in pw0.invalid_curves
    assert pw0.invalid_curves["saved_ghost"].color == INVALID_CURVE_COLOR


def test_apply_pure_placeholder_cell(loaded_window, qtbot):
    """纯占位格子（curves 空、仅 invalid_curves）：applied_count 不增但占位仍建"""
    mw = loaded_window
    payload = {
        "layout_rows": 1, "layout_cols": 1,
        "plots": [{"curves": [], "invalid_curves": ["only_ghost"]}],
    }
    config = PlotSessionConfig.from_dict(payload)

    assert mw.plot_config_manager.apply_config(mw, config) is True
    qtbot.wait(50)

    pw0 = _pw(mw.current_page)
    assert list(pw0.curves) == []
    assert "only_ghost" in pw0.invalid_curves


def test_export_apply_roundtrip_preserves_placeholders(loaded_window, qtbot):
    """导出→打乱→应用：占位资产不丢不重（§11 往返验收）"""
    mw = loaded_window
    first = mw.current_page
    assert _pw(first).plot_variable("speed")
    _pw(first).demote_to_invalid("ghost", "red", True)

    config = mw.plot_config_manager.export_current_config(mw)
    restored = PlotSessionConfig.from_dict(json.loads(json.dumps(config.to_dict())))

    # 打乱现状：清掉曲线与占位
    _pw(first).clear_plot_item(clear_invalid=True)
    assert not _pw(first).curves and not _pw(first).invalid_curves

    assert mw.plot_config_manager.apply_config(mw, restored) is True
    qtbot.wait(50)

    pw0 = _pw(first)
    assert set(pw0.curves) == {"speed"}
    assert set(pw0.invalid_curves) == {"ghost"}
    assert not (set(pw0.curves) & set(pw0.invalid_curves))


def test_apply_clears_preexisting_placeholders(loaded_window, qtbot):
    """模板=完整还原：apply 前既有占位随旧曲线一并清除，不残留上一会话资产（§7）"""
    mw = loaded_window
    first = mw.current_page
    # 先制造一个上一会话的占位（模板未列出）
    _pw(first).demote_to_invalid("stale_ghost", "red", True)
    assert "stale_ghost" in _pw(first).invalid_curves

    # 应用一个只含 speed 的模板
    payload = {"layout_rows": 1, "layout_cols": 1, "plots": [{"curves": ["speed"]}]}
    config = PlotSessionConfig.from_dict(payload)
    assert mw.plot_config_manager.apply_config(mw, config) is True
    qtbot.wait(50)

    pw0 = _pw(first)
    assert set(pw0.curves) == {"speed"}
    # 未列出的 stale_ghost 不残留（完整还原语义）
    assert "stale_ghost" not in pw0.invalid_curves
    assert list(pw0.invalid_curves) == []


def test_old_template_without_invalid_key_zero_regression(loaded_window, qtbot):
    """旧模板（无 invalid_curves 键）：零回归，正常还原曲线、无占位"""
    mw = loaded_window
    payload = {
        "created_at": "2026-01-01T00:00:00",
        "layout_rows": 1, "layout_cols": 1,
        "time_factor": 1.0, "time_offset": 0.0,
        "plots": [{"curves": ["speed"]}],
    }
    config = PlotSessionConfig.from_dict(payload)
    assert config.plots[0].invalid_curves == []  # 缺省容错

    assert mw.plot_config_manager.apply_config(mw, config) is True
    qtbot.wait(50)

    pw0 = _pw(mw.current_page)
    assert set(pw0.curves) == {"speed"}
    assert list(pw0.invalid_curves) == []
