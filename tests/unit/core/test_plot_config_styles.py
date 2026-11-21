"""曲线外观束（curve_styles）schema 单元层：往返 / 旧档兼容 / 畸形守卫。

束只承载偏离默认值（color 恒写例外）；auto 值三字段在 from_dict 丢弃
（放行 draw_mode:'auto' 会在套用端误钉 source='user'）。
"""

from types import SimpleNamespace

import pytest

from src.core.curve_strategy import UnifiedCurveStrategy
from src.core.data_types import (
    DRAW_MODE_SOURCE_AUTO,
    DRAW_MODE_SOURCE_USER,
    LINE_PRESET_DASH,
    SYMBOL_POLICY_ON,
)
from src.core.plot_config import PlotConfig


def test_roundtrip_full_bundle():
    cfg = PlotConfig(
        curves=["speed"],
        curve_styles={
            "speed": {
                "color": "#D62728",
                "line_preset": LINE_PRESET_DASH,
                "draw_mode": "step-post",
                "symbol_policy": SYMBOL_POLICY_ON,
            }
        },
    )
    restored = PlotConfig.from_dict(cfg.to_dict())
    assert restored.curve_styles == cfg.curve_styles


def test_to_dict_without_styles_matches_legacy_keys():
    """无 styles 时序列化逐键与旧 schema 一致（旧档/旧软件零回归）。"""
    cfg = PlotConfig(curves=["a"], invalid_curves=["b"], cell_index=1,
                     cell_row=0, cell_col=1)
    assert set(cfg.to_dict()) == {
        "curves", "invalid_curves", "cell_index", "cell_row", "cell_col"
    }
    assert PlotConfig.from_dict({"curves": ["a"]}).curve_styles == {}


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "garbage",
        ["speed"],
        {"speed": "garbage"},
        {"speed": {"color": 123}},
        {"speed": {"color": "not-a-color"}},
        {"speed": {"line_preset": "diagonal"}},
        {"speed": {"line_preset": "auto"}},
        {"speed": {"draw_mode": "diagonal"}},
        {"speed": {"symbol_policy": "auto"}},
        {"speed": {"symbol_policy": "sometimes"}},
    ],
)
def test_from_dict_guards_drop_invalid(raw):
    assert PlotConfig.from_dict({"curves": ["speed"], "curve_styles": raw}).curve_styles == {}


def test_from_dict_keeps_valid_partial_entry():
    cfg = PlotConfig.from_dict({
        "curves": ["speed"],
        "curve_styles": {"speed": {"color": "red", "line_preset": "auto"}},
    })
    assert cfg.curve_styles == {"speed": {"color": "red"}}


def test_user_pinned_draw_mode_auto_survives_roundtrip():
    """user 钉住的显式直线（枚举通道 override）跨模板链保持，往返幂等。"""
    cfg = PlotConfig(
        curves=["enum_ch"],
        curve_styles={"enum_ch": {"color": "red", "draw_mode": "auto"}},
    )
    restored = PlotConfig.from_dict(cfg.to_dict())
    assert restored.curve_styles == cfg.curve_styles
    assert PlotConfig.from_dict(restored.to_dict()).curve_styles == cfg.curve_styles


def test_strategy_styles_color_always_others_pinned_only():
    ci = SimpleNamespace(
        color="#1658DB",
        line_preset="auto",
        draw_mode="step-post",
        draw_mode_source=DRAW_MODE_SOURCE_AUTO,
        symbol_policy="auto",
    )
    strategy = UnifiedCurveStrategy(SimpleNamespace(curves={"v": ci}))
    assert strategy.get_curve_styles("v") == {"color": "#1658DB"}
    assert strategy.get_curve_styles("ghost") == {}

    ci.line_preset = LINE_PRESET_DASH
    ci.draw_mode_source = DRAW_MODE_SOURCE_USER
    ci.symbol_policy = SYMBOL_POLICY_ON
    assert strategy.get_curve_styles("v") == {
        "color": "#1658DB",
        "line_preset": LINE_PRESET_DASH,
        "draw_mode": "step-post",
        "symbol_policy": SYMBOL_POLICY_ON,
    }
