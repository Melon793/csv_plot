"""decide_draw_mode 单元测试：枚举通道自动阶梯判定（保守版 v1）。

规则见 src/core/data_types.py::decide_draw_mode docstring。
保守边界（回归防线，防将来被顺手加回来）：
- 不做统计推断（唯一值/零差分/整数率）
- 不做 bool 推断（bit_count==1 不判）
- CSV/Excel（无元数据）一律直线
- step-pre 永不自动产出
"""

from __future__ import annotations

import pytest

from src.core.data_types import (
    DRAW_MODE_AUTO,
    DRAW_MODE_STEP_POST,
    DRAW_MODE_STEP_PRE,
    decide_draw_mode,
    step_mode_for,
)
from src.data.metadata import VarMetadata


def _meta(**overrides) -> VarMetadata:
    """构造测试用 VarMetadata（默认：连续量通道）。"""
    defaults = dict(name="v", unit="V")
    defaults.update(overrides)
    return VarMetadata(**defaults)


class TestDecideDrawMode:
    def test_enum_channel_steps(self):
        # 枚举通道（含 RTABX 场景：is_enum=True 但 enum_map=None）→ step-post
        m = _meta(is_enum=True)
        assert decide_draw_mode(m) == DRAW_MODE_STEP_POST

    def test_enum_channel_without_text_map_still_steps(self):
        # RTABX：is_enum=True、enum_map 提取失败（None）→ 仍判 step
        # （绘图数据是 raw 码值，离散无中间态，阶梯语义正确）
        m = _meta(is_enum=True, enum_map=None)
        assert decide_draw_mode(m) == DRAW_MODE_STEP_POST

    def test_continuous_channel_line(self):
        # 连续量（有单位、有采样率）→ 直线
        m = _meta(nominal_raster_s=0.01, effective_rate_hz=100.0)
        assert decide_draw_mode(m) == DRAW_MODE_AUTO

    def test_no_metadata_line(self):
        # meta 为 None（CSV/Excel loader 无 get_metadata、查询失败）→ 直线
        assert decide_draw_mode(None) == DRAW_MODE_AUTO
        assert decide_draw_mode(None, y_format="enum") == DRAW_MODE_AUTO

    def test_y_format_enum_fallback(self):
        # y_format=='enum' 双保险：即使 meta.is_enum 为 False 也判 step
        # （覆盖 meta 接口与 y_format 来源不一致的调用方）
        assert decide_draw_mode(_meta(), y_format="enum") == DRAW_MODE_STEP_POST

    @pytest.mark.parametrize(
        "field", ["is_time_channel", "is_date", "is_time_of_day"]
    )
    def test_time_channels_defensive_veto(self, field):
        # 时间/日期通道防御性否决：即使 is_enum 异常为真也保持直线
        m = _meta(is_enum=True, **{field: True})
        assert decide_draw_mode(m) == DRAW_MODE_AUTO

    def test_switch_off_disables(self, monkeypatch):
        # 总开关关闭 → 一切回到直线（decide_draw_mode 内延迟 import，
        # monkeypatch config 模块属性即可生效）
        import src.core.config as config_mod

        monkeypatch.setattr(config_mod, "DRAW_MODE_ENUM_AS_STEP", False)
        m = _meta(is_enum=True)
        assert decide_draw_mode(m) == DRAW_MODE_AUTO
        assert decide_draw_mode(m, y_format="enum") == DRAW_MODE_AUTO

    def test_step_pre_never_auto(self):
        # 防线：自动判定只可能产出 auto/step-post，永不产出 step-pre
        m = _meta(is_enum=True)
        result = decide_draw_mode(m, y_format="enum")
        assert result != DRAW_MODE_STEP_PRE
        assert step_mode_for(result) in (None, "right")


class TestConservativeBoundary:
    """保守边界回归：这些输入在 v1 设计下必须保持直线。"""

    def test_csv_like_no_inference(self):
        # 无元数据 = 无判定（不读数据、不看列名、不猜）
        assert decide_draw_mode(None, y_format="number") == DRAW_MODE_AUTO

    def test_one_bit_channel_not_inferred(self):
        # bool 推断防线：bit_count==1 不判（MDF 规范无 bool 类型，
        # VarMetadata 也不携带位长；刻意保守）
        m = _meta(unit="")
        assert decide_draw_mode(m) == DRAW_MODE_AUTO

    def test_boolean_like_int_channel_not_inferred(self):
        # 0/1 整数列防线：唯一值统计/零差分统计一律不做
        m = _meta(unit="")
        assert decide_draw_mode(m, y_format="number") == DRAW_MODE_AUTO
