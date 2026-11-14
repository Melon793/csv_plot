"""强调色解析与 token 派生：Windows 跟系统强调色，其余平台品牌蓝。

这里锁的是「颜色语义」而不是观感：ABGR 解析、亮度钳制、tint 派生的
关系与对比度下限；像素级观感交给真机截图验收。
"""

import colorsys
import sys

import pytest

from src.ui import theme


def _rgb(value):
    return theme._hex_to_rgb(value)


def _hue(value):
    return colorsys.rgb_to_hsv(*(c / 255 for c in _rgb(value)))[0]


def _contrast(a, b):
    la, lb = sorted(
        (theme.relative_luminance(a), theme.relative_luminance(b)), reverse=True
    )
    return (la + 0.05) / (lb + 0.05)


def test_abgr_dword_uses_low_bytes():
    # DWM 的 AccentColor 是 0xAABBGGRR：#0078D4 → 0xFFD47800（高位 alpha 忽略）
    assert theme._abgr_to_hex(0xFFD47800) == "#0078D4"
    assert theme._abgr_to_hex(0x00D47800) == "#0078D4"


def test_mix_endpoints():
    assert theme.mix_hex("#000000", "#FFFFFF", 0.0) == "#000000"
    assert theme.mix_hex("#000000", "#FFFFFF", 1.0) == "#FFFFFF"
    assert theme.mix_hex("#000000", "#FFFFFF", 0.5) == "#808080"


def test_lighten_keeps_hue_and_orders_luminance():
    top = theme.lighten_hex(theme.BRAND_BLUE, 1.35)
    bot = theme.lighten_hex(theme.BRAND_BLUE, 0.87)
    base_lum = theme.relative_luminance(theme.BRAND_BLUE)
    assert theme.relative_luminance(top) > base_lum > theme.relative_luminance(bot)
    # 色相不变（HSV 明度缩放只动 V）
    assert abs(_hue(top) - _hue(theme.BRAND_BLUE)) < 0.02
    assert abs(_hue(bot) - _hue(theme.BRAND_BLUE)) < 0.02


def test_clamp_only_darkens_light_colors():
    # Windows 默认强调色本来可读，不动
    assert theme.clamp_luminance("#0078D4") == "#0078D4"
    # 浅黄是 Windows 强调色里最坏的一档：压暗到亮度上限内
    clamped = theme.clamp_luminance("#FFD800")
    assert theme.relative_luminance(clamped) <= theme._MAX_ACCENT_LUM
    assert _rgb(clamped) != _rgb("#FFD800")


def test_non_windows_uses_brand_blue(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert theme._read_windows_accent() is None
    assert theme.resolve_accent() == theme.BRAND_BLUE


def test_windows_reads_accent_then_clamps(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(theme, "_read_windows_accent", lambda: "#FFD800")
    assert theme.resolve_accent() == theme.clamp_luminance("#FFD800")


def test_windows_falls_back_when_registry_unreadable(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(theme, "_read_windows_accent", lambda: None)
    assert theme.resolve_accent() == theme.WINDOWS_FALLBACK


def test_chip_tokens_follow_accent():
    assert theme.CHIP_SEL_BD == theme.ACCENT
    assert theme.CHIP_CUR_BG == theme.TINT_SOFT
    assert theme.CHIP_CUR_BD == theme.BORDER_SOFT
    assert theme.CHIP_SEL_BG_TOP == theme.lighten_hex(theme.ACCENT, 1.35)
    assert theme.CHIP_SEL_BG_BOT == theme.lighten_hex(theme.ACCENT, 0.87)


def test_soft_tokens_sit_between_accent_and_white():
    accent_lum = theme.relative_luminance(theme.ACCENT)
    assert accent_lum < theme.relative_luminance(theme.BORDER_SOFT)
    assert theme.relative_luminance(theme.BORDER_SOFT) < theme.relative_luminance(
        theme.TINT_SOFT
    )
    assert theme.relative_luminance(theme.TINT_SOFT) < 1.0


def test_tint_qcolor_alpha_and_base():
    color = theme.tint_qcolor()
    assert color.alpha() == theme.TINT_ALPHA == 64
    assert color.name().upper() == theme.ACCENT.upper()
    assert theme.accent_qcolor(128).alpha() == 128
    assert theme.tint_qcolor(theme.TINT_SOFT_ALPHA).alpha() == 26


def test_on_accent_text_picks_better_contrast():
    # 不变量：ON_ACCENT 是白字/深字中对比度更高的一档
    # （真实强调色未必都能达到 4.5，例如 #0078D7 的白字是 4.4989）
    accent = theme.ACCENT
    best = max(
        _contrast("#FFFFFF", accent), _contrast(theme.TEXT_PRIMARY, accent)
    )
    assert _contrast(theme.ON_ACCENT, accent) == pytest.approx(best)
    # 品牌蓝本身达到 WCAG AA（白字 5.18:1）
    assert _contrast("#FFFFFF", theme.BRAND_BLUE) >= 4.5
    # 浅色强调色要翻成深字
    assert theme.on_accent_for("#FFD800") == theme.TEXT_PRIMARY
    # Windows 默认蓝：白字略优于深字，取白字（不是简单阈值翻转）
    assert theme.on_accent_for("#0078D7") == "#FFFFFF"
    assert theme.on_accent_for("#0078D4") == "#FFFFFF"
