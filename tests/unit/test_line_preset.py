"""样式预设（line_preset）单元层：resolve_line_pen 解析矩阵与默认值。

矩阵 = 4 预设 × symbol 显/隐 × 选中/未选（16 格）。
选中高亮 = 生效基础宽度 +2；auto 态 1+2=3 / 2+2=4 与
HIGHLIGHT_THIN/LINE_WIDTH 数值一致（既有行为恒等）。
"""

from types import SimpleNamespace

import numpy as np
import pytest

from src.core.config import (
    HIGHLIGHT_LINE_WIDTH,
    HIGHLIGHT_THIN_LINE_WIDTH,
    THICK_LINE_WIDTH,
    THIN_LINE_WIDTH,
)
from src.core.data_types import (
    LINE_PRESET_AUTO,
    LINE_PRESET_BOLD,
    LINE_PRESET_DASH,
    LINE_PRESET_DOT,
    CurveInfo,
    resolve_line_pen,
)


@pytest.mark.parametrize(
    "preset, show_symbols, highlighted, expected",
    [
        # auto：现有状态机（细 1 / 粗 2），选中 +2
        (LINE_PRESET_AUTO, True, False, (THIN_LINE_WIDTH, "solid")),
        (LINE_PRESET_AUTO, False, False, (THICK_LINE_WIDTH, "solid")),
        (LINE_PRESET_AUTO, True, True, (HIGHLIGHT_THIN_LINE_WIDTH, "solid")),
        (LINE_PRESET_AUTO, False, True, (HIGHLIGHT_LINE_WIDTH, "solid")),
        # bold：恒 4，选中 6，不随 symbol 态切换
        (LINE_PRESET_BOLD, True, False, (4, "solid")),
        (LINE_PRESET_BOLD, False, False, (4, "solid")),
        (LINE_PRESET_BOLD, True, True, (6, "solid")),
        (LINE_PRESET_BOLD, False, True, (6, "solid")),
        # dash：恒 2，选中 4
        (LINE_PRESET_DASH, True, False, (2, "dash")),
        (LINE_PRESET_DASH, False, False, (2, "dash")),
        (LINE_PRESET_DASH, True, True, (4, "dash")),
        (LINE_PRESET_DASH, False, True, (4, "dash")),
        # dot：恒 2（低 DPI 护栏），选中 4
        (LINE_PRESET_DOT, True, False, (2, "dot")),
        (LINE_PRESET_DOT, False, False, (2, "dot")),
        (LINE_PRESET_DOT, True, True, (4, "dot")),
        (LINE_PRESET_DOT, False, True, (4, "dot")),
    ],
)
def test_resolve_line_pen_matrix(preset, show_symbols, highlighted, expected):
    ci = SimpleNamespace(line_preset=preset)
    assert resolve_line_pen(ci, show_symbols, highlighted) == expected


def test_resolve_line_pen_unknown_value_falls_back_to_auto():
    """防御：未知预设值按 auto 状态机处理（与 step_mode_for 同口径）。"""
    ci = SimpleNamespace(line_preset="garbage")
    assert resolve_line_pen(ci, False, False) == (THICK_LINE_WIDTH, "solid")


def test_resolve_line_pen_missing_field_defaults_auto():
    ci = SimpleNamespace()
    assert resolve_line_pen(ci, True, False) == (THIN_LINE_WIDTH, "solid")


def test_curve_info_default_preset():
    ci = CurveInfo(
        var_name="v",
        curve=None,
        x_data=np.arange(5, dtype=float),
        y_data=np.zeros(5),
    )
    assert ci.line_preset == LINE_PRESET_AUTO