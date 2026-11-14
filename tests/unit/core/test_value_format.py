"""游标 X 标签精度工具函数的单元测试。

``decimals_for_step`` / ``format_fixed_trim`` 是游标与标记统计窗口共用的
唯一精度口径，这里锁住边界值与既有 factor 语义（1 -> 0 位、0.1 -> 1 位）。
"""

from __future__ import annotations

import math

import pytest

from src.core.value_format import (
    MAX_X_DECIMALS,
    decimals_for_step,
    format_fixed_trim,
)


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        (1.0, 0),
        (2.0, 0),
        (0.5, 1),
        (0.1, 1),
        (0.01, 2),
        (0.001, 3),
        (0.002, 3),
        (1e-7, 7),
        (1e-12, MAX_X_DECIMALS),
    ],
)
def test_decimals_for_step(step, expected):
    assert decimals_for_step(step) == expected


@pytest.mark.parametrize("step", [0.0, -1.0, float("nan"), float("inf")])
def test_decimals_for_step_invalid(step):
    assert decimals_for_step(step) == 0


def test_decimals_for_step_respects_cap():
    assert decimals_for_step(1e-12, cap=6) == 6


@pytest.mark.parametrize(
    ("value", "decimals", "expected"),
    [
        (1370.001, 3, "1370.001"),
        (1370.100, 3, "1370.1"),
        (1370.000, 3, "1370"),
        (1370.6, 0, "1371"),
        (1370.0, 0, "1370"),
        (-0.0, 3, "0"),
        (-0.0, 0, "0"),
        (0.0, 2, "0"),
        (1.5e-7, 7, "0.0000001"),
    ],
)
def test_format_fixed_trim(value, decimals, expected):
    assert format_fixed_trim(value, decimals) == expected


def test_max_x_decimals_is_sane():
    assert 0 < MAX_X_DECIMALS <= 12
    assert math.isfinite(MAX_X_DECIMALS)
