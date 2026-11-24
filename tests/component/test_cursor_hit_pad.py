"""光标线抓取带（命中区）护栏（offscreen component）。

背景：现场反馈固定光标「不好选中/拖动」。pyqtgraph 的命中矩形只由笔宽
推导（InfiniteLine._computeBoundingRect：半宽 = max(pen宽, hoverPen宽)/2 + 1），
4px 笔宽只有 ±3px，点空即被 ViewBox 当平移。GrabPaddedInfiniteLine 只在
boundingRect 上外扩 CURSOR_HIT_PAD_PX（shape 跟随），绘制笔宽不变。

本文件锁住两件事：
1. 命中带宽度 ≥ 目标值（防升级 pyqtgraph 后被悄悄收回）；
2. 绘制笔宽不变粗（用户明确要求：只放宽识别区，红线外观不动）。
"""

from __future__ import annotations

import pyqtgraph as pg
import pytest

from src.ui.widgets.plot_ui_manager import (
    CURSOR_HIT_PAD_PX,
    GrabPaddedInfiniteLine,
)

PEN_WIDTH = 4
# 视觉半宽（pen 4 → 2） + pyqtgraph 固定 +1 → 视觉侧 3px，再加外扩
BASE_HALF_PX = PEN_WIDTH / 2 + 1


def _setup(plot_factory):
    pw = plot_factory()
    pw.resize(500, 400)
    pw.window().show()
    pw.show()
    pw._cursor_manager  # 确保管理器已构建
    return pw


def _px_per_unit(line) -> float:
    """1 个逻辑像素对应的局部坐标长度（垂直方向）"""
    _, ortho = line.pixelVectors(direction=pg.Point(1, 0))
    assert ortho is not None
    return abs(ortho.y())


def _half_band_px(line) -> float:
    return line.boundingRect().height() / 2 / _px_per_unit(line)


def _shape_contains_px(line, offset_px: float) -> bool:
    """shape 是否覆盖线中心两侧各 offset_px（局部 y 即垂直方向）"""
    d = offset_px * _px_per_unit(line)
    shape = line.shape()
    return all(shape.contains(pg.Point(0, sign * d)) for sign in (1, -1))


def test_both_cursor_lines_are_grab_padded(plot_factory, qapp):
    """单/双光标两条线都换成扩宽命中区的子类"""
    pw = _setup(plot_factory)
    assert isinstance(pw.vline, GrabPaddedInfiniteLine)
    assert isinstance(pw.vline2, GrabPaddedInfiniteLine)


def test_hit_band_pixels(plot_factory, qapp):
    """命中半宽 = 视觉半宽 + 外扩；且实际可命中的最远偏移与之一致"""
    pw = _setup(plot_factory)
    line = pw.vline
    expected_half = BASE_HALF_PX + CURSOR_HIT_PAD_PX

    assert _half_band_px(line) == pytest.approx(expected_half, abs=0.6)
    # 几何证据：外扩范围内可命中，超出则不可（不是"代码里算了就算数"）
    assert _shape_contains_px(line, expected_half - 1)
    assert not _shape_contains_px(line, expected_half + 1)


def test_hit_band_is_wider_than_plain_infinite_line(plot_factory, qapp):
    """与同笔宽的裸 InfiniteLine 对照：命中带必须严格更宽（防外扩失效）"""
    pw = _setup(plot_factory)
    plain = pg.InfiniteLine(
        angle=90,
        movable=True,
        pen=pg.mkPen((255, 0, 0, 100), width=PEN_WIDTH),
    )
    pw.plot_item.addItem(plain, ignoreBounds=True)
    try:
        assert _half_band_px(pw.vline) > _half_band_px(plain)
        assert _shape_contains_px(plain, BASE_HALF_PX - 1)
        assert not _shape_contains_px(plain, BASE_HALF_PX + 1)
    finally:
        pw.plot_item.removeItem(plain)


def test_drawn_pen_width_unchanged(plot_factory, qapp):
    """红线外观不变：常态与悬停态的绘制笔宽都保持 4px"""
    pw = _setup(plot_factory)
    line = pw.vline

    assert line.pen.width() == PEN_WIDTH
    assert line.hoverPen.width() == PEN_WIDTH
    assert line.currentPen.width() == PEN_WIDTH

    line.setMouseHover(True)
    try:
        assert line.currentPen.width() == PEN_WIDTH
    finally:
        line.setMouseHover(False)
    assert line.currentPen.width() == PEN_WIDTH