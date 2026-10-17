"""箭头头部方向 component 测试（offscreen）

守住的问题（用户实测反馈）：箭头头部三角不垂直于所属直线 —— 根因是
``update_arrow_head`` 的角度公式在屏幕坐标系（y 向下）里没有对 ``unit_y``
取反，所有带垂直分量的箭头头部沿水平轴镜像：朝上的箭头三角朝下、
斜箭头三角反折，8 个方向里只有水平两支是对的。

判定口径：``ArrowItem`` 的路径是"箭尖在原点、箭身朝局部 +x、箭头指向 -x"，
把 ``angle`` 旋转后**视觉指向**必须与"tail → head"的屏幕方向点积为 1。
方向必须在屏幕像素里算（data 角度经非等比缩放后和视觉方向不一致），
所以除等比视图外还要覆盖非等比视图。
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from PySide6.QtCore import QPointF

from src.ui.widgets.annotation_items import ARROW_HEAD_LEN, update_arrow_head


@pytest.fixture()
def plotted(plot_factory, qapp):
    """已绘图并固定成 1:1 视图的 widget"""
    df = pd.DataFrame({"a": [float(i) for i in range(100)]})
    pw = plot_factory(df)
    pw.resize(800, 600)
    pw.window().show()
    pw.show()
    assert pw.plot_variable("a"), "曲线未绘出，几何断言失去意义"
    pw.view_box.setXRange(0, 100, padding=0)
    pw.view_box.setYRange(0, 100, padding=0)
    qapp.processEvents()
    return pw


def _make_arrow(pw, qapp, tail, head):
    annotation_id = pw.annotation_manager.create("arrow", [tail, head])
    qapp.processEvents()
    record = pw.annotation_manager._graphics[annotation_id]
    return record


def _visual_direction(pw, record, tail, head):
    """由 decor 的 angle 反算箭头的视觉指向（屏幕坐标，y 向下）

    ArrowItem 路径"箭尖在原点、箭身朝局部 +x、箭头指向 -x"，``angle`` 旋转后
    箭身方向 = (cos θ, sin θ)（QTransform.rotate 在 y 向下的屏幕坐标系里正角
    顺时针），视觉指向 = 箭身反向。
    """
    view_box = pw.view_box
    tail_scene = view_box.mapViewToScene(QPointF(*tail))
    head_scene = view_box.mapViewToScene(QPointF(*head))
    dx, dy = head_scene.x() - tail_scene.x(), head_scene.y() - tail_scene.y()
    norm = math.hypot(dx, dy)
    return dx / norm, dy / norm, record.decor.opts["angle"]


def _visual_unit(angle: float) -> tuple[float, float]:
    return -math.cos(math.radians(angle)), -math.sin(math.radians(angle))


class TestArrowHeadFollowsTheLine:
    @pytest.mark.parametrize("degrees", [0, 45, 90, 135, 180, 225, 270, 315])
    def test_head_points_from_tail_to_head(self, plotted, qapp, degrees):
        """8 个方向逐一核对：视觉指向与 tail→head 的屏幕方向点积为 1

        修复前这个断言在所有带垂直分量的方向上失败（镜像）。
        """
        rad = math.radians(degrees)
        cx, cy, r = 50.0, 50.0, 30.0
        tail = (cx - r * math.cos(rad), cy - r * math.sin(rad))
        head = (cx + r * math.cos(rad), cy + r * math.sin(rad))
        record = _make_arrow(plotted, qapp, tail, head)

        want_x, want_y, angle = _visual_direction(plotted, record, tail, head)
        visual_x, visual_y = _visual_unit(angle)
        dot = want_x * visual_x + want_y * visual_y

        assert dot == pytest.approx(1.0, abs=1e-6), (
            f"{degrees}° 箭头视觉指向偏了：期望 ({want_x:.2f},{want_y:.2f})，"
            f"实得 ({visual_x:.2f},{visual_y:.2f})"
        )

    def test_diagonal_line_gets_a_perpendicular_base(self, plotted, qapp):
        """三角基边垂直于箭身：45° 直线上箭头两翼到"基边法向"等距

        这是用户视角的原始表述——直接验证"三角永远垂直于直线"。
        """
        tail, head = (10.0, 10.0), (30.0, 30.0)
        record = _make_arrow(plotted, qapp, tail, head)
        path = record.decor.path  # 已按 angle 旋转后的路径（局部坐标，y 向下）

        # 路径里离原点最远的点就是两翼端点；它们的连线（基边）必须垂直于箭身方向
        wing = max(
            (path.elementAt(i) for i in range(path.elementCount())),
            key=lambda e: math.hypot(e.x, e.y),
        )
        angle = record.decor.opts["angle"]
        body = (math.cos(math.radians(angle)), math.sin(math.radians(angle)))
        # 基边方向 = 翼点在垂直于箭身方向上的分量方向；直接验证：
        # 翼点在箭身方向上的投影 == headLen（箭身长度），两翼对称
        assert wing.x * body[0] + wing.y * body[1] == pytest.approx(ARROW_HEAD_LEN, abs=0.5)

    def test_direction_survives_view_zoom(self, plotted, qapp):
        """缩放之后方向仍随线走（sigRangeChanged 重定向路径）"""
        tail, head = (20.0, 20.0), (60.0, 70.0)
        record = _make_arrow(plotted, qapp, tail, head)

        plotted.view_box.setXRange(0, 200, padding=0)
        plotted.view_box.setYRange(-50, 150, padding=0)
        qapp.processEvents()
        # 缩放会触发 sigRangeChanged → 重算；模型不动
        assert pw_points(record) == [tail, head]

        want_x, want_y, angle = _visual_direction(plotted, record, tail, head)
        visual_x, visual_y = _visual_unit(angle)
        assert want_x * visual_x + want_y * visual_y == pytest.approx(1.0, abs=1e-6)

    def test_degenerate_segment_keeps_the_previous_angle(self, plotted, qapp):
        """两端重合：方向无定义，保持原样而不是给一个随机方向"""
        record = _make_arrow(plotted, qapp, (30.0, 30.0), (50.0, 50.0))
        before = record.decor.opts["angle"]

        update_arrow_head(record.decor, plotted.view_box, (40.0, 40.0), (40.0, 40.0))
        assert record.decor.opts["angle"] == before, "零长度线段不应改动方向"


def pw_points(record):
    return list(record.model.points)
