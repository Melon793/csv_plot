"""引导线可读性护栏（offscreen component）。

密集单列时 elbow 竖段共用同一 x 会叠成线束，因此：
- 最内列（LabelLayout.first_col=True）用两点直线，斜率区分归属；
- 第二列及更远仍用四点正交折线（直线会横穿第一列标签）；
- 左侧列端点取标签右缘，避免线压进标签下方；
- 贴住交点（manhattan ≤ 8px）与 pinned 项不画。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from PySide6.QtCore import QPointF

from src.ui.widgets.cursor_manager import LabelLayout
from tests.component.test_cursor_label_edge import (
    _make_peak_df,
    _refresh,
    _setup,
    _visible_labels,
    _x_where,
)


def _leader_scene_pts(pw, leader) -> list[QPointF]:
    vb = pw.view_box
    xs, ys = leader.getData()
    return [vb.mapViewToScene(QPointF(x, y)) for x, y in zip(xs, ys)]


def _text_width(pw, lbl) -> float:
    fm = pw._cursor_manager._cached_font_metrics
    return fm.horizontalAdvance(lbl.toPlainText()) + 12


def test_first_column_leader_is_straight(plot_factory, qapp):
    """最内列：两点直线；端点落交点与标签左缘；线对比度提升"""
    pw = _setup(plot_factory, _make_peak_df(), size=(500, 420))
    vb = pw.view_box
    vb.setXRange(0, 99, padding=0.05)
    vb.setYRange(0, 200, padding=0)
    qapp.processEvents()
    peak_x = _x_where(pw, "y", 100.0)
    pw.vline.setVisible(True)
    pw.vline.setPos(peak_x)
    _refresh(pw)

    labels = _visible_labels(pw)
    assert len(labels) == 1
    leaders = [ld for ld in pw._cursor_item_pool["leaders"] if ld.isVisible()]
    assert leaders, "标签离交点 >8px 时应有引导线"
    leader = leaders[0]
    xs, _ = leader.getData()
    assert len(xs) == 2, "最内列应为两点直线（不再走 elbow）"

    pts = _leader_scene_pts(pw, leader)
    cross = vb.mapViewToScene(QPointF(peak_x, 100.0))
    assert abs(pts[0].x() - cross.x()) < 1e-6
    assert abs(pts[0].y() - cross.y()) < 1e-6
    assert abs(pts[-1].x() - labels[0].scenePos().x()) < 1e-6
    assert abs(pts[-1].y() - labels[0].scenePos().y()) < 1e-6

    pen = leader.opts.get("pen") or leader.pen
    assert pen.color().alpha() == 200
    assert pen.widthF() == pytest.approx(1.25)


def test_elbow_second_column_uses_four_points(plot_factory, qapp):
    """第二列（first_col=False）：仍为四点折线，竖段在标签左缘外侧 6px"""
    pw = _setup(plot_factory, _make_peak_df(), size=(500, 420))
    vb = pw.view_box
    vb.setXRange(0, 99, padding=0.05)
    vb.setYRange(0, 200, padding=0)
    qapp.processEvents()
    peak_x = _x_where(pw, "y", 100.0)
    cross = vb.mapViewToScene(QPointF(peak_x, 100.0))
    mgr = pw._cursor_manager
    mgr._clear_cursor_items()
    layout = LabelLayout(
        layout_x=cross.x() + 120, layout_y=cross.y() + 40, index=0,
        x_pos=peak_x, y_pos=100.0, y_value="100",
        color=pw.curves["y"].color, first_col=False,
    )
    mgr._render_label_layout([layout])

    leaders = [ld for ld in pw._cursor_item_pool["leaders"] if ld.isVisible()]
    assert leaders
    leader = leaders[0]
    xs, ys = leader.getData()
    assert len(xs) == 4
    assert abs(xs[1] - xs[2]) < 1e-9
    pts = _leader_scene_pts(pw, leader)
    assert abs(pts[1].x() - (layout.layout_x - 6)) < 1e-6, "竖段 = 左缘 − 6"
    assert abs(pts[-1].x() - layout.layout_x) < 1e-6, "终点 = 左缘"


def test_left_column_straight_ends_at_right_edge(plot_factory, qapp):
    """标签在光标左侧（最内列）：直线终点取标签右缘，不压进标签"""
    df = pd.DataFrame({"y": np.full(100, 100.0)})
    pw = _setup(plot_factory, df, size=(500, 420))
    vb = pw.view_box
    vb.setXRange(0, 99, padding=0.05)
    vb.setYRange(0, 200, padding=0)
    qapp.processEvents()
    pw.vline.setVisible(True)
    pw.vline.setPos(95.0)  # 右侧放不下一列 → 标签放到左侧
    _refresh(pw)

    labels = _visible_labels(pw)
    assert len(labels) == 1
    lbl = labels[0]
    cross = vb.mapViewToScene(QPointF(95.0, 100.0))
    assert lbl.scenePos().x() < cross.x(), "前置：标签应在光标左侧"

    leaders = [ld for ld in pw._cursor_item_pool["leaders"] if ld.isVisible()]
    assert leaders
    leader = leaders[0]
    xs, _ = leader.getData()
    assert len(xs) == 2, "最内列直线"
    pts = _leader_scene_pts(pw, leader)
    right_edge = lbl.scenePos().x() + _text_width(pw, lbl)
    assert abs(pts[-1].x() - right_edge) < 1e-6, "终点应为标签右缘"
    assert abs(pts[-1].y() - lbl.scenePos().y()) < 1e-6


def test_elbow_left_second_column_uses_right_edge(plot_factory, qapp):
    """左侧第二列（first_col=False）：竖段 = 右缘 + 6，终点回右缘"""
    pw = _setup(plot_factory, _make_peak_df(), size=(500, 420))
    vb = pw.view_box
    vb.setXRange(0, 99, padding=0.05)
    vb.setYRange(0, 200, padding=0)
    qapp.processEvents()
    peak_x = _x_where(pw, "y", 100.0)
    cross = vb.mapViewToScene(QPointF(peak_x, 100.0))
    mgr = pw._cursor_manager
    mgr._clear_cursor_items()
    layout = LabelLayout(
        layout_x=cross.x() - 150, layout_y=cross.y() + 40, index=0,
        x_pos=peak_x, y_pos=100.0, y_value="100",
        color=pw.curves["y"].color, first_col=False,
    )
    mgr._render_label_layout([layout])

    leaders = [ld for ld in pw._cursor_item_pool["leaders"] if ld.isVisible()]
    assert leaders
    leader = leaders[0]
    xs, _ = leader.getData()
    assert len(xs) == 4
    pts = _leader_scene_pts(pw, leader)
    text_width = mgr._cached_font_metrics.horizontalAdvance("100") + 12
    right_edge = layout.layout_x + text_width
    assert abs(pts[1].x() - (right_edge + 6)) < 1e-6
    assert abs(pts[-1].x() - right_edge) < 1e-6


def test_no_leader_when_label_touches_cross(plot_factory, qapp):
    """标签中心与交点重合（manhattan ≤ 8px）：不画引导线"""
    pw = _setup(plot_factory, _make_peak_df(), size=(500, 420))
    vb = pw.view_box
    vb.setXRange(0, 99, padding=0.05)
    vb.setYRange(0, 200, padding=0)
    qapp.processEvents()
    peak_x = _x_where(pw, "y", 100.0)
    cross = vb.mapViewToScene(QPointF(peak_x, 100.0))

    pw._cursor_manager._clear_cursor_items()
    layout = LabelLayout(
        layout_x=cross.x(), layout_y=cross.y(), index=0,
        x_pos=peak_x, y_pos=100.0, y_value="100",
        color=pw.curves["y"].color, first_col=True,
    )
    pw._cursor_manager._render_label_layout([layout])

    assert not [ld for ld in pw._cursor_item_pool["leaders"] if ld.isVisible()], (
        "贴住交点时不应画引导线"
    )


def test_pinned_labels_have_no_leader(plot_factory, qapp):
    """pinned 边缘标签不画引导线（避免横穿视口）"""
    df = pd.DataFrame(
        {f"y{i}": np.full(100, float(v)) for i, v in enumerate((0, 100, 200, 300, 400))}
    )
    pw = _setup(plot_factory, df, size=(500, 500))
    vb = pw.view_box
    vb.setXRange(0, 99, padding=0.05)
    vb.setYRange(450, 550, padding=0)  # 全部交点出界 → 全部 pinned
    qapp.processEvents()
    pw.vline.setVisible(True)
    pw.vline.setPos(50.0)
    _refresh(pw, times=4)

    labels = _visible_labels(pw)
    assert len(labels) == 5
    assert all(lbl.toPlainText().startswith("▼") for lbl in labels)
    assert not [ld for ld in pw._cursor_item_pool["leaders"] if ld.isVisible()], (
        "pinned 标签不应画引导线"
    )
