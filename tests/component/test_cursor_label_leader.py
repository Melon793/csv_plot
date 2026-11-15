"""引导线 elbow 正交路由护栏（offscreen component 测试）。

四点折线：交点 p0 → (mx, p0.y) → (mx, p1.y) → 标签左缘 p1。
标签在右：mx = 标签左缘 − 6；标签在左：mx = 标签右缘 + 6。
贴住交点（manhattan ≤ 8px）与 pinned 项不画。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from PySide6.QtCore import QPointF

from src.ui.widgets.cursor_manager import LabelLayout
from tests.component.test_cursor_label_edge import (
    _make_peak_df,
    _refresh,
    _setup,
    _visible_labels,
    _x_where,
)


def test_elbow_vertical_segment_sits_outside_label_column(plot_factory, qapp):
    """近同 y 簇被 PAV 推开：每条引导线为四点折线，竖段在标签列外侧 6px"""
    df = pd.DataFrame(
        {f"y{i}": np.full(100, 50.0 + i) for i in range(5)}
    )
    pw = _setup(plot_factory, df, size=(500, 260))
    vb = pw.view_box
    vb.setXRange(0, 99, padding=0.05)
    vb.setYRange(40, 60, padding=0)
    qapp.processEvents()
    pw.vline.setVisible(True)
    pw.vline.setPos(50.0)
    _refresh(pw, times=2)

    leaders = pw._cursor_item_pool["leaders"]
    labels = pw._cursor_item_pool["labels"]
    checked = 0
    for leader, label in zip(leaders, labels):
        if not leader.isVisible() or not label.isVisible():
            continue
        xs, ys = leader.getData()
        assert len(xs) == 4, "elbow 引导线应为四点折线"
        assert abs(xs[1] - xs[2]) < 1e-9, "竖段 x 应一致"
        assert abs(ys[0] - ys[1]) < 1e-9, "首段应水平"
        assert abs(ys[2] - ys[3]) < 1e-9, "末段应水平"

        # 该用例标签全部在光标右侧：竖段 = 标签左缘 − 6px
        label_left_scene = label.scenePos().x()
        elbow_scene_x = vb.mapViewToScene(QPointF(xs[1], ys[1])).x()
        assert abs((elbow_scene_x - label_left_scene) + 6.0) < 1e-6
        checked += 1

    assert checked >= 2, f"应至少检查 2 条引导线，实际 {checked}"


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
        color=pw.curves["y"].color,
    )
    pw._cursor_manager._render_label_layout([layout])

    assert not [ld for ld in pw._cursor_item_pool["leaders"] if ld.isVisible()], (
        "贴住交点时不应画引导线"
    )


def test_elbow_left_column_uses_right_edge_guide(plot_factory, qapp):
    """标签在光标左侧：竖段 = 标签右缘 + 6px（右缘按渲染文本宽计算）"""
    df = pd.DataFrame({"y": np.full(100, 100.0)})
    pw = _setup(plot_factory, df, size=(500, 420))
    vb = pw.view_box
    vb.setXRange(0, 99, padding=0.05)
    vb.setYRange(0, 200, padding=0)
    qapp.processEvents()
    pw.vline.setVisible(True)
    pw.vline.setPos(95.0)  # 右侧空间不足一个列宽 → 标签放左侧
    _refresh(pw)

    labels = _visible_labels(pw)
    assert len(labels) == 1
    lbl = labels[0]
    cross = vb.mapViewToScene(QPointF(95.0, 100.0))
    assert lbl.scenePos().x() < cross.x(), "前置：标签应位于光标左侧"

    leaders = [ld for ld in pw._cursor_item_pool["leaders"] if ld.isVisible()]
    assert leaders, "标签离交点 >8px 时应有引导线"
    xs, ys = leaders[0].getData()
    assert len(xs) == 4

    text_width = (
        pw._cursor_manager._cached_font_metrics.horizontalAdvance(
            lbl.toPlainText()
        ) + 12
    )
    elbow_scene_x = vb.mapViewToScene(QPointF(xs[1], ys[1])).x()
    assert abs(elbow_scene_x - (lbl.scenePos().x() + text_width + 6)) < 1e-6
    # 终点仍回标签左缘
    label_left_scene = lbl.scenePos().x()
    end_scene_x = vb.mapViewToScene(QPointF(xs[-1], ys[-1])).x()
    assert abs(end_scene_x - label_left_scene) < 1e-6


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