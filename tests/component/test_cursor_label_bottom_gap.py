"""游标数值标签底边净空护栏（offscreen）。

现场反馈：标签底边贴/压曲线。根因：旧实现把 LABEL_Y_OFFSET 当
「标签中心到交点」的偏移使用，而标签高 21px（布局按 23px 取半高），
底边落在交点 +0.5px。现语义改为底边净空 LABEL_BOTTOM_GAP=6：
ideal_center = 交点 − (布局半高 + GAP)，视觉净空 ≈7px。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import QPointF

from src.ui.widgets.cursor_manager import CursorManager
from tests.component.test_cursor_label_edge import (
    _make_peak_df,
    _refresh,
    _setup,
    _visible_labels,
    _x_where,
)

# LABEL_BOTTOM_GAP(6) + 1px 布局高度补偿（boundingRect 21 → 布局 23）
VISUAL_GAP = 7.0


def _make_cluster_df() -> pd.DataFrame:
    """两条近乎同 y 的斜坡曲线：x=50 处 y=100 / 100.05

    斜坡让 Y 数据范围保持 0~198，避免两条常量曲线把自动范围收到 Δ0.1。
    """
    x = np.arange(100.0)
    return pd.DataFrame({"a": x * 2.0, "b": x * 2.0 + 0.05})


def _wide_y_view(vb) -> None:
    """放开 Y limits 并固定 0~200 视口（产品默认把 limits 收在数据范围）"""
    vb.setLimits(yMin=-1e9, yMax=1e9)
    vb.disableAutoRange()
    vb.setYRange(0, 200, padding=0)


def _bottom_gap(vb, lbl, cross_y: float) -> float:
    """交点 scene y 到标签底边的净空（正值 = 底边在交点上方）"""
    bottom = lbl.scenePos().y() + lbl.boundingRect().height() / 2
    return cross_y - bottom


def test_single_label_bottom_gap_above_curve(plot_factory, qapp):
    """单标签：底边净空 ≈7px，不再压线"""
    pw = _setup(plot_factory, _make_peak_df(), size=(500, 420))
    vb = pw.view_box
    vb.setXRange(0, 99, padding=0.05)
    _wide_y_view(vb)  # 交点居中，避免顶部 clamp 吞掉净空
    qapp.processEvents()
    peak_x = _x_where(pw, "y", 100.0)
    pw.vline.setVisible(True)
    pw.vline.setPos(peak_x)
    _refresh(pw)

    labels = _visible_labels(pw)
    assert len(labels) == 1
    cross_y = vb.mapViewToScene(QPointF(peak_x, 100.0)).y()
    gap = _bottom_gap(vb, labels[0], cross_y)
    assert gap == pytest.approx(VISUAL_GAP, abs=1.5), (
        f"底边净空={gap}px，预期 ≈{VISUAL_GAP}px"
    )
    assert gap > 2.0, "底边必须在交点上方，不得压线"


def test_cluster_top_label_keeps_gap_after_pav(plot_factory, qapp):
    """近同 y 簇：PAV 对称分裂后，最上方标签仍保留净空（不被压回交点）"""
    pw = _setup(plot_factory, _make_cluster_df(), size=(500, 420))
    vb = pw.view_box
    vb.setXRange(0, 99, padding=0.05)
    _wide_y_view(vb)
    qapp.processEvents()
    pw.vline.setVisible(True)
    pw.vline.setPos(50.0)
    _refresh(pw, times=3)  # 越过列数滞回

    labels = sorted(_visible_labels(pw), key=lambda lbl: lbl.scenePos().y())
    assert len(labels) == 2, "两条曲线应各有一个标签"
    cross_top = vb.mapViewToScene(QPointF(50.0, 100.05)).y()
    top_gap = _bottom_gap(vb, labels[0], cross_top)
    # 簇对称分裂会给最上方标签额外上移量，故净空大于单标签场景；
    # 旧公式（中心偏移 10）同场景仅 ≈13px，阈值 15 可捕获回归
    assert top_gap > 15.0, f"簇顶部净空={top_gap}px，疑似回退到中心偏移"
    # 两标签仍保持最小垂直间距
    sep = labels[1].scenePos().y() - labels[0].scenePos().y()
    assert sep >= 21.0, f"标签垂直间距={sep}px，疑似重叠"


def test_pinned_label_ideal_ignores_bottom_gap(plot_factory, qapp):
    """pinned 项仍吸附视口边缘，不叠加底边净空"""
    pw = _setup(plot_factory, _make_peak_df(), size=(500, 420))
    vb = pw.view_box
    vb.setXRange(0, 99, padding=0.05)
    _wide_y_view(vb)
    qapp.processEvents()
    pw.vline.setVisible(True)
    pw.vline.setPos(_x_where(pw, "y", 100.0))
    # 触发一次布局，确保标签高度/字体度量缓存已就绪
    _refresh(pw)
    mgr: CursorManager = pw._cursor_manager
    half = mgr._cached_label_height_pixels / 2
    top, bottom = 50.0, 350.0

    up = mgr._ideal_label_scene_y(
        {"pinned": True, "up": True, "x_pos": 0.0, "y_pos": 0.0},
        pw.view_box, top, bottom, half,
    )
    down = mgr._ideal_label_scene_y(
        {"pinned": True, "up": False, "x_pos": 0.0, "y_pos": 0.0},
        pw.view_box, top, bottom, half,
    )
    assert up == pytest.approx(top + half + 2)
    assert down == pytest.approx(bottom - half - 2)
