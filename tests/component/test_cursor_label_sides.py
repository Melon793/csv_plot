"""多列标签双侧交替排布（offscreen component）。

2026-11-16 拍板契约：
- 单游标：列方向 右/左/右… 交替，两侧最内列分别贴光标 GAP；
- 某侧无预算（光标贴边）→ 自动退化为单侧；
- 双游标：交替未启用，各组保持旧的右先左后（向外分裂由
  test_cursor_label_columns 覆盖）；
- 引导线：每侧最内列两点直线，外侧列四点 elbow。

Windows 兼容：列数/容量/侧预算均显式 monkeypatch，不依赖字体自然
高度与宽度。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from PySide6.QtCore import QPointF

from src.ui.widgets.cursor_manager import CursorManager
from tests.component.test_cursor_label_edge import (
    _refresh,
    _setup,
    _visible_labels,
)


def _label_df(values) -> pd.DataFrame:
    return pd.DataFrame(
        {f"y{i}": np.full(100, float(v)) for i, v in enumerate(values)}
    )


def _reset_view(pw, x0, x1, y0, y1):
    """放宽 Y limits 后设置视窗（绘图期 limits 会钳制 setYRange）。"""
    pw.plot_item.setLimits(yMin=None, yMax=None)
    vb = pw.view_box
    vb.setXRange(x0, x1, padding=0.05)
    vb.setYRange(y0, y1, padding=0)


def _pin_columns(pw, monkeypatch, columns: int, capacity: int):
    """固定列数与列容量，排除字体/DPI 对列数收敛的影响。"""
    monkeypatch.setattr(
        pw._cursor_manager, "_apply_column_hysteresis",
        lambda new_count, cursor_id=0: columns,
    )
    monkeypatch.setattr(
        pw._cursor_manager, "_column_capacity",
        lambda available, h, gap: capacity,
    )


def _pin_side_budget(monkeypatch, min_available: float, cols: int):
    """固定单侧预算口径：可用宽 ≥ min_available 才有 cols 列。

    必须打在类上：_assign_column_sides 走 CursorManager._possible_columns，
    max_cols 截断走 self._possible_columns，两侧必须同口径。
    """
    def fake(available, col_width, margin=0.0):
        return cols if available >= min_available else 0

    monkeypatch.setattr(
        CursorManager, "_possible_columns", staticmethod(fake)
    )


def _cursor_scene_x(pw, x, y=100.0) -> float:
    return pw.view_box.mapViewToScene(QPointF(x, y)).x()


def _text_width(pw, lbl) -> float:
    fm = pw._cursor_manager._cached_font_metrics
    return fm.horizontalAdvance(lbl.toPlainText()) + 12


def _split_sides(labels, cursor_x):
    right = [lbl for lbl in labels if lbl.scenePos().x() > cursor_x]
    left = [lbl for lbl in labels if lbl.scenePos().x() < cursor_x]
    return right, left


def _leader_for(pw, lbl):
    idx = pw._cursor_item_pool["labels"].index(lbl)
    return pw._cursor_item_pool["leaders"][idx]


def _leader_scene_pts(pw, lbl):
    leader = _leader_for(pw, lbl)
    xs, ys = leader.getData()
    return leader, [
        pw.view_box.mapViewToScene(QPointF(x, y)) for x, y in zip(xs, ys)
    ]


# ---------- 居中宽图：2 列一右一左 ----------

def test_centered_two_columns_split_right_and_left(
    plot_factory, qapp, monkeypatch
):
    pw = _setup(
        plot_factory, _label_df((0, 40, 80, 120, 160, 200)), size=(700, 420)
    )
    _pin_columns(pw, monkeypatch, columns=2, capacity=3)
    _reset_view(pw, 0, 99, -10, 220)
    qapp.processEvents()
    pw.vline.setVisible(True)
    pw.vline.setPos(50.0)
    _refresh(pw, times=4)

    mgr = pw._cursor_manager
    labels = _visible_labels(pw)
    assert len(labels) == 6
    cursor_x = _cursor_scene_x(pw, 50.0)
    right, left = _split_sides(labels, cursor_x)
    assert len(right) == 3 and len(left) == 3, (
        f"交替应形成 3 右 + 3 左，实际 {len(right)} 右 + {len(left)} 左"
    )
    # 分配顺序：高 y 先入右内列
    assert {lbl.toPlainText() for lbl in right} == {"200", "160", "120"}
    assert {lbl.toPlainText() for lbl in left} == {"80", "40", "0"}

    right_x = right[0].scenePos().x()
    left_x = left[0].scenePos().x()
    assert all(abs(lbl.scenePos().x() - right_x) < 0.5 for lbl in right)
    assert all(abs(lbl.scenePos().x() - left_x) < 0.5 for lbl in left)
    assert abs(right_x - (cursor_x + mgr.LABEL_GAP_TO_CURSOR)) < 0.5, (
        "右列左缘应贴光标+GAP"
    )
    w_left = max(_text_width(pw, lbl) for lbl in left)
    assert abs(
        (left_x + w_left) - (cursor_x - mgr.LABEL_GAP_TO_CURSOR)
    ) < 0.5, "左列右缘应贴光标−GAP"


# ---------- 3 列：右/左/右，外侧列 elbow ----------

def test_three_columns_alternate_and_outer_column_elbows(
    plot_factory, qapp, monkeypatch
):
    pw = _setup(
        plot_factory, _label_df((0, 40, 80, 120, 160, 200)), size=(700, 420)
    )
    _pin_columns(pw, monkeypatch, columns=3, capacity=2)
    _reset_view(pw, 0, 99, -10, 220)
    qapp.processEvents()
    pw.vline.setVisible(True)
    pw.vline.setPos(50.0)
    _refresh(pw, times=4)

    mgr = pw._cursor_manager
    labels = _visible_labels(pw)
    assert len(labels) == 6
    cursor_x = _cursor_scene_x(pw, 50.0)
    right, left = _split_sides(labels, cursor_x)
    assert len(right) == 4 and len(left) == 2

    right_xs = sorted({round(lbl.scenePos().x(), 3) for lbl in right})
    left_xs = sorted({round(lbl.scenePos().x(), 3) for lbl in left})
    assert len(right_xs) == 2 and len(left_xs) == 1, (
        f"3 列应为 右2/左1，实际 right={right_xs} left={left_xs}"
    )
    x_inner, x_outer = right_xs
    assert abs(x_inner - (cursor_x + mgr.LABEL_GAP_TO_CURSOR)) < 0.5

    # 按 y 序（上→下）= 分配序：右内 2 → 左内 2 → 右外 2
    ordered = sorted(labels, key=lambda lbl: lbl.scenePos().y())
    assert all(abs(lbl.scenePos().x() - x_inner) < 0.5 for lbl in ordered[:2])
    assert all(abs(lbl.scenePos().x() - left_xs[0]) < 0.5 for lbl in ordered[2:4])
    assert all(abs(lbl.scenePos().x() - x_outer) < 0.5 for lbl in ordered[4:])

    # 两侧最内列直线（2 点），外侧列 elbow（4 点、竖段在左缘外 6px）
    for lbl in ordered[:4]:
        leader, pts = _leader_scene_pts(pw, lbl)
        assert leader.isVisible()
        assert len(pts) == 2, f"{lbl.toPlainText()} 属于最内列，应为直线"
    for lbl in ordered[4:]:
        leader, pts = _leader_scene_pts(pw, lbl)
        assert leader.isVisible()
        assert len(pts) == 4, f"{lbl.toPlainText()} 属于外侧列，应为 elbow"
        assert abs(pts[1].x() - pts[2].x()) < 1e-6
        assert abs(pts[1].x() - (lbl.scenePos().x() - 6)) < 1e-6
        assert abs(pts[-1].x() - lbl.scenePos().x()) < 1e-6


# ---------- 光标贴边：退化单侧 ----------

@pytest.mark.parametrize("cursor_x, expect", [(2.0, "right"), (98.0, "left")])
def test_cursor_near_edge_degenerates_to_single_side(
    plot_factory, qapp, monkeypatch, cursor_x, expect
):
    pw = _setup(
        plot_factory, _label_df((0, 40, 80, 120, 160, 200)), size=(700, 420)
    )
    _pin_columns(pw, monkeypatch, columns=2, capacity=3)
    # 200px 以下可用宽一律视为无预算 → 与字体宽度无关
    _pin_side_budget(monkeypatch, min_available=200.0, cols=5)
    _reset_view(pw, 0, 99, -10, 220)
    qapp.processEvents()
    pw.vline.setVisible(True)
    pw.vline.setPos(cursor_x)
    _refresh(pw, times=4)

    labels = _visible_labels(pw)
    assert len(labels) == 6
    c = _cursor_scene_x(pw, cursor_x)
    if expect == "right":
        assert all(lbl.scenePos().x() > c for lbl in labels), (
            f"贴左缘应退化为全右侧：{[lbl.scenePos().x() for lbl in labels]}"
        )
    else:
        assert all(lbl.scenePos().x() < c for lbl in labels), (
            f"贴右缘应退化为全左侧：{[lbl.scenePos().x() for lbl in labels]}"
        )
    # 两列仍在（退化只看侧，不减列）
    assert len({round(lbl.scenePos().x(), 3) for lbl in labels}) == 2


# ---------- 双游标：交替未启用，各组单侧 ----------

def test_dual_cursor_groups_keep_single_side(
    plot_factory, qapp, monkeypatch
):
    pw = _setup(
        plot_factory, _label_df((98, 99, 100, 101, 102)), size=(900, 420)
    )
    _pin_columns(pw, monkeypatch, columns=2, capacity=3)
    _pin_side_budget(monkeypatch, min_available=120.0, cols=3)
    _reset_view(pw, 0, 99, 80, 120)
    qapp.processEvents()
    pw.plot_context.cursor_mode = "2 anchored cursor"
    pw._cursor_manager.pinned_x_values = [25.0, 75.0]
    pw.vline.setVisible(True)
    pw.vline2.setVisible(True)
    _refresh(pw, times=4)

    labels = _visible_labels(pw)
    assert len(labels) == 10
    c_a = _cursor_scene_x(pw, 25.0)
    c_b = _cursor_scene_x(pw, 75.0)
    group_a, group_b = labels[:5], labels[5:]

    # 交替若误启用，组 A 会出现左列/向中间缝隙塞列
    assert all(lbl.scenePos().x() > c_a for lbl in group_a), (
        "双游标禁用交替：组 A 应全部在自身光标右侧"
    )
    assert all(lbl.scenePos().x() > c_b for lbl in group_b), (
        "双游标禁用交替：组 B 应全部在自身光标右侧"
    )
    assert len({round(lbl.scenePos().x(), 3) for lbl in group_a}) == 2
    assert len({round(lbl.scenePos().x(), 3) for lbl in group_b}) == 2


# ---------- 两侧最内列均为两点直线 ----------

def test_both_inner_columns_use_straight_leaders(
    plot_factory, qapp, monkeypatch
):
    pw = _setup(
        plot_factory, _label_df((0, 40, 80, 120, 160, 200)), size=(700, 420)
    )
    _pin_columns(pw, monkeypatch, columns=2, capacity=3)
    _reset_view(pw, 0, 99, -10, 220)
    qapp.processEvents()
    pw.vline.setVisible(True)
    pw.vline.setPos(50.0)
    _refresh(pw, times=4)

    labels = _visible_labels(pw)
    assert len(labels) == 6
    cursor_x = _cursor_scene_x(pw, 50.0)
    right, left = _split_sides(labels, cursor_x)
    assert len(right) == 3 and len(left) == 3

    for lbl in right:
        leader, pts = _leader_scene_pts(pw, lbl)
        assert leader.isVisible()
        assert len(pts) == 2, "右内列应为两点直线"
        assert abs(pts[-1].x() - lbl.scenePos().x()) < 1e-6
        assert abs(pts[-1].y() - lbl.scenePos().y()) < 1e-6
    for lbl in left:
        leader, pts = _leader_scene_pts(pw, lbl)
        assert leader.isVisible()
        assert len(pts) == 2, "左内列应为两点直线"
        right_edge = lbl.scenePos().x() + _text_width(pw, lbl)
        assert abs(pts[-1].x() - right_edge) < 1e-6, "左列终点应为标签右缘"