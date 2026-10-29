"""游标数值标签边缘显示 component 测试（offscreen）。

对应 tmp/2026-10-29/y_padding_and_cursor_label_dev_doc.md 优化二：
多行小窗口（矮图）下交点靠近子图上下边缘时，Y 数值标签必须完整
显示（Phase 1 过滤改吸附）；列布局不得把标签画出视口；极矮图触发
极限保护统一隐藏。

设计原则：交点在视图内时标签必须可见；宁可在视口内重叠，也不画出
视口消失。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from PySide6.QtCore import QPointF
from PySide6.QtGui import QFontMetrics


# 矮图窗口尺寸：viewBox 高约 80~100px（对应 4 行小窗口的单个子图）
SHORT_SIZE = (500, 160)


def _make_peak_df() -> pd.DataFrame:
    """单点峰数据：x=50 处 y=100，其余 0（交点位置精确可控）"""
    y = np.zeros(100)
    y[50] = 100.0
    return pd.DataFrame({"y": y})


def _setup(plot_factory, df, size=SHORT_SIZE):
    pw = plot_factory(df)
    pw.resize(*size)
    pw.window().show()
    pw.show()
    for name in df.columns:
        assert pw.plot_variable(name)
    # pw 默认「仅显示 x 位置」；本文件全部用例针对数值标签
    pw.show_values_only = False
    return pw


def _x_where(pw, var, value) -> float:
    """反查曲线中 y==value 的数据点 x（不依赖 x_data 的 index 起点约定）"""
    ci = pw.curves[var]
    hits = np.where(ci.y_data == value)[0]
    assert len(hits), f"数据中不存在 y={value} 的点"
    return float(ci.x_data[int(hits[0])])


def _refresh(pw, times=1):
    """直接驱动标签更新链（绕过时间节流），必要时多次越过列数滞回

    程序化 setXRange/setYRange 会经 sigRangeChanged 进入交互防抖态
    （_is_interacting=True，UI_DEBOUNCE_DELAY_MS 后才复位），此处显式
    收尾，等价于交互已结束。
    """
    for _ in range(times):
        pw._is_interacting = False
        pw._last_cursor_update_time = 0.0
        pw._cursor_manager._update_multi_curve_cursor_label()


def _visible_labels(pw) -> list:
    return [l for l in pw._cursor_item_pool["labels"] if l.isVisible()]


# ---------- 用例 A：矮图 + 交点在视图上/下边缘 → 标签吸附显示 ----------

def test_label_visible_at_top_edge_short_plot(plot_factory, qapp):
    """交点 y 恰在上边缘（波峰）：标签必须可见且与视图相交

    修复前 Phase 1 的 y 死区过滤（半个标签高 ≈11px）把靠边交点的
    标签整体隐藏；修复后由 _layout_column 的边界 clamp 吸附到边缘
    内侧完整显示。
    """
    pw = _setup(plot_factory, _make_peak_df())
    vb = pw.view_box
    vb.setXRange(0, 99, padding=0.05)
    vb.setYRange(0, 100, padding=0)  # 波峰 y=100 恰在上边缘
    qapp.processEvents()
    pw.vline.setVisible(True)
    pw.vline.setPos(_x_where(pw, "y", 100.0))
    _refresh(pw)

    labels = _visible_labels(pw)
    assert len(labels) == 1, "交点在上边缘时标签必须显示（吸附而非隐藏）"
    view_rect = vb.sceneBoundingRect()
    assert labels[0].sceneBoundingRect().intersects(view_rect), (
        "标签必须与视图相交（在视口内）"
    )


def test_label_visible_at_bottom_edge_short_plot(plot_factory, qapp):
    """交点 y 恰在下边缘（波谷 y=0）：标签必须可见且与视图相交"""
    pw = _setup(plot_factory, _make_peak_df())
    vb = pw.view_box
    vb.setXRange(0, 99, padding=0.05)
    vb.setYRange(0, 100, padding=0)  # 波谷 y=0 恰在下边缘
    qapp.processEvents()
    pw.vline.setVisible(True)
    pw.vline.setPos(_x_where(pw, "y", 0.0))
    _refresh(pw)

    labels = _visible_labels(pw)
    assert len(labels) == 1, "交点在下边缘时标签必须显示（吸附而非隐藏）"
    view_rect = vb.sceneBoundingRect()
    assert labels[0].sceneBoundingRect().intersects(view_rect)


# ---------- 用例 B：矮图 + 多曲线 → 标签列不得画出视口 ----------

def test_all_labels_within_viewport_multi_curve_short_plot(plot_factory, qapp):
    """矮图窄窗 + 5 条曲线：所有标签的列起点必须在视口内

    修复前列数超出两侧容量时溢出列被"压缩到右侧"（col_idx 从已用列
    起算），第 2/3 列起点画出视口外被裁剪（消失）；修复后列数按可用
    宽度截断，超容量标签由列内防重叠 Pass 压缩消化（宁重叠不出界）。
    """
    df = pd.DataFrame({f"y{i}": np.full(100, float(i * 25)) for i in range(5)})
    pw = _setup(plot_factory, df, size=(400, SHORT_SIZE[1]))
    vb = pw.view_box
    vb.setXRange(0, 99, padding=0.05)
    vb.setYRange(0, 100, padding=0)
    qapp.processEvents()
    pw.vline.setVisible(True)
    pw.vline.setPos(50.0)  # 视图中央 → 两侧各容不下一个标准列宽
    _refresh(pw, times=4)  # 越过列数切换滞回（HYSTERESIS_THRESHOLD=3）

    labels = _visible_labels(pw)
    # 5 条曲线 5 个交点全部可见（含 y=0/100 两个边缘交点）
    assert len(labels) == 5, f"应有 5 个可见标签，实际 {len(labels)}"
    view_rect = vb.sceneBoundingRect()
    for label in labels:
        x = label.scenePos().x()
        assert view_rect.left() <= x <= view_rect.right(), (
            f"标签列起点 x={x} 出视口 [{view_rect.left()}, {view_rect.right()}]"
        )


# ---------- 用例 C：极矮图 → 极限保护统一隐藏，不崩溃 ----------

def test_labels_hidden_on_tiny_plot_without_crash(plot_factory, qapp):
    """图高不足以放置一行标签：全部隐藏且不崩溃

    阈值按运行时 QFontMetrics 推导（label_h+间距+2×margin），不硬编码
    像素；触发条件即修复前被 max(1, ...) 掩蔽的死代码分支。
    """
    pw = _setup(plot_factory, _make_peak_df())
    vb = pw.view_box
    vb.setXRange(0, 99, padding=0.05)
    vb.setYRange(0, 200, padding=0)  # 交点 y=100 在视图中部（避开边缘，纯测极限保护）
    qapp.processEvents()
    pw.vline.setVisible(True)
    pw.vline.setPos(_x_where(pw, "y", 100.0))
    _refresh(pw)  # 正常尺寸下先渲染一次，建立 pool 与字体缓存
    labels = pw._cursor_item_pool["labels"]
    assert labels, "前置：正常尺寸下标签应已建立"

    fm = QFontMetrics(labels[0].textItem.font())  # TextItem 内部 QGraphicsTextItem
    label_h = fm.height() + 6  # 与 _position_labels_new 的缓存公式一致
    needed_h = (
        label_h
        + pw._cursor_manager.LABEL_VERTICAL_GAP
        + 2 * 0.1 * label_h  # margin_pixels_scene = label_h * 0.1
    )
    for h in range(150, 30, -5):
        pw.resize(500, h)
        qapp.processEvents()
        if vb.height() < needed_h:
            break
    assert vb.height() < needed_h, "未能构造出极矮图（offscreen 几何异常）"

    _refresh(pw)  # 不得崩溃
    assert not _visible_labels(pw), "极矮图下标签应全部隐藏"


# ---------- 用例 D（回归）：正常高度图 + 交点居中 → 位置与现行为一致 ----------

def test_label_position_unchanged_for_centered_cross(plot_factory, qapp):
    """正常高度图、交点在视图中部：单标签维持交点上方 10px 的既有行为"""
    pw = _setup(plot_factory, _make_peak_df(), size=(500, 420))
    vb = pw.view_box
    vb.setXRange(0, 99, padding=0.05)
    vb.setYRange(0, 200, padding=0)  # 波峰 y=100 恰在视图垂直中部
    qapp.processEvents()
    pw.vline.setVisible(True)
    pw.vline.setPos(_x_where(pw, "y", 100.0))
    _refresh(pw)

    labels = _visible_labels(pw)
    assert len(labels) == 1
    peak_x = _x_where(pw, "y", 100.0)
    cross_scene_y = vb.mapViewToScene(QPointF(peak_x, 100.0)).y()
    # TextItem anchor=(0, 0.5)：pos 即标签垂直中心；现行为 = 交点上偏 10px
    center_y = labels[0].scenePos().y()
    assert center_y == pytest.approx(cross_scene_y - 10.0, abs=1.5), (
        f"标签中心 y={center_y}，预期交点上方 10px（{cross_scene_y - 10.0}）"
    )


# ---------- 用例 E（双游标）：一游标交点靠边、另一居中 → 各自可见 ----------

def test_two_cursor_labels_both_visible_when_one_at_edge(plot_factory, qapp):
    """2 anchored cursor：靠边交点的标签不再挤掉另一游标的标签（pool 无错位）"""
    y = np.zeros(100)
    y[50] = 100.0  # 游标 A：波峰，恰在上边缘
    y[30] = 50.0   # 游标 B：视图垂直中部
    pw = _setup(plot_factory, pd.DataFrame({"y": y}))
    vb = pw.view_box
    vb.setXRange(0, 99, padding=0.05)
    vb.setYRange(0, 100, padding=0)
    qapp.processEvents()
    pw.plot_context.cursor_mode = "2 anchored cursor"
    pw._cursor_manager.pinned_x_values = [
        _x_where(pw, "y", 100.0),  # 靠边游标（上边缘）
        _x_where(pw, "y", 50.0),   # 居中游标
    ]
    pw.vline.setVisible(True)
    pw.vline2.setVisible(True)
    _refresh(pw)

    pool = pw._cursor_item_pool["labels"]
    assert len(pool) >= 2, f"应有 2 个标签入池，实际 {len(pool)}"
    assert pool[0].isVisible(), "靠边交点（上边缘）的标签必须可见"
    assert pool[1].isVisible(), "居中交点的标签必须可见"
    assert pool[0].toPlainText() == "100", "pool[0] 应对应靠边游标的交点值"
    assert pool[1].toPlainText() == "50", "pool[1] 应对应居中游标的交点值"
    view_rect = vb.sceneBoundingRect()
    for lbl in (pool[0], pool[1]):
        assert lbl.sceneBoundingRect().intersects(view_rect)


# ---------- 用例 F（回归）：交点 x 出视图 → 标签仍隐藏 ----------

def test_label_hidden_when_cross_x_outside_view(plot_factory, qapp):
    """anchored 游标交点 x 在视图外：标签隐藏（保留的 x 方向过滤）"""
    pw = _setup(plot_factory, _make_peak_df())
    vb = pw.view_box
    vb.setXRange(20, 80, padding=0)
    vb.setYRange(0, 200, padding=0)  # 交点 y=100 在视图中部（避开边缘，纯测 x 过滤）
    qapp.processEvents()
    pw.plot_context.cursor_mode = "1 anchored cursor"
    pw.vline.setVisible(True)

    # 先在视图内建立 pool（标签可见），再移出视图验证隐藏路径
    pw._cursor_manager.pinned_x_value = _x_where(pw, "y", 100.0)
    _refresh(pw)
    assert _visible_labels(pw), "前置：视图内交点标签应已显示"

    # 数据内但视图外（X 视窗 [20, 80]）：x_data 域约 1..100，取视图右界外
    ci = pw.curves["y"]
    x_out = float(ci.x_data[int(len(ci.x_data) * 0.95)])  # ≈ x=95
    assert x_out > 80, "构造的出界 x 应在视图右界外"
    pw._cursor_manager.pinned_x_value = x_out
    _refresh(pw)
    assert not _visible_labels(pw), "交点 x 出视图后标签应隐藏"


# ---------- 辅助方法单元：clamp 倒挂防御 ----------

def test_clamp_label_y_inverted_interval_falls_back_to_center():
    """极矮图 clamp 区间倒挂（下界 > 上界）：回退列垂直中心，不出界"""
    from src.ui.widgets.cursor_manager import CursorManager

    # 倒挂：lo = 0+8 > hi = 10-8 → 返回中心 (0+10)/2
    assert CursorManager._clamp_label_y(50.0, 0.0, 10.0, 8.0) == pytest.approx(5.0)
    # 正常区间：夹到边界内
    assert CursorManager._clamp_label_y(-5.0, 0.0, 100.0, 10.0) == pytest.approx(10.0)
    assert CursorManager._clamp_label_y(50.0, 0.0, 100.0, 10.0) == pytest.approx(50.0)
    assert CursorManager._clamp_label_y(200.0, 0.0, 100.0, 10.0) == pytest.approx(90.0)
