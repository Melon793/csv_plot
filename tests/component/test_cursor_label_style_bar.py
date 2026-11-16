"""游标数值标签定稿样式用例（offscreen）：90α 白底 + 左侧全色条。

对应 feature/cursor-label-bar 的样式已合入 main；同色底变体（tint*）已弃用。
"""

from __future__ import annotations

import pytest

from PySide6.QtCore import QPointF, Qt

from src.ui.widgets.cursor_manager import LabelLayout
from tests.component.test_cursor_label_edge import (
    _make_peak_df,
    _refresh,
    _setup,
    _visible_labels,
    _x_where,
)

# 采样容差：渐变边缘 + 抗锯齿的舍入误差
_TOL = 6


def test_pinned_style_switches_with_same_color(plot_factory, qapp):
    """颜色不变时 pinned 与普通标签也必须切换虚线/透明度（缓存键含 pinned）"""
    pw = _setup(plot_factory, _make_peak_df(), size=(500, 420))
    vb = pw.view_box
    vb.setXRange(0, 99, padding=0.05)
    vb.setYRange(0, 200, padding=0)
    qapp.processEvents()
    peak_x = _x_where(pw, "y", 100.0)
    cross = vb.mapViewToScene(QPointF(peak_x, 100.0))
    mgr = pw._cursor_manager
    mgr._clear_cursor_items()
    color = pw.curves["y"].color

    mgr._render_label_layout([LabelLayout(
        layout_x=cross.x(), layout_y=cross.y() - 60, index=0,
        x_pos=peak_x, y_pos=250.0, y_value="250", color=color,
        pinned=True, up=True,
    )])
    lbl = pw._cursor_item_pool["labels"][0]
    assert lbl.toPlainText() == "▲ 250"
    assert lbl.border.style() == Qt.PenStyle.DashLine
    assert lbl.opacity() == pytest.approx(0.8)

    # 同一图元、同一颜色：退出 pinned 后必须恢复实线与不透明
    mgr._render_label_layout([LabelLayout(
        layout_x=cross.x(), layout_y=cross.y() - 10, index=0,
        x_pos=peak_x, y_pos=100.0, y_value="100", color=color,
    )])
    assert lbl.toPlainText() == "100"
    assert lbl.border.style() == Qt.PenStyle.SolidLine
    assert lbl.opacity() == pytest.approx(1.0)


def _show_label(pw):
    """把交点放到视图中部并刷新出数值标签

    关网格：底色 90α 半透明，像素采样点需落在纯白背景上才能算期望色。
    """
    pw.plot_item.showGrid(x=False, y=False)
    vb = pw.view_box
    vb.setXRange(0, 99, padding=0.05)
    vb.setYRange(0, 200, padding=0)
    pw.vline.setVisible(True)
    pw.vline.setPos(_x_where(pw, "y", 100.0))
    _refresh(pw)
    labels = _visible_labels(pw)
    assert labels, "前置：数值标签应可见"
    return labels[0]


def _grab_and_origin(pw, qapp, lbl):
    """抓取绘图控件位图，返回 (image, 标签左缘/中线的图像坐标)。

    注意：不能用 QGraphicsScene.render —— TextItem 的 paint 只在场景切换时
    重设 painter 变换，离屏 scene.render 会画错位置（实测填充与描边都不出现）。
    """
    qapp.processEvents()
    img = pw.grab().toImage()
    dpr = img.devicePixelRatio()
    pt = pw.mapFromScene(lbl.scenePos())
    return img, int(round(pt.x() * dpr)), int(round(pt.y() * dpr))


def _near(actual, expected_hex: str, tol: int = _TOL) -> bool:
    exp = [int(expected_hex[i:i + 2], 16) for i in (1, 3, 5)]
    got = [actual.red(), actual.green(), actual.blue()]
    return all(abs(a - b) <= tol for a, b in zip(got, exp))


def test_label_bar_style_white_fill_and_color_bar(plot_factory, qapp):
    """白底不透明 + 左侧 4px 曲线色条（渐变刷，不新增图元）"""
    pw = _setup(plot_factory, _make_peak_df(), size=(500, 420))
    lbl = _show_label(pw)
    curve_color = pw.curves["y"].color.lower()

    assert lbl.fill.style() == Qt.BrushStyle.LinearGradientPattern
    stops = lbl.fill.gradient().stops()
    assert len(stops) >= 3
    # 0~4px 是不透明全色条；4px 之后是 90α 白底（透出曲线）
    assert stops[0][1].name().lower() == curve_color
    assert stops[0][1].alpha() == 255
    assert stops[1][1].name().lower() == curve_color
    assert stops[-1][1].name().lower() == "#ffffff"
    assert stops[-1][1].alpha() == 90

    # 文字始终黑色；边框为曲线色 1.5px（main 原色，浅灰对白底几乎不可见）
    assert lbl.textItem.defaultTextColor().name().lower() == "#000000"
    assert lbl.border.color().name().lower() == curve_color
    assert lbl.border.widthF() == 1.5

    # 色条与文字之间留 3px 间隙（默认左边距 4 + 3 = 7）
    assert lbl.textItem.document().rootFrame().frameFormat().leftMargin() == 7


def test_label_bar_rendered_in_curve_color(plot_factory, qapp):
    """像素验证：标签左缘内侧 2px 是曲线色（色条），同一行右侧是纯白"""
    pw = _setup(plot_factory, _make_peak_df(), size=(500, 420))
    lbl = _show_label(pw)
    curve_color = pw.curves["y"].color
    img, x, y = _grab_and_origin(pw, qapp, lbl)

    # 取文字上方 6px 的一行（仍在标签内、避开字形），色条与白底同排可比
    row = y - 6
    assert _near(img.pixelColor(x + 2, row), curve_color), (
        f"色条像素 {img.pixelColor(x + 2, row).name()} 应为 {curve_color}"
    )
    assert _near(img.pixelColor(x + 20, row), "#ffffff"), (
        f"白底像素 {img.pixelColor(x + 20, row).name()} 应为 #ffffff"
    )


def test_label_height_cache_not_underestimated(plot_factory, qapp):
    """高度缓存必须 >= 实测高度：否则列内 4px 间距会缩水（旧 +6 公式的缺陷）"""
    pw = _setup(plot_factory, _make_peak_df(), size=(500, 420))
    lbl = _show_label(pw)
    assert pw._cursor_manager._cached_label_height_pixels >= lbl.boundingRect().height()


def test_leader_line_connects_point_to_label(plot_factory, qapp):
    """最内列引导线：两点直线，端点落交点与标签左缘"""
    pw = _setup(plot_factory, _make_peak_df(), size=(500, 420))
    lbl = _show_label(pw)
    leaders = [ld for ld in pw._cursor_item_pool["leaders"] if ld.isVisible()]
    assert leaders, "标签离交点 >8px 时应有引导线"
    xs, ys = leaders[0].getData()
    assert len(xs) == 2, "最内列应为两点直线（第二列才用 elbow）"

    vb = pw.view_box
    lab = vb.mapSceneToView(lbl.scenePos())  # 标签左缘中点（数据坐标）
    peak_x = _x_where(pw, "y", 100.0)
    ends = list(zip(xs, ys))
    assert abs(ends[0][0] - peak_x) < 1e-6 and abs(ends[0][1] - 100.0) < 1e-6, (
        "直线起点应落在交点"
    )
    assert abs(ends[-1][0] - lab.x()) < 1e-6 and abs(ends[-1][1] - lab.y()) < 1e-6, (
        "直线终点应落在标签左缘"
    )


def test_leader_pool_cleared_on_full_cleanup(plot_factory, qapp):
    """彻底清理（hide_only=False）后 leaders 池清空，不泄漏图元"""
    pw = _setup(plot_factory, _make_peak_df(), size=(500, 420))
    _show_label(pw)
    pw._cursor_manager._clear_cursor_items(hide_only=False)
    assert pw._cursor_item_pool["leaders"] == [], "彻底清理后 leaders 池应清空"
