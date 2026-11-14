"""游标数值标签 A1'（白底 + 左侧全色条）样式用例（offscreen）。

分支对比用：本文件对应 feature/cursor-label-bar；
另一支 feature/cursor-label-tint90 是 90α 高透明底色，两者样式互斥。
"""

from __future__ import annotations

from PySide6.QtCore import Qt

from tests.component.test_cursor_label_edge import (
    _make_peak_df,
    _refresh,
    _setup,
    _visible_labels,
    _x_where,
)

# 采样容差：渐变边缘 + 抗锯齿的舍入误差
_TOL = 6


def _show_label(pw):
    """把交点放到视图中部并刷新出数值标签"""
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
    # 0~4px 是曲线色，4px 之后是白色；两端都必须完全不透明
    assert stops[0][1].name().lower() == curve_color
    assert stops[0][1].alpha() == 255
    assert stops[1][1].name().lower() == curve_color
    assert stops[-1][1].name().lower() == "#ffffff"
    assert stops[-1][1].alpha() == 255

    # 文字始终黑色、边框浅灰（不随曲线色变化）
    assert lbl.textItem.defaultTextColor().name().lower() == "#000000"
    assert lbl.border.color().name().lower() == "#d0d4da"


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
