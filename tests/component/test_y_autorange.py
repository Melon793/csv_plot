"""Y 轴范围行为 component 测试（offscreen）。

当前分支现行语义（经实测诊断确认，见 tmp/ 诊断记录）：
- 首绘后 Y 轴范围被显式设为全量数据范围（含窗口外极值，带 padding），
  X/Y autoRange 均关闭（生产中等价于 layout_manager 建链时的
  enableAutoRange(x=False)）
- 缩放 X 轴不触发 Y 重算（Y 保持稳定，历史 Y 抖动缺陷的防回归断言）
- Ctrl+Y「仅调节 Y 轴」（auto_y_in_x_range）启用 Y autoRange，
  配合 autoVisibleOnly 在当前 X 范围内重算 Y（可见段跟随）

对应 help.md 4️⃣「Ctrl+Y 仅调节Y轴」与缩放交互的 Y 稳定性。
"""

import numpy as np
import pandas as pd
import pytest


N = 10000


def _make_extreme_df() -> pd.DataFrame:
    """主体 40~60 的正弦数据，在可见窗口外注入极值
    （index 100 → 500，index 9500 → -300）。"""
    y = np.sin(np.linspace(0, 100, N) / 5.0) * 10 + 50
    y[100] = 500.0
    y[9500] = -300.0
    return pd.DataFrame({"y": y})


@pytest.fixture()
def plotted(plot_factory, qapp):
    """绘制带窗口外极值的曲线，并施加生产对齐的 autoRange 初始化

    注意：宿主窗口必须 show（有效视图尺寸非零），否则 pyqtgraph
    auto-range 无法基于可见窗口重算（实测诊断确认）。
    """
    pw = plot_factory(_make_extreme_df())
    pw.resize(800, 600)
    pw.window().show()
    pw.show()
    assert pw.plot_variable("y")
    vb = pw.view_box
    # layout_manager 建立 XLink 时对每个 viewbox 的等价初始化
    vb.enableAutoRange(x=False)
    qapp.processEvents()
    return pw, vb


def test_first_render_y_covers_full_data(plotted):
    """首绘后：Y 范围覆盖全量数据（含极值，带 padding），X/Y autoRange 关闭"""
    _, vb = plotted
    y_min, y_max = vb.viewRange()[1]
    assert y_max >= 500, f"Y 上界应覆盖极值 500: {y_max}"
    assert y_min <= -300, f"Y 下界应覆盖极值 -300: {y_min}"
    assert not vb.state["autoRange"][0]
    assert not vb.state["autoRange"][1]
    # Y autoVisibleOnly 开启：后续启用 Y autoRange 时仅统计可见段
    assert vb.state["autoVisibleOnly"][1]


def test_zoom_x_keeps_y_stable(plotted, qapp):
    """缩放 X 轴：Y 范围保持不变（不重算、不抖动）"""
    _, vb = plotted
    y_before = list(vb.viewRange()[1])

    vb.setXRange(4000, 6000, padding=0)
    qapp.processEvents()
    qapp.processEvents()

    y_after = list(vb.viewRange()[1])
    assert y_after[0] == pytest.approx(y_before[0], abs=1e-6)
    assert y_after[1] == pytest.approx(y_before[1], abs=1e-6)


def test_consecutive_zooms_keep_y_stable(plotted, qapp):
    """连续多次缩放：Y 全程稳定，不被窗口外极值扰动"""
    _, vb = plotted
    y_before = list(vb.viewRange()[1])

    for left, right in ((4000, 6000), (4500, 5500), (2000, 3000), (7000, 8000)):
        vb.setXRange(left, right, padding=0)
        qapp.processEvents()
        y_min, y_max = vb.viewRange()[1]
        assert y_min == pytest.approx(y_before[0], abs=1e-6), f"窗口 [{left},{right}]"
        assert y_max == pytest.approx(y_before[1], abs=1e-6), f"窗口 [{left},{right}]"


def test_auto_y_fits_visible_segment(plotted, qapp):
    """Ctrl+Y：缩放后调用 auto_y_in_x_range，Y 重算为可见段范围（约 40~60）"""
    pw, vb = plotted
    vb.setXRange(4000, 6000, padding=0)
    qapp.processEvents()

    pw.auto_y_in_x_range()
    qapp.processEvents()
    qapp.processEvents()

    assert bool(vb.state["autoRange"][1]) is True
    y_min, y_max = vb.viewRange()[1]
    # 可见段主体 40~60（允许 pyqtgraph 默认 padding 余量）
    assert 30 < y_min < 45, f"Y 下界异常: {y_min}"
    assert 55 < y_max < 70, f"Y 上界异常: {y_max}"
    # 未被窗口外极值污染
    assert y_min > -100 and y_max < 200


def test_auto_y_covers_extremes_when_zoomed_out(plotted, qapp):
    """Ctrl+Y 在全量 X 范围：Y 重新覆盖窗口外极值"""
    pw, vb = plotted
    vb.setXRange(1, N, padding=0)
    qapp.processEvents()

    pw.auto_y_in_x_range()
    qapp.processEvents()
    qapp.processEvents()

    y_min, y_max = vb.viewRange()[1]
    assert y_max >= 500, f"Y 上界应覆盖极值 500: {y_max}"
    assert y_min <= -300, f"Y 下界应覆盖极值 -300: {y_min}"


def test_auto_y_padding_matches_auto_range(plotted, qapp):
    """Ctrl+Y 与「自动缩放」的 Y padding 必须一致（均 ≈ 0.1）

    defaultPadding 统一前：Ctrl+Y 走 pyqtgraph 动态 suggestPadding
    （clip(1/√高度px, 0.02, 0.1)，随子图像素高度浮动；offscreen
    height=0 时退化为 0.02），「自动缩放」固定 setYRange(padding=0.1)，
    同一数据来回切换时 Y viewRange 非预期跳变。

    断言用相对 padding (视图span/数据span-1)/2 而非绝对范围：两条路径
    边界计算内核不同（childrenBounds vs np 极值）。注意 autoRange 会把
    线宽像素当量（pixelPadding=线宽×0.7072，折算≈数据span的0.3%）
    计入边界，Ctrl+Y 的相对 padding 恒为 0.1+像素当量，与「自动缩放」
    的精确 0.1 存在固有微小差，故一致性断言容差取 0.005（统一前的
    动态 padding 偏差 ≈0.055，仍能可靠区分）。
    """
    pw, vb = plotted
    # 核心语义断言：suggestPadding 恒 0.1（与图高无关，含 offscreen
    # height=0 分支）—— defaultPadding=DEFAULT_PADDING_VAL_Y 的直接效果
    assert vb.suggestPadding(vb.YAxis) == pytest.approx(0.1, abs=1e-9)

    # 首绘 X 视窗（数据域+5% padding）覆盖全部数据点 →
    # Ctrl+Y 的 autoVisibleOnly 可见段 = 全量数据，与「自动缩放」同源
    y_data = pw.curves["y"].y_data
    data_span = float(np.nanmax(y_data) - np.nanmin(y_data))
    assert data_span > 0

    # 路径 1：Ctrl+Y（Y autoRange + autoVisibleOnly → suggestPadding）
    pw.auto_y_in_x_range()
    qapp.processEvents()
    qapp.processEvents()
    y_ctrl_min, y_ctrl_max = vb.viewRange()[1]
    pad_ctrl = ((y_ctrl_max - y_ctrl_min) / data_span - 1) / 2

    # 路径 2：「自动缩放」（np 极值 → setYRange(padding=DEFAULT_PADDING_VAL_Y)）
    assert pw.auto_range() is True
    qapp.processEvents()
    y_auto_min, y_auto_max = vb.viewRange()[1]
    pad_auto = ((y_auto_max - y_auto_min) / data_span - 1) / 2

    assert pad_auto == pytest.approx(0.1, abs=1e-6), (
        f"「自动缩放」的 Y padding 异常: {pad_auto}"
    )
    assert abs(pad_ctrl - pad_auto) < 0.005, (
        f"两条路径的 Y padding 不一致: Ctrl+Y={pad_ctrl}, 自动缩放={pad_auto}"
    )
