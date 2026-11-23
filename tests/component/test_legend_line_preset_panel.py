"""图例快捷面板「样式预设」组组件测试（offscreen）。

面板第三组分段控件（默认/粗实/虚线/点线）：单选语义（点选即关菜单、
picked 仅一键）、单条应用播报、批量应用一次播报。核心 pen/Z 语义见
test_curve_line_preset.py。
"""

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QLabel, QPushButton

from tests.fixtures.waits import settle

from src.core.data_types import LINE_PRESET_DASH
from src.ui.widgets.curve_style_panel import _SegmentedControl


def _y(n: int) -> np.ndarray:
    return np.sin(np.linspace(0.0, 6.0, n))


@pytest.fixture()
def preset_plot(plot_factory, qapp):
    n = 30
    pw = plot_factory(pd.DataFrame({"v": _y(n), "w": _y(n)}))
    pw.resize(800, 600)
    pw.window().show()
    pw.show()
    assert pw.add_variable_to_plot("v", x_values=np.arange(n, dtype=float), y_values=_y(n))
    assert pw.add_variable_to_plot("w", x_values=np.arange(n, dtype=float), y_values=_y(n))
    settle()
    return pw


class _Closer:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def _preset_segment(panel):
    for seg in panel.findChildren(_SegmentedControl):
        buttons = seg.findChildren(QPushButton)
        if buttons and buttons[0].toolTip() == "默认（自动粗细，实线）":
            return seg, buttons
    raise AssertionError("面板未找到样式预设分段控件")


def test_panel_preset_group_pick_and_apply(preset_plot):
    pw = preset_plot
    msgs = []
    pw.window()._show_status_message = lambda msg: msgs.append(msg)
    picked = {}
    closer = _Closer()
    panel = pw.legend_label._build_style_panel_widget("v", picked, [closer])
    seg, buttons = _preset_segment(panel)
    assert [b.text() for b in buttons] == ["默认", "粗实", "虚线", "点线"]
    assert buttons[0].isChecked()  # 当前 auto

    dash_btn = next(b for b in buttons if b.text() == "虚线")
    QTest.mouseClick(dash_btn, Qt.MouseButton.LeftButton)
    assert picked == {"line_preset": LINE_PRESET_DASH}  # 单选语义
    assert closer.closed

    pw.legend_label._apply_curve_panel_picks("v", picked)
    assert pw.curves["v"].line_preset == LINE_PRESET_DASH
    assert msgs == ["已切换 v 样式：虚线"]
    panel.deleteLater()


def test_batch_preset_apply_and_announce(preset_plot):
    pw = preset_plot
    msgs = []
    pw.window()._show_status_message = lambda msg: msgs.append(msg)
    pw.legend_label._apply_batch_style_picks(
        ["v", "w"], {"line_preset": LINE_PRESET_DASH}
    )
    assert pw.curves["v"].line_preset == LINE_PRESET_DASH
    assert pw.curves["w"].line_preset == LINE_PRESET_DASH
    assert msgs == ["已修改 2 个变量的样式：虚线"]


def test_style_caption_has_inline_perf_note(preset_plot):
    """「样式」标题行右侧常驻性能小注：同行、更小字号、行宽不超面板。"""
    pw = preset_plot
    panel = pw.legend_label._build_style_panel_widget("v", {}, [])
    labels = panel.findChildren(QLabel)
    note = next((lb for lb in labels if lb.text() == "(大数据量建议实线)"), None)
    cap = next((lb for lb in labels if lb.text() == "样式"), None)
    assert note is not None and cap is not None
    panel.layout().activate()
    assert note.y() == cap.y()  # 同一行，零新增高度
    if note.font().pointSizeF() > 0 and cap.font().pointSizeF() > 0:
        assert note.font().pointSizeF() < cap.font().pointSizeF()
    row = note.parentWidget()
    assert row.sizeHint().width() <= panel.sizeHint().width()  # 不换行
    panel.deleteLater()


def test_caption_shows_current_value(preset_plot):
    """当前档文字提醒（2026-11-23 拍板）：分组标题后直接给出当前值。

    值独立成 QLabel（深色正文），标题文本保持原样，既有「样式/小注同行」
    断言不受影响；按钮 ACCENT 图标保留作次级线索。
    """
    pw = preset_plot
    panel = pw.legend_label._build_style_panel_widget("v", {}, [])
    texts = [lb.text() for lb in panel.findChildren(QLabel)]
    assert "：直线" in texts  # 绘制方式当前 auto
    assert "：自动" in texts  # 数据点当前 auto
    assert "：默认" in texts  # 样式当前 auto
    panel.deleteLater()