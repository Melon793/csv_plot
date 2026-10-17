"""TimestampAxisItem / IndexToTimeTextAdapter 的 component 测试

时间戳 X 轴 = 轴渲染层映射：曲线数据空间（索引）不动，底轴把刻度渲染成
时间戳竖排文字。本文件守住轴图元本体；模式切换的全链路在
``tests/e2e/test_x_axis_timestamp.py``。
"""

import numpy as np
import pandas as pd
import pytest
import pyqtgraph as pg

from src.core.time_axis_mapping import TimeAxisMapping, build_timestamps
from src.ui.widgets.time_axis_item import IndexToTimeTextAdapter, TimestampAxisItem

N_ROWS = 500


@pytest.fixture()
def mapping():
    series = pd.Series(
        pd.date_range("2026-10-03 11:51:53", periods=N_ROWS, freq="4s")
    )
    return TimeAxisMapping(build_timestamps(series), date_format="%H:%M:%S")


class TestTickStrings:
    def test_maps_display_x_via_mapper(self, qapp, mapping):
        axis = TimestampAxisItem(orientation="bottom", index_to_text=mapping.text_for_index)

        texts = axis.tickStrings([1.0, 250.0, 500.0], 1, 100)

        assert texts[0] == "11:51:53"
        # 行 250 = 11:51:53 + 249×4s = 12:08:29（烟囱实测锚定，防止 ±1 行漂移）
        assert texts[1] == "12:08:29"
        # 行 500 = 11:51:53 + 499×4s = 12:25:09
        assert texts[2] == "12:25:09"

    def test_out_of_range_tick_is_blank_not_clamped(self, qapp, mapping):
        axis = TimestampAxisItem(orientation="bottom", index_to_text=mapping.text_for_index)

        texts = axis.tickStrings([-5.0, 99999.0], 1, 100)

        assert texts == ["", ""]

    def test_none_mapper_falls_back_to_numeric(self, qapp):
        axis = TimestampAxisItem(orientation="bottom")

        texts = axis.tickStrings([100.0, 200.5], 1, 100)

        # 父类按 spacing 定精度（100 → 整数口径）；关键是不含时间冒号
        assert all(texts) and all(":" not in t for t in texts)


class TestRotatedGeometry:
    def test_axis_height_grows_before_first_paint(self, qapp, mapping):
        """高度预算必须在首绘前就位（绘制中改几何会整帧作废）"""
        axis = TimestampAxisItem(orientation="bottom", index_to_text=mapping.text_for_index)
        plain = pg.AxisItem(orientation="bottom")

        assert axis.height() > plain.height(), "竖排文字的宽度预算没撑起轴高"
        assert axis._rotated_text_width > 0

    def test_draw_picture_renders_without_error(self, qapp, mapping):
        from PySide6.QtGui import QColor, QPainter, QPixmap

        win = pg.GraphicsLayoutWidget(size=(600, 400))
        win.setBackground("w")
        plot = win.addPlot()
        plot.plot(x=np.arange(1, N_ROWS + 1), y=np.sin(np.arange(N_ROWS) / 30))
        axis = TimestampAxisItem(orientation="bottom", index_to_text=mapping.text_for_index)
        plot.setAxisItems({"bottom": axis})
        win.show()
        qapp.processEvents()

        pm = QPixmap(600, 400)
        pm.fill(QColor("white"))
        painter = QPainter(pm)
        win.render(painter)
        painter.end()

        # 墨迹落在轴区域（文字真的画出来了，而不是整块空白）
        band = pm.copy(0, pm.height() - axis.height() - 8, pm.width(), axis.height() + 8)
        assert band.toImage().bits() is not None
        colored = sum(
            1
            for y in range(0, band.height(), 4)
            for x in range(0, band.width(), 4)
            if band.toImage().pixel(x, y) != 0xFFFFFFFF
        )
        assert colored > 0, "轴区域没有任何墨迹，刻度文字没画出来"


class TestTickDensity:
    def test_density_is_double_of_default(self, qapp):
        """用户实测默认密度太稀疏：时间轴密度 = 默认 × 2"""
        axis = TimestampAxisItem(orientation="bottom")
        plain = pg.AxisItem(orientation="bottom")

        assert axis._tickDensity == pytest.approx(2.0)
        assert plain._tickDensity == pytest.approx(1.0)

    def test_major_tick_spacing_is_half_of_default(self, qapp):
        """spacing = dif / (2.25 × density × √(size/ref))：密度 2 → 间距减半"""
        axis = TimestampAxisItem(orientation="bottom")
        plain = pg.AxisItem(orientation="bottom")

        spacing_dense = axis.tickSpacing(0, 1000, 800)[0][0]
        spacing_plain = plain.tickSpacing(0, 1000, 800)[0][0]

        assert spacing_dense == pytest.approx(spacing_plain / 2.0)


class TestIndexToTimeTextAdapter:
    def test_sample_text_comes_from_first_row(self, mapping):
        class _Pw:
            factor, offset = 1.0, 0.0

        adapter = IndexToTimeTextAdapter(mapping, _Pw())
        assert adapter.sample_text == "11:51:53"

    def test_call_undoes_offset_factor(self, mapping):
        """显示 X = offset + factor × 行号，适配器必须按同口径反解"""

        class _Pw:
            factor, offset = 2.0, 10.0

        adapter = IndexToTimeTextAdapter(mapping, _Pw())
        # 显示 X 510 → 行号 (510-10)/2 = 250 → 12:08:29
        assert adapter(510.0) == "12:08:29"

    def test_zero_factor_returns_none(self, mapping):
        class _Pw:
            factor, offset = 0.0, 0.0

        adapter = IndexToTimeTextAdapter(mapping, _Pw())
        assert adapter(100.0) is None

    def test_adapter_keeps_mapping_semantics_with_plain_attrs(self, mapping):
        """pw.factor/offset 是普通 Python 属性，无需 Qt 对象存活也能反解"""

        class _Pw:
            factor, offset = 1.0, 0.0

        adapter = IndexToTimeTextAdapter(mapping, _Pw())
        assert adapter(1.0) == "11:51:53"
        assert adapter(2.0) == "11:51:57"
