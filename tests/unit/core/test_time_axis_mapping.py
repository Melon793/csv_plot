"""时间戳 X 轴映射（TimeAxisMapping / build_timestamps）的单测

口径背景：CSV/Excel 路径下游标、标注、时间校正都工作在**索引空间**
（显示 X = offset + factor × 行号，1-based）。时间戳 X 轴只在轴渲染层把
行号翻译成时间文本 —— 本文件守住翻译机的数值行为。
"""

import numpy as np
import pandas as pd
import pytest

from src.core.time_axis_mapping import (
    TICK_FORMAT_TIME_ONLY,
    TICK_FORMAT_WITH_DATE,
    TimeAxisMapping,
    build_timestamps,
)


def _uniform_series(n: int, start: str, freq: str) -> pd.Series:
    return pd.Series(pd.date_range(start, periods=n, freq=freq))


class TestBuildTimestamps:
    def test_uniform_series_maps_to_unix_seconds(self):
        series = _uniform_series(3, "2026-10-03 11:51:53", "4s")
        ts = build_timestamps(series)

        assert ts.dtype == np.float64
        expected = pd.Timestamp("2026-10-03 11:51:53").timestamp()
        assert ts[0] == pytest.approx(expected)
        assert ts[1] - ts[0] == pytest.approx(4.0)

    def test_nat_becomes_nan(self):
        series = pd.Series(
            pd.to_datetime(["2026-10-03 11:51:53", None, "2026-10-03 11:52:01"])
        )
        ts = build_timestamps(series)

        assert np.isnan(ts[1])
        assert np.isfinite(ts[0]) and np.isfinite(ts[2])

    def test_none_series_entry_is_coerced(self):
        # loader 对坏时间戳用 errors="coerce" 的口径一致：洗成 NaT → NaN
        series = pd.Series(["垃圾", "11:51:53"])
        ts = build_timestamps(pd.to_datetime(series, errors="coerce"))
        assert np.isnan(ts).sum() >= 1

    def test_string_series_with_format_parses_like_production(self):
        """生产路径 CSV 时间列保持字符串 dtype（do_parse_date 默认 False），
        映射器按 date_formats 的格式串解析；纯时刻由 pandas 补"今天"。"""
        series = pd.Series(["11:51:53", "11:51:57", "11:52:01"])
        ts = build_timestamps(series, date_format="%H:%M:%S")

        assert np.isfinite(ts).all()
        assert ts[1] - ts[0] == pytest.approx(4.0)
        assert ts[2] - ts[0] == pytest.approx(8.0)

    def test_string_series_with_bad_rows_becomes_nan(self):
        series = pd.Series(["11:51:53", "无效", "11:52:01"])
        ts = build_timestamps(series, date_format="%H:%M:%S")

        assert np.isfinite(ts[0]) and np.isfinite(ts[2])
        assert np.isnan(ts[1])


class TestTickFormatDetection:
    def test_pure_time_format_always_time_only(self):
        ts = build_timestamps(_uniform_series(4, "2026-10-03 11:00:00", "1h"))
        mapping = TimeAxisMapping(ts, date_format="%H:%M:%S")
        assert mapping.tick_format == TICK_FORMAT_TIME_ONLY

    def test_same_date_datetime_shows_time_only(self):
        ts = build_timestamps(_uniform_series(4, "2026-10-03 11:00:00", "1h"))
        mapping = TimeAxisMapping(ts, date_format="%Y-%m-%d %H:%M:%S")
        assert mapping.tick_format == TICK_FORMAT_TIME_ONLY

    def test_cross_day_datetime_shows_date_prefix(self):
        ts = build_timestamps(_uniform_series(3, "2026-10-03 23:00:00", "12h"))
        mapping = TimeAxisMapping(ts, date_format="%Y-%m-%d %H:%M:%S")
        assert mapping.tick_format == TICK_FORMAT_WITH_DATE

    def test_empty_mapping_falls_back_to_time_only(self):
        mapping = TimeAxisMapping(np.array([np.nan]), date_format="%Y-%m-%d")
        assert mapping.tick_format == TICK_FORMAT_TIME_ONLY


class TestTextForIndex:
    def test_row_one_maps_to_first_timestamp(self):
        mapping = TimeAxisMapping(
            build_timestamps(_uniform_series(2000, "2026-10-03 11:51:53", "4s")),
            date_format="%H:%M:%S",
        )
        assert mapping.text_for_index(1.0) == "11:51:53"

    def test_fractional_index_interpolates_between_rows(self):
        series = _uniform_series(10, "2026-10-03 00:00:00", "4s")
        mapping = TimeAxisMapping(build_timestamps(series), date_format="%H:%M:%S")
        # 行 1.5 = 行 1 与行 2 的中点 → +2s
        assert mapping.text_for_index(1.5) == "00:00:02"

    def test_out_of_range_returns_none_with_half_row_tolerance(self):
        mapping = TimeAxisMapping(
            build_timestamps(_uniform_series(10, "2026-10-03 00:00:00", "4s")),
            date_format="%H:%M:%S",
        )
        assert mapping.text_for_index(10.4) is not None  # 半行容差内
        assert mapping.text_for_index(-5.0) is None
        assert mapping.text_for_index(99.0) is None

    def test_non_finite_and_empty_return_none(self):
        mapping = TimeAxisMapping(
            build_timestamps(_uniform_series(10, "2026-10-03 00:00:00", "4s")),
            date_format="%H:%M:%S",
        )
        assert mapping.text_for_index(float("nan")) is None
        assert TimeAxisMapping(np.array([np.nan])).text_for_index(1.0) is None

    def test_cross_day_format_prefixes_the_date(self):
        series = pd.Series(
            pd.to_datetime(["2026-10-03 23:59:00", "2026-10-04 00:01:00"])
        )
        mapping = TimeAxisMapping(build_timestamps(series))
        text = mapping.text_for_index(2.0)
        assert text == "10-04 00:01:00"
