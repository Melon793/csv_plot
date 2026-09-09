"""FastDataLoader 单元测试：格式检测（分隔符/标题行/单位行）与 CSV 加载。"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.data_types import AutoDetectError
from src.data.loader import FastDataLoader
from tests.fixtures.data_factory import make_simple_rows, write_csv

SIMPLE_HEADER = ["time", "speed", "rpm", "flag"]
SIMPLE_UNITS = ["s", "km/h", "rpm", "-"]


# ---------- 分隔符检测 ----------

class TestDetectSep:
    def test_comma(self):
        lines = ["a,b,c", "1,2,3", "4,5,6"]
        assert FastDataLoader._detect_sep_from_lines(lines) == ","

    def test_semicolon(self):
        lines = ["a;b;c", "1;2;3", "4;5;6"]
        assert FastDataLoader._detect_sep_from_lines(lines) == ";"

    def test_tab(self):
        lines = ["a\tb\tc", "1\t2\t3", "4\t5\t6"]
        assert FastDataLoader._detect_sep_from_lines(lines) == "\t"

    def test_no_separator_returns_none(self):
        lines = ["abc", "def", "ghi"]
        assert FastDataLoader._detect_sep_from_lines(lines) is None


# ---------- 标题行检测 ----------

class TestDetectHeader:
    def test_plain_header_at_first_line(self):
        lines = ["time,speed,rpm", "0.0,10.5,800", "0.1,11.0,810"]
        assert FastDataLoader._detect_header_from_lines(lines, ",") == 0

    def test_metadata_region_jump_detection(self):
        """元数据区（2 列）后接真实标题行（8 列）→ 列数跳跃定位到标题行"""
        lines = [
            "File,test.csv",
            "Author,tester",
            "time,speed,rpm,temp,a,b,c,d",
            "0.0,10.5,800,25,1,2,3,4",
        ]
        assert FastDataLoader._detect_header_from_lines(lines, ",") == 2

    def test_picks_max_column_candidate_without_jump(self):
        """无数量级跳跃时选择列数更多的候选行"""
        lines = [
            "key,value",
            "time,speed,rpm,temp",
            "0.0,10.5,800,25",
        ]
        assert FastDataLoader._detect_header_from_lines(lines, ",") == 1


# ---------- 单位行检测 ----------

class TestDetectHasUnit:
    def test_unit_row_detected(self):
        lines = ["time,speed,rpm", "s,km/h,rpm", "0.0,10.5,800"]
        assert FastDataLoader._detect_has_unit_from_lines(lines, ",", 0) is True

    def test_numeric_row_not_unit(self):
        lines = ["time,speed,rpm", "0.0,10.5,800", "0.1,11.0,810"]
        assert FastDataLoader._detect_has_unit_from_lines(lines, ",", 0) is False

    def test_header_at_last_line_returns_false(self):
        lines = ["time,speed,rpm"]
        assert FastDataLoader._detect_has_unit_from_lines(lines, ",", 0) is False


# ---------- auto_detect 统一入口 ----------

class TestAutoDetect:
    def test_simple_csv_with_units(self, tmp_path):
        f = write_csv(
            tmp_path / "simple.csv",
            header=SIMPLE_HEADER, units=SIMPLE_UNITS, rows=make_simple_rows(),
        )
        fmt = FastDataLoader.auto_detect(str(f))
        assert fmt.sep == ","
        assert fmt.header_row == 0
        assert fmt.has_unit is True
        assert fmt.encoding is not None

    def test_too_few_lines_raises(self, tmp_path):
        f = tmp_path / "one_line.csv"
        f.write_text("only_one_line_no_sep\n", encoding="utf-8")
        with pytest.raises(AutoDetectError):
            FastDataLoader.auto_detect(str(f))

    def test_no_separator_raises(self, tmp_path):
        f = tmp_path / "no_sep.csv"
        f.write_text("lineA\nlineB\nlineC\n", encoding="utf-8")
        with pytest.raises(AutoDetectError):
            FastDataLoader.auto_detect(str(f))


# ---------- 完整加载 ----------

class TestFastDataLoaderLoad:
    @pytest.fixture
    def simple_csv(self, tmp_path):
        return write_csv(
            tmp_path / "simple.csv",
            header=SIMPLE_HEADER, units=SIMPLE_UNITS, rows=make_simple_rows(20),
        )

    def test_header_and_units(self, simple_csv):
        loader = FastDataLoader(str(simple_csv), has_unit=True, sep=",")
        assert loader.var_names == SIMPLE_HEADER
        assert loader.units["speed"] == "km/h"
        assert loader.datalength == 20
        assert loader.LOADER_TYPE == "csv"

    def test_validity_flags(self, simple_csv):
        """speed/rpm 非常量 → 1，flag 常量 → 0"""
        loader = FastDataLoader(str(simple_csv), has_unit=True, sep=",")
        validity = loader.df_validity
        assert validity["speed"] == 1
        assert validity["rpm"] == 1
        assert validity["flag"] == 0

    def test_float_downcast_to_float32(self, simple_csv):
        loader = FastDataLoader(str(simple_csv), has_unit=True, sep=",")
        assert loader.df["speed"].dtype == np.float32

    def test_downcast_false_still_loads_correct_values(self, simple_csv):
        """downcast_float=False 跳过后处理降转换；
        注意 schema 推断层仍可能将安全数值列读为 float32，此处只验证数据正确性"""
        loader = FastDataLoader(
            str(simple_csv), has_unit=True, sep=",", downcast_float=False
        )
        assert loader.datalength == 20
        assert loader.df["speed"].iloc[0] == pytest.approx(10.0)
        assert loader.df["speed"].iloc[-1] == pytest.approx(19.5)

    def test_na_values_parsed(self, tmp_path):
        rows = [[0.0, 10.5], [0.1, "N/A"], [0.2, 12.5]]
        f = write_csv(
            tmp_path / "na.csv", header=["time", "v"], units=["s", "-"], rows=rows
        )
        loader = FastDataLoader(str(f), has_unit=True, sep=",")
        assert np.isnan(loader.df["v"].iloc[1])
        assert loader.df["v"].iloc[0] == pytest.approx(10.5)

    def test_duplicate_column_names_made_unique(self, tmp_path):
        f = write_csv(
            tmp_path / "dup.csv",
            header=["a", "a", "b"], units=["-", "-", "-"],
            rows=[[1, 2, 3], [4, 5, 6]],
        )
        loader = FastDataLoader(str(f), has_unit=True, sep=",")
        assert loader.var_names == ["a", "a_1", "b"]

    def test_explicit_gb18030_encoding(self, tmp_path):
        f = write_csv(
            tmp_path / "gb.csv",
            header=["时间", "转速"], units=["s", "rpm"],
            rows=[[0.0, 800], [0.1, 900]],
            encoding="gb18030",
        )
        loader = FastDataLoader(str(f), has_unit=True, sep=",", encoding="gb18030")
        assert loader.var_names == ["时间", "转速"]
        assert loader.datalength == 2

    def test_without_unit_row(self, tmp_path):
        f = write_csv(
            tmp_path / "no_unit.csv",
            header=["a", "b"], rows=[[1, 2], [3, 4]],
        )
        loader = FastDataLoader(str(f), has_unit=False, sep=",")
        assert loader.datalength == 2
        assert loader.units["a"] == "-"

    def test_get_value_from_name_returns_quadruple(self, simple_csv):
        loader = FastDataLoader(str(simple_csv), has_unit=True, sep=",")
        index, values, unit, enum_map = loader.get_value_from_name("speed")
        assert len(index) == 20
        assert len(values) == 20
        assert unit == "km/h"
        assert enum_map == {}

    def test_progress_callback_reaches_100(self, simple_csv):
        progress: list[int] = []
        FastDataLoader(str(simple_csv), has_unit=True, sep=",", _progress=progress.append)
        assert progress[-1] == 100
        assert all(0 <= p <= 100 for p in progress)
