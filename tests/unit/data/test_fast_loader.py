"""FastDataLoader 单元测试：格式检测（分隔符/标题行/单位行）与 CSV 加载。"""

from __future__ import annotations

import numpy as np
import pandas as pd
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


class TestEncodingFallbackChain:
    """编码回退链必须真正生效（V6.0 P0-5：旧实现用 errors="replace" 永不抛错，
    首轮即 break → 整条链是死代码，误判编码被静默采纳）。"""

    @staticmethod
    def _fake_detect(encoding: str, coherence: float):
        """把 charset_normalizer 的探测结果钉成指定编码（模拟误判）"""

        class _Match:
            def __init__(self):
                self.encoding = encoding
                self.coherence = coherence

        class _Result:
            def best(self):
                return _Match()

        return lambda raw: _Result()

    def test_auto_detect_falls_through_wrong_candidate(self, tmp_path, monkeypatch):
        """探测链误判 utf-8（高置信度）时，严格解码须筛掉它并选中 gb18030"""
        f = write_csv(
            tmp_path / "gbk.csv",
            header=["时间", "速度", "转速"], units=["s", "km/h", "rpm"],
            rows=[[0.0, 10.5, 800], [0.1, 11.0, 810]],
            encoding="gb18030",
        )
        monkeypatch.setattr(
            "charset_normalizer.from_bytes", self._fake_detect("utf-8", 0.95)
        )

        fmt = FastDataLoader.auto_detect(str(f))

        assert fmt.encoding == "gb18030"
        assert fmt.sep == ","
        assert fmt.has_unit is True

    def test_header_units_falls_back_to_gb18030(self, tmp_path):
        """表头/单位链须含 gb18030：上游误传 utf-8 时不得退化成 cp1252 乱码"""
        f = write_csv(
            tmp_path / "gbk2.csv",
            header=["时间", "速度", "转速"], units=["s", "km/h", "rpm"],
            rows=[[0.0, 10.5, 800], [0.1, 11.0, 810]],
            encoding="gb18030",
        )

        loader = FastDataLoader(str(f), has_unit=True, sep=",", encoding="utf-8")

        assert loader.var_names == ["时间", "速度", "转速"]
        assert loader.encoding_used == "gb18030"
        assert loader.units["速度"] == "km/h"
        assert loader.datalength == 2


class TestBadLineTolerance:
    """dtype 推断样本必须与正式分块读取同口径跳过坏行（V6.0 P2-36）。

    样本读取少一个 `on_bad_lines="skip"`，后果不是「少一行」而是整个文件打不开：
    首 200 行内一条列数超标的行就抛 `ParserError: Expected 3 fields in line 4, saw 4`，
    而这些行在正式读取里本来就会被跳过。
    """

    @staticmethod
    def _csv(tmp_path, n: int, corrupt_at: list[int]):
        rows = []
        for i in range(n):
            row = [float(i), 10.0 + i, 800 + i]
            if i in corrupt_at:
                row.append("JUNK")  # 多出一列 → 该行列数超标
            rows.append(row)
        return write_csv(
            tmp_path / "bad_line.csv",
            header=["time", "speed", "rpm"],
            units=["s", "km/h", "rpm"],
            rows=rows,
        )

    def test_bad_line_inside_sample_window_still_loads(self, tmp_path):
        loader = FastDataLoader(
            str(self._csv(tmp_path, 300, [5])), has_unit=True, sep=","
        )
        assert loader.datalength == 299

    def test_sample_window_and_rest_now_behave_alike(self, tmp_path):
        """窗口外的坏行本来就只丢一行 —— 修复后窗口内也是同样结果。"""
        inside = FastDataLoader(
            str(self._csv(tmp_path, 300, [5])), has_unit=True, sep=","
        )
        outside = FastDataLoader(
            str(self._csv(tmp_path, 300, [250])), has_unit=True, sep=","
        )
        assert inside.datalength == outside.datalength == 299

    def test_skipping_the_line_does_not_shift_the_remaining_rows(self, tmp_path):
        loader = FastDataLoader(
            str(self._csv(tmp_path, 300, [5])), has_unit=True, sep=","
        )
        times = [float(v) for v in loader.df["time"].iloc[:8]]
        assert times == [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0], "只丢第 5 行，其余不得错位"
        assert loader.df["speed"].iloc[5] == pytest.approx(16.0)


class TestBlankLineRowAccounting:
    """表头之前的空行不得让 skiprows 错位（全仓审查 P1-5）。

    采样时 `auto_detect` 把空行过滤掉再找表头，返回的 header_row 却是
    **「过滤后列表的下标」**，而下游 pandas 的 `skiprows` 数的是**物理行**。
    文件里只要出现空行（导出程序爱在描述块和表头之间留一条），就会整段错位：
    少跳一行 → 把单位行当表头读；多跳一行 → 丢掉首行数据。
    """

    #: 描述行 2 列 → 表头 3 列（列数更多会被选为表头），中间夹一条空行
    WITH_BLANK_BEFORE_HEADER = (
        "note,ver\n"
        "\n"
        "time,speed,rpm\n"
        "s,km/h,rpm\n"
        "0.0,10.0,800\n"
        "0.1,10.5,810\n"
    )

    def _write(self, tmp_path, text, name="blank_line.csv"):
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_header_row_is_counted_in_physical_lines(self, tmp_path):
        path = self._write(tmp_path, self.WITH_BLANK_BEFORE_HEADER)

        fmt = FastDataLoader.auto_detect(str(path))

        assert fmt.sep == ","
        assert fmt.has_unit is True
        # 过滤下标是 1（第 2 个非空行），物理行号是 2（desc + 空行）
        assert fmt.header_row == 2, "header_row 必须是物理行号口径"

    def test_loader_gets_right_columns_and_keeps_every_row(self, tmp_path):
        path = self._write(tmp_path, self.WITH_BLANK_BEFORE_HEADER)
        fmt = FastDataLoader.auto_detect(str(path))

        loader = FastDataLoader(
            str(path),
            sep=fmt.sep,
            desc_rows=fmt.header_row,
            has_unit=fmt.has_unit,
            encoding=fmt.encoding,
        )

        assert loader.var_names == ["time", "speed", "rpm"]
        assert loader.datalength == 2, "首行数据不得被当成表头/单位吃掉"
        # 用 approx：默认 downcast_float=True 会把列降成 float32，
        # 0.1 存成 0.10000000149…（这本身是 P1-8 记的那条精度问题）
        assert [float(v) for v in loader.df["time"]] == [
            pytest.approx(0.0), pytest.approx(0.1)
        ]
        assert [float(v) for v in loader.df["speed"]] == [
            pytest.approx(10.0), pytest.approx(10.5)
        ]

    def test_file_without_blank_lines_is_unchanged(self, tmp_path):
        """对照组：没有空行时物理行号与过滤下标重合，行为不得变。"""
        path = self._write(
            tmp_path,
            self.WITH_BLANK_BEFORE_HEADER.replace("note,ver\n\n", "note,ver\n"),
            name="no_blank.csv",
        )

        fmt = FastDataLoader.auto_detect(str(path))

        assert fmt.header_row == 1
        assert fmt.has_unit is True

    def test_several_leading_blank_lines(self, tmp_path):
        path = self._write(
            tmp_path,
            "\n\n\n" + self.WITH_BLANK_BEFORE_HEADER,
            name="leading_blanks.csv",
        )

        fmt = FastDataLoader.auto_detect(str(path))
        loader = FastDataLoader(
            str(path),
            sep=fmt.sep,
            desc_rows=fmt.header_row,
            has_unit=fmt.has_unit,
            encoding=fmt.encoding,
        )

        assert fmt.header_row == 5  # 3 条空行 + desc + 1 条空行
        assert loader.var_names == ["time", "speed", "rpm"]
        assert loader.datalength == 2


class TestLateTextDtypeTolerance:
    """采样窗口外的非数值文本不得让整文件打不开（全仓审查 P1-6）。

    dtype 是拿前 `max_rows_infer`（默认 200）行推断的，而正式读取带着这个 dtype。
    文件后段一旦出现文本（现场常见：结尾几行状态标记、传感器故障行写字符串），
    pandas 直接抛 `ValueError: could not convert string to float` —— 用户看到的
    不是「少一个点」，而是「这个文件打不开」。
    """

    @staticmethod
    def _csv(tmp_path, rows_total=251, bad_at=250, bad_token="SPD_LOST", name=None):
        rows = []
        for i in range(rows_total):
            if i == bad_at:
                rows.append([f"{i * 0.1:.1f}", bad_token, 800 + i])
            else:
                rows.append([f"{i * 0.1:.1f}", f"{10.0 + i * 0.5:.2f}", 800 + i])
        return write_csv(
            tmp_path / (name or "late_text.csv"),
            header=["time", "speed", "rpm"],
            units=["s", "km/h", "rpm"],
            rows=rows,
        )

    def test_file_opens_and_only_the_bad_cell_is_missing(self, tmp_path):
        loader = FastDataLoader(
            str(self._csv(tmp_path)), has_unit=True, sep=",", desc_rows=0
        )

        assert loader.datalength == 251, "整行都该保留，只有那一个单元格缺失"
        col = loader.df["speed"]
        assert int(col.isna().sum()) == 1
        # 只有真混入文本的列进入 repairs；干净的列必须仍是数值 dtype
        # （重读不带 dtype，不能把所有列都退化成字符串再转）
        assert loader.dtype_repairs == {"speed": 1}
        assert pd.api.types.is_numeric_dtype(loader.df["time"])
        assert pd.api.types.is_numeric_dtype(loader.df["rpm"])
        # 其余数值不受影响（第 250 行才是坏格，取它的上一行）
        assert float(col.iloc[0]) == pytest.approx(10.0)
        assert float(col.iloc[249]) == pytest.approx(10.0 + 249 * 0.5)

    def test_bad_cell_inside_sample_window_behaves_the_same(self, tmp_path):
        """窗口内的坏格会改变**推断结果**（该列降成文本列），窗口外走容错重读
        （数值列 + 那一格缺失）。两条路径都必须：文件打得开、行的数量一致、
        缺的恰好是那一格 —— 坏格落在第几行不再决定文件能否打开。
        """
        inside = FastDataLoader(
            str(self._csv(tmp_path, bad_at=50, name="in.csv")),
            has_unit=True, sep=",", desc_rows=0,
        )
        outside = FastDataLoader(
            str(self._csv(tmp_path, bad_at=250, name="out.csv")),
            has_unit=True, sep=",", desc_rows=0,
        )

        assert inside.datalength == outside.datalength == 251
        inside_num = pd.to_numeric(inside.df["speed"], errors="coerce")
        outside_num = pd.to_numeric(outside.df["speed"], errors="coerce")
        assert int(inside_num.isna().sum()) == int(outside_num.isna().sum()) == 1
        assert inside_num.index[inside_num.isna()].tolist() == [50]
        assert outside_num.index[outside_num.isna()].tolist() == [250]
        assert outside.dtype_repairs.get("speed") == 1
        # 两列在「都有效的那些行」上取值完全一致
        both_valid = inside_num.notna() & outside_num.notna()
        assert int(both_valid.sum()) == 249  # 251 行里两个文件各缺一格，且位置不同
        assert float((inside_num[both_valid] - outside_num[both_valid]).abs().max()) == 0.0

    def test_na_token_still_counts_as_missing_not_a_repair(self, tmp_path):
        """`_NA_VALUES` 里的记号本来就走缺失值通道，不该被算进「容错重读」。"""
        path = write_csv(
            tmp_path / "na_token.csv",
            header=["time", "speed", "rpm"],
            units=["s", "km/h", "rpm"],
            rows=[["0.0", "10.0", "800"], ["0.1", "N/A", "810"]],
        )
        loader = FastDataLoader(str(path), has_unit=True, sep=",", desc_rows=0)

        assert loader.datalength == 2
        assert getattr(loader, "dtype_repairs", {}) == {}

    @pytest.mark.parametrize(
        "message, expected",
        [
            ("could not convert string to float: 'SPD_LOST'", True),
            ("Cannot cast array data from dtype('O') to dtype('float32') "
             "according to the rule 'safe'", True),
            ("Expected 3 fields in line 4, saw 4", False),
            ("Unicode strings with encoding declaration are not supported", False),
        ],
    )
    def test_only_the_narrow_dtype_signatures_are_tolerated(self, message, expected):
        assert FastDataLoader._is_dtype_conversion_error(ValueError(message)) is expected
