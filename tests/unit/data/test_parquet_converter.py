"""parquet_converter 转换器测试（P3）。

覆盖 DoD 11 条：S1–S5 各一组、有效性 8 行规则表、降档三细节、溢出 inf
夹具、行数守恒 oracle、三条负向护栏、meta.json 内容、失败清理、
Excel 链 dtype 往返。夹具全部来自 tests/fixtures/data_factory.py。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.data.loader import FastDataLoader
from src.data.parquet_converter import (
    ENUM_LABEL_MAX,
    ParquetConversionError,
    _pandas_series_to_polars,
    convert_to_parquet,
)
from tests.fixtures.data_factory import make_simple_rows, write_csv, write_xlsx


def _convert(tmp_path, csv, *, has_unit=True, desc_rows=0, sep=",", encoding=None):
    outdir = tmp_path / "out"
    return convert_to_parquet(
        str(csv),
        has_unit=has_unit,
        desc_rows=desc_rows,
        sep=sep,
        encoding=encoding,
        outdir=outdir,
    )


def _read_parquet(result) -> pl.DataFrame:
    return pl.read_parquet(result.parquet_path)


def _meta(result) -> dict:
    return json.loads(Path(result.meta_path).read_text(encoding="utf-8"))


class TestS1NullValues:
    """S1：NA 判定 32 项并集（_NA_VALUES + '<NA>' + 'null'）。"""

    def test_na_tokens_become_null(self, tmp_path):
        # pandas 默认 NA 集里两个不在 _NA_VALUES 的 token + 常规 token
        csv = write_csv(
            tmp_path / "s1.csv",
            header=["a", "b", "c", "d"],
            units=["-", "-", "-", "-"],
            rows=[
                ["N/A", "<NA>", "null", ""],
                ["1.5", "2.5", "3.5", "4.5"],
            ],
        )
        result = _convert(tmp_path, csv)
        frame = _read_parquet(result)
        row0 = frame.row(0, named=True)
        assert row0["a"] is None
        assert row0["b"] is None  # pandas 默认集成员，_NA_VALUES 没有 → 必须显式补
        assert row0["c"] is None  # 同上
        assert row0["d"] is None  # 空串（empty_string_is_null）
        row1 = frame.row(1, named=True)
        assert row1 == {"a": 1.5, "b": 2.5, "c": 3.5, "d": 4.5}

    def test_parity_with_fast_loader_na(self, tmp_path):
        rows = [["<NA>", "null", "nan", "NA"], ["1", "2", "3", "4"]]
        csv = write_csv(
            tmp_path / "s1b.csv",
            header=["p", "q", "r", "s"],
            units=["-", "-", "-", "-"],
            rows=rows,
        )
        result = _convert(tmp_path, csv)
        got = _read_parquet(result).to_numpy()
        ref = FastDataLoader(str(csv)).df.to_numpy()
        # numpy 2.4 的 assert_array_equal 无 equal_nan 参数，用 array_equal
        assert np.array_equal(got, ref, equal_nan=True)


class TestS2RowConservation:
    """S2：坏行可检测 + 行数守恒 oracle。"""

    def test_fewer_fields_padded_not_dropped(self, tmp_path):
        # 字段过少：pandas 补 NaN 不丢行，polars 一致 → 守恒过
        csv = write_csv(
            tmp_path / "s2a.csv",
            header=["x", "y", "z"],
            units=["-", "-", "-"],
            rows=[[1, 2, 3], ["7", "", ""], [4, 5, 6]],  # 第二行只有 1 个字段
        )
        result = _convert(tmp_path, csv)
        assert result.rows == 3
        frame = _read_parquet(result)
        row1 = frame.row(1, named=True)
        assert row1["x"] == 7 and row1["y"] is None and row1["z"] is None

    def test_more_fields_raises(self, tmp_path):
        # 字段过多：转换链不得静默保留/截断。若畸形行落在头部探测区，允许
        # 与内存 loader 同抛 ParserError；若越过探测区则由转换器抛
        # ParquetConversionError 交给 D10 回退。
        csv = write_csv(
            tmp_path / "s2b.csv",
            header=["x", "y", "z"],
            units=["-", "-", "-"],
            rows=[["1", "2", "3", "99"], ["4", "5", "6"]],  # 首数据行 4 字段
        )
        with pytest.raises((ParquetConversionError, pd.errors.ParserError)):
            _convert(tmp_path, csv)
        assert not (tmp_path / "out" / "data.parquet").exists()

    def test_blank_line_fails_oracle(self, tmp_path):
        # 唯一静默分歧：空行 pandas 跳过、polars 读成全 null 行 → oracle 拦截
        path = tmp_path / "s2c.csv"
        path.write_text("x,y,z\n-,-,-\n1,2,3\n\n4,5,6\n", encoding="utf-8")
        with pytest.raises(ParquetConversionError, match="行数守恒"):
            _convert(tmp_path, path)

    def test_whitespace_line_fails_oracle(self, tmp_path):
        # 纯空格行与真空行同属 pandas 跳过、polars 读成 null 行的静默分歧。
        path = tmp_path / "s2c_ws.csv"
        path.write_text("x,y,z\n-,-,-\n1,2,3\n   \n4,5,6\n", encoding="utf-8")
        with pytest.raises(ParquetConversionError, match="行数守恒"):
            _convert(tmp_path, path)

    def test_quoted_separator_parity(self, tmp_path):
        # 引号内分隔符：pandas 与 polars 都把 "1,23" 解析为单字段（非坏行），
        # 两侧一致成功 → S2 等价面成立（v2.1 实测：polars 引号字段可解析）
        path = tmp_path / "s2d.csv"
        path.write_text('x,y,z\n-,-,-\n"1,23",4,5\n6,7,8\n', encoding="utf-8")
        result = _convert(tmp_path, path)
        ref = FastDataLoader(str(path)).df
        assert result.rows == len(ref) == 2
        # x 样本含非数值文本 → category → 低基数枚举（D8），码值还原后与
        # pandas 逐值一致
        labels = _meta(result)["enum_labels"]["x"]
        codes = _read_parquet(result)["x"].to_list()
        restored = [labels[c] for c in codes]
        assert restored == ref["x"].tolist() == ["1,23", "6"]

    def test_row_count_matches_fast_loader(self, tmp_path):
        csv = write_csv(
            tmp_path / "s2e.csv",
            header=["time", "speed", "rpm", "flag"],
            units=["s", "km/h", "-", "-"],
            rows=make_simple_rows(50),
        )
        result = _convert(tmp_path, csv)
        assert result.rows == 50 == len(FastDataLoader(str(csv)).df)


class TestS3ColumnNameFidelity:
    """S3：列名取自 _make_unique 产物，中文/空格/重复名保真。"""

    def test_chinese_and_space_and_duplicate_names(self, tmp_path):
        csv = write_csv(
            tmp_path / "s3.csv",
            header=["时间 列", "转速", "转速", "A_1"],
            units=["s", "-", "-", "-"],
            rows=[[1, 2, 3, 4], [5, 6, 7, 8]],
        )
        loader = FastDataLoader(str(csv))
        result = _convert(tmp_path, csv)
        frame = _read_parquet(result)
        # schema 键 = probe.var_names = FastDataLoader._var_names（含去重）
        assert frame.columns == loader._var_names
        # _make_unique 产 "转速_1"（计数从 1 起，且字面 A_1 保号）
        assert frame.columns == ["时间 列", "转速", "转速_1", "A_1"]


class TestS4CategoryAndEnum:
    """S4：类别顺序不假设（码值+同源表）；全空列。"""

    def test_enum_codes_and_labels_same_cast(self, tmp_path):
        # 出现序 Z 先于 A（pandas 排序会反过来）——码值下标必须对齐同源表
        values = ["Z", "A"] * 5 + ["Z"]
        csv = write_csv(
            tmp_path / "s4.csv",
            header=["state", "v"],
            units=["-", "-"],
            rows=[[v, i] for i, v in enumerate(values)],
        )
        result = _convert(tmp_path, csv)
        meta = _meta(result)
        assert meta["columns"]["state"]["is_enum"] is True
        # 出现序：Z 先出现
        assert meta["enum_labels"]["state"][0] == "Z"
        frame = _read_parquet(result)
        codes = frame["state"].to_list()
        # 码值 = labels.index(文本)：Z→0, A→1
        assert codes[0] == 0 and codes[1] == 1
        # 读回还原文本逐个正确（pd.Categorical.from_codes 同源）
        restored = pd.Categorical.from_codes(
            [c if c is not None else -1 for c in codes],
            categories=meta["enum_labels"]["state"],
        )
        assert list(restored) == values

    def test_enum_null_stays_null(self, tmp_path):
        csv = write_csv(
            tmp_path / "s4b.csv",
            header=["state", "v"],
            units=["-", "-"],
            rows=[["ON", 1], ["N/A", 2], ["OFF", 3]],
        )
        result = _convert(tmp_path, csv)
        frame = _read_parquet(result)
        assert frame["state"][1] is None  # 脏值 → 码值 null

    def test_all_empty_column(self, tmp_path):
        # 全空列：pandas category + 0 categories → meta all_empty + String 全 null
        csv = write_csv(
            tmp_path / "s4c.csv",
            header=["v", "empty"],
            units=["-", "-"],
            rows=[[1, ""], [2, ""], [3, ""]],
        )
        result = _convert(tmp_path, csv)
        meta = _meta(result)
        cm = meta["columns"]["empty"]
        assert cm["all_empty"] is True
        assert cm["dtype_str"] == "category"
        assert cm["validity"] == -1
        assert cm["is_enum"] is False
        frame = _read_parquet(result)
        assert frame["empty"].null_count() == 3

    def test_high_cardinality_text_not_enum(self, tmp_path):
        rows = [[f"txt_{i:04d}", i] for i in range(ENUM_LABEL_MAX + 1)]
        csv = write_csv(
            tmp_path / "s4d.csv", header=["id", "v"], units=["-", "-"], rows=rows
        )
        result = _convert(tmp_path, csv)
        meta = _meta(result)
        cm = meta["columns"]["id"]
        assert cm["is_enum"] is False
        assert cm["validity"] == -1  # to_numeric 失败
        assert "id" not in meta["enum_labels"]


class TestS5Encoding:
    """S5：转码是「抛错后重试」；常见路径无转码中间文件。"""

    def test_gb18030_data_area_transcoded(self, tmp_path):
        # 数据区含非 UTF-8 中文 → 直解析失败 → 按 gb18030 转码重试成功。
        # 数据区必须用「汉」类字：gb18030 编码 baba 不是合法 UTF-8；
        # 「状态」(d7b4ccac) 恰好是合法 UTF-8 序列，直读不报错 → 不转码
        # → 乱码（探针实测）
        rows = [[f"汉{i % 3}", i * 0.5] for i in range(10)]
        csv = write_csv(
            tmp_path / "s5.csv",
            header=["状态", "数值"],
            units=["-", "-"],
            rows=rows,
            encoding="gb18030",
        )
        result = _convert(tmp_path, csv, encoding="gb18030")
        meta = _meta(result)
        # 状态列 3 个唯一值 ≤ 200 → D8 枚举码值（Int32 + 出现序标签表）
        assert meta["enum_labels"]["状态"][0] == "汉0"
        codes = _read_parquet(result)["状态"].to_list()
        assert codes[0] == 0  # "汉0" → 出现序首类 → 码值 0
        # 转码中间文件必须删掉
        assert not (tmp_path / "out" / "_transcoded_utf8.csv").exists()

    def test_ascii_common_path_no_transcode(self, tmp_path):
        csv = write_csv(
            tmp_path / "s5b.csv",
            header=["a", "b"],
            units=["-", "-"],
            rows=[[1, 2], [3, 4]],
        )
        _convert(tmp_path, csv)
        files = sorted(p.name for p in (tmp_path / "out").iterdir())
        assert files == ["data.parquet", "meta.json"]

    def test_undecodable_raises_and_cleans(self, tmp_path, monkeypatch):
        # 转码也失败（按探测编码仍解不开）→ ParquetConversionError + 无残留
        csv = write_csv(
            tmp_path / "s5c.csv",
            header=["a", "b"],
            units=["-", "-"],
            rows=[[1, 2]],
        )

        def _boom(src, dst, encoding):
            raise UnicodeDecodeError(encoding, b"\xff", 0, 1, "bad")

        import src.data.parquet_converter as pc

        # 强制直解析走 utf-8 失败分支：给源文件塞一个非 UTF-8 字节
        data = Path(csv).read_bytes()
        Path(csv).write_bytes(data.replace(b"1,2", b"1,\xc82"))
        monkeypatch.setattr(pc, "_transcode_to_utf8", _boom)
        with pytest.raises(ParquetConversionError, match="转码失败"):
            _convert(tmp_path, csv)
        assert not (tmp_path / "out" / "_transcoded_utf8.csv").exists()

    def test_transcode_disk_error_raises_conversion(self, tmp_path, monkeypatch):
        # D10：转码中间文件写不进 outdir 也属于转换失败，必须包成
        # ParquetConversionError 让工厂清理并回退，不能把 OSError 原样上抛。
        csv = write_csv(
            tmp_path / "s5d.csv",
            header=["a", "b"],
            units=["-", "-"],
            rows=[[1, 2]],
        )

        def _boom(src, dst, encoding):
            raise OSError("read-only")

        import src.data.parquet_converter as pc

        data = Path(csv).read_bytes()
        Path(csv).write_bytes(data.replace(b"1,2", b"1,\xc82"))
        monkeypatch.setattr(pc, "_transcode_to_utf8", _boom)
        with pytest.raises(ParquetConversionError, match="转码失败"):
            _convert(tmp_path, csv)
        assert not (tmp_path / "out" / "_transcoded_utf8.csv").exists()

    def test_transcoded_second_read_raises_conversion(self, tmp_path, monkeypatch):
        # D10：转码后二次解析失败也不能把 polars 原生错误原样上抛，否则
        # 后续工厂的 ParquetConversionError 回退分支接不住。
        csv = write_csv(
            tmp_path / "s5e.csv",
            header=["a", "b"],
            units=None,
            rows=[[1, 2]],
        )

        import src.data.parquet_converter as pc

        data = Path(csv).read_bytes()
        Path(csv).write_bytes(data.replace(b"1,2", b"1,\xc82"))
        monkeypatch.setattr(pc, "_transcode_to_utf8", lambda *args: None)
        with pytest.raises(ParquetConversionError, match="转码后 CSV 解析失败"):
            convert_to_parquet(
                str(csv),
                has_unit=False,
                sep=",",
                outdir=tmp_path / "out2",
            )


class TestValidityRuleTable:
    """§2.1.1 有效性 8 行规则表：一行规则一个夹具一条断言。"""

    def _one(self, tmp_path, name, rows, header=None):
        header = header or ["v"]
        csv = write_csv(
            tmp_path / f"{name}.csv", header=header, units=["-"] * len(header), rows=rows
        )
        result = _convert(tmp_path, csv)
        return _meta(result)["columns"], _read_parquet(result)

    def test_date_column_validity_1(self, tmp_path):
        cm, _ = self._one(
            tmp_path,
            "d",
            [[f"00:00:{i:02d}.000"] for i in range(10)],
            header=["time"],
        )
        assert cm["time"]["validity"] == 1
        assert cm["time"]["dtype_str"] == "object"

    def test_all_empty_validity_minus_1(self, tmp_path):
        # 全空列须搭配数值列（单列全空 → 物理行成空行，会被 oracle 判废）
        cm, _ = self._one(tmp_path, "e", [["", 1], ["", 2]], header=["v", "w"])
        assert cm["v"]["validity"] == -1
        assert cm["v"]["all_empty"] is True

    def test_float_constant_validity_0(self, tmp_path):
        cm, _ = self._one(tmp_path, "c", [[5.0], [5.0], [5.0]])
        assert cm["v"]["validity"] == 0

    def test_float_varying_validity_1(self, tmp_path):
        cm, _ = self._one(tmp_path, "v1", [[1.0], [2.0], [3.0]])
        assert cm["v"]["validity"] == 1

    def test_float_with_nan_extremes(self, tmp_path):
        # NaN 不影响 min/max：[1.0, NaN, 3.0] → min=1 max=3 → 1
        cm, _ = self._one(tmp_path, "v2", [["1.0"], ["NaN"], ["3.0"]])
        assert cm["v"]["validity"] == 1

    def test_low_card_text_enum_overrides_grey(self, tmp_path):
        # 规则表「非数值 to_numeric 失败 → -1」在低基数列被 D8 枚举覆盖为
        # 1（有意正向差异：从灰变可选）。注：CSV 链不存在「全数字文本列」
        # ——pandas 样本推断直接判 float，规则表该行在 CSV 链不可达
        cm, _ = self._one(tmp_path, "n", [["A"], ["A"], ["B"]])
        assert cm["v"]["is_enum"] is True
        assert cm["v"]["validity"] == 1

    def test_text_to_numeric_fail_minus_1(self, tmp_path):
        # 高基数非数字 → to_numeric 失败 → -1（TestS4 已覆盖同路径，
        # 这里钉规则表行本身）
        rows = [[f"k{i}"] for i in range(ENUM_LABEL_MAX + 1)]
        cm, _ = self._one(tmp_path, "t", rows)
        assert cm["v"]["validity"] == -1


class TestDowncastAndInf:
    """降档三细节（dtype_map 同源）+ inf 清理。"""

    def test_small_values_downcast_float32(self, tmp_path):
        csv = write_csv(
            tmp_path / "dc.csv",
            header=["small", "big"],
            units=["-", "-"],
            rows=[[i * 0.5, 1e39 + i] for i in range(30)],
        )
        result = _convert(tmp_path, csv)
        frame = _read_parquet(result)
        assert frame["small"].dtype == pl.Float32  # max_abs <= F32_MAX
        assert frame["big"].dtype == pl.Float64   # max_abs > F32_MAX
        # 与 FastDataLoader 的 dtype 同源（probe 同一函数）
        ref = FastDataLoader(str(csv)).df
        assert str(ref["small"].dtype) == "float32"
        assert str(ref["big"].dtype) == "float64"

    def test_nanmax_skips_nan(self, tmp_path):
        # nanmax 跳 NaN：NaN 混大值 → 大值决定档位
        csv = write_csv(
            tmp_path / "dc2.csv",
            header=["mix"],
            units=["-"],
            rows=[["NaN"], ["1e39"], ["2.0"]],
        )
        result = _convert(tmp_path, csv)
        assert _read_parquet(result)["mix"].dtype == pl.Float64

    def test_overflow_literals_cleaned_to_null(self, tmp_path):
        # 1e400 / -1e400（混合列，v2.1 答案卡场景）：pandas 侧带 na_values
        # 读取时正溢出 1e400→NaN、-1e400→-inf→清理后 NaN；polars 侧 ±inf
        # → null 清理，读回 NaN → 逐列值级相等。
        # （整列全 1e400 的列两侧同判文本列——样本全 NA → category，非分歧）
        csv = write_csv(
            tmp_path / "inf.csv",
            header=["a", "b", "c"],
            units=["-", "-", "-"],
            rows=[["1e400", "-1e400", "1.5"], ["2.5", "-1e400", "2.5"]],
        )
        result = _convert(tmp_path, csv)
        frame = _read_parquet(result)
        ref = FastDataLoader(str(csv)).df
        for col in ("a", "b", "c"):
            got = frame[col].to_numpy()  # null → NaN
            assert not np.isinf(got).any(), col  # inf 必须清干净
            assert np.array_equal(got, ref[col].to_numpy(), equal_nan=True), col

    def test_bitwise_parity_float32(self, tmp_path):
        # 逐位与位模式一致（含 -0.0 符号位）
        rows = [["-0.0", "1e30", "-1e30"], ["0.0", "2.5", "-2.5"]]
        csv = write_csv(
            tmp_path / "bits.csv", header=["a", "b", "c"], units=["-", "-", "-"], rows=rows
        )
        result = _convert(tmp_path, csv)
        got = _read_parquet(result).to_numpy()
        ref = FastDataLoader(str(csv)).df.to_numpy()
        assert got.dtype == ref.dtype
        np.testing.assert_array_equal(got.view(np.uint32), ref.view(np.uint32))


class TestMetaJson:
    def test_meta_fields_complete(self, tmp_path):
        csv = write_csv(
            tmp_path / "m.csv",
            header=["time", "v", "state"],
            units=["s", "km/h", "-"],
            rows=[[f"00:00:{i:02d}.000", i * 0.5, "ON" if i % 2 else "OFF"] for i in range(10)],
        )
        result = _convert(tmp_path, csv)
        meta = _meta(result)
        for key in ("loader_type", "var_names", "units", "time_column_name",
                    "date_formats", "rows", "columns", "enum_labels", "source_fingerprint"):
            assert key in meta, key
        assert meta["loader_type"] == "parquet"
        assert meta["time_column_name"] is None
        assert meta["rows"] == 10
        assert meta["date_formats"] == {"time": "%H:%M:%S.%f"}
        assert meta["units"] == {"time": "s", "v": "km/h", "state": "-"}
        for col in ("time", "v", "state"):
            for field in ("dtype_str", "all_empty", "validity", "unit", "n_unique", "is_enum"):
                assert field in meta["columns"][col], (col, field)
        # 指纹格式 size:mtime:sha16
        parts = meta["source_fingerprint"].split(":")
        assert len(parts) == 3 and len(parts[2]) == 16


class TestNegativeGuardsAndCleanup:
    def test_corrupt_source_no_parquet_written(self, tmp_path):
        # 任意转换失败：outdir 不得残留 data.parquet（D10 由工厂清目录，
        # 但转换器自身必须保证不写半截文件）
        csv = write_csv(
            tmp_path / "ng.csv", header=["x", "y"], units=["-", "-"], rows=[[1, 2, 3]]
        )
        with pytest.raises((ParquetConversionError, pd.errors.ParserError)):
            _convert(tmp_path, csv)
        assert not (tmp_path / "out" / "data.parquet").exists()

    def test_progress_callbacks_emitted(self, tmp_path):
        csv = write_csv(
            tmp_path / "pg.csv", header=["x"], units=["-"], rows=[[i] for i in range(5)]
        )
        marks: list[int] = []
        convert_to_parquet(
            str(csv), has_unit=True, outdir=tmp_path / "out2", progress_cb=marks.append
        )
        assert marks[0] <= 10 and marks[-1] == 95
        assert len(marks) >= 4  # 不少于今天 5/15/20/100 的粒度


class TestExcelChain:
    def test_xlsx_roundtrip_dtypes(self, tmp_path):
        import datetime as dt

        xlsx = write_xlsx(
            tmp_path / "e.xlsx",
            header=["t", "v", "s"],
            units=["-", "-", "-"],
            rows=[
                # i + 0.5 强制小数：整数浮点会被 openpyxl 存成整数字面量，
                # calamine 读回 int64（那也是忠实镜像，但本测试钉 float 链路
                [dt.datetime(2026, 1, 1, 0, 0, i), i + 0.5, "ON" if i % 2 else "OFF"]
                for i in range(10)
            ],
        )
        result = convert_to_parquet(
            str(xlsx), is_excel=True, sheet_name=0, outdir=tmp_path / "out"
        )
        frame = _read_parquet(result)
        meta = _meta(result)
        assert frame["t"].dtype == pl.Datetime  # 真 datetime → Datetime 存
        assert frame["v"].dtype in (pl.Float32, pl.Float64)
        assert meta["columns"]["s"]["is_enum"] is True
        assert meta["columns"]["t"]["dtype_str"].startswith("datetime64")

    def test_datetime64_seconds_upsampled_to_ms(self):
        # datetime64[s] 必须升采 Datetime("ms") 存（polars 无 "s" 单位）
        s = pd.Series(
            pd.to_datetime(["2026-01-01 00:00:01", "2026-01-01 00:00:02"]).astype(
                "datetime64[s]"
            ),
            name="t",
        )
        ps, entry = _pandas_series_to_polars(pl, "t", s)
        assert ps.dtype == pl.Datetime("ms")
        assert entry["dtype_str"] == "datetime64[s]"
        # 读回还原（to_numpy 路径——pyarrow 非本项目依赖，to_pandas 不可用）
        restored = pd.Series(ps.to_numpy()).astype("datetime64[s]")
        assert restored.tolist() == s.tolist()

    def test_datetime64_ns_kept(self):
        # pandas 3.0 to_datetime 推断 us——ns 列必须显式构造（探针实测）
        s = pd.Series(
            np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[ns]"), name="t"
        )
        ps, entry = _pandas_series_to_polars(pl, "t", s)
        assert ps.dtype == pl.Datetime("ns")
        assert entry["dtype_str"] == "datetime64[ns]"

    def test_excel_text_high_cardinality_not_enum(self, tmp_path):
        xlsx = write_xlsx(
            tmp_path / "h.xlsx",
            header=["id", "v"],
            units=["-", "-"],
            rows=[[f"row{i:04d}", i] for i in range(ENUM_LABEL_MAX + 1)],
        )
        result = convert_to_parquet(
            str(xlsx), is_excel=True, sheet_name=0, outdir=tmp_path / "out"
        )
        meta = _meta(result)
        assert meta["columns"]["id"]["is_enum"] is False
        # pandas 3.0 read_excel 文本列是新 str dtype（2.x 是 object）
        assert meta["columns"]["id"]["dtype_str"] in ("object", "str")

    def test_excel_string_date_is_not_enum(self, tmp_path):
        """D9：Excel 字符串日期列必须按文本日期存，不能落入 D8 枚举列。"""
        import pandas as pd

        from src.data.parquet_lazy_loader import ParquetLazyLoader
        from src.data.temp_cache_dir import TempCacheDir

        xlsx = write_xlsx(
            tmp_path / "string_date.xlsx",
            header=["date_str", "v"],
            units=["-", "-"],
            rows=[[f"2024-01-{(i % 28) + 1:02d}", float(i)] for i in range(12)],
        )
        result = convert_to_parquet(
            str(xlsx), is_excel=True, sheet_name=0, outdir=tmp_path / "out"
        )
        meta = _meta(result)
        assert meta["date_formats"] == {"date_str": "%Y-%m-%d"}
        assert meta["columns"]["date_str"]["is_enum"] is False
        assert meta["columns"]["date_str"]["validity"] == 1
        assert meta["columns"]["date_str"]["dtype_str"] in ("object", "str")

        loader = ParquetLazyLoader(str(xlsx), TempCacheDir(Path(result.meta_path).parent))
        try:
            assert str(loader.get_series("date_str").dtype) in ("object", "str")
            _, y, _, text_map = loader.get_value_from_name("date_str")
            assert text_map == {}
            assert isinstance(y, pd.Series)
        finally:
            loader.close()


if __name__ == "__main__":
    pytest.main([__file__])
