"""ParquetLazyLoader 等价性测试（P4 DoD 1/2/3）。

DoD 1（CSV）与 DoD 2（Excel）：同一夹具喂内存 loader（FastDataLoader /
ExcelDataLoader）与「转换 + ParquetLazyLoader」，元数据与数据逐位相等
（np.array_equal，不用 allclose）；日期列额外断言 dtype。
DoD 3（枚举列有意差异，D8）：**单独一组断言**，不混进通用等价断言——
内存 loader validity=-1 / 新 loader validity=1；get_series 两边都返回
文本 category；get_value_from_name 新 loader 返回码值 + 非空 text_map。
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from src.data.excel_loader import ExcelDataLoader
from src.data.loader import FastDataLoader
from src.data.parquet_converter import ParquetConversionError
from tests.fixtures.data_factory import write_csv, write_xlsx

# 高基数列需要 > 200 unique（ENUM_LABEL_MAX=200）才不进枚举分支
N_ROWS = 205

# 脏值 token 循环表（S1 32 项集合的代表性子集；全脏列会变全空列）
_DIRTY_TOKENS = [
    "NULL", "1.5", "----", "2.5", "data err", "3.5", "**", "4.5",
    "inf", "5.5", "null ", "null", "<NA>", "6.5", "nan", "7.5",
    "N/A", "8.5", "None", "9.5",
]

_CSV_HEADER = [
    "t", "转速", "转速", "A_1", "温度", "date_col", "time_hms",
    "timestamp_ymd", "v", "w", "c", "h", "dirty", "mix", "big",
]
_CSV_UNITS = [
    "s", "rpm", "rpm2", "-", "°C", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-",
]


def _csv_rows(n: int) -> list[list]:
    rows = []
    for i in range(n):
        mix = "1e400" if i % 4 == 0 else ("-1e400" if i % 4 == 1 else f"{i}.5")
        rows.append([
            f"{i * 0.1:.1f}",                          # t → float32
            f"{10.0 + i * 0.5:.2f}",                   # 转速（重复名→转速_1）
            f"{20.0 + i * 0.5:.2f}",                   # 转速
            f"{i + 0.5:.1f}",                          # A_1（字面撞名）
            f"{36.5 + (i % 5) * 0.1:.2f}",             # 温度（°C）
            f"2024-01-{(i % 28) + 1:02d}",             # date_col %Y-%m-%d
            f"{(i // 3600) % 24:02d}:{(i // 60) % 60:02d}:{i % 60:02d}",  # time_hms %H:%M:%S
            f"2024/{(i % 12) + 1:02d}/{(i % 28) + 1:02d}",                # timestamp_ymd %Y/%m/%d
            "",                                        # v 全空列
            f"{i * 1.0:.1f}",                          # w 数值列
            "7",                                       # c 常量列
            f"h{i}",                                   # h 高基数文本
            _DIRTY_TOKENS[i % len(_DIRTY_TOKENS)],     # dirty 脏值混数值
            mix,                                       # mix 溢出混合
            f"{1e39 + i:.5e}",                         # big float64（>3.4e38 不降档）
        ])
    return rows


@pytest.fixture(scope="module")
def big_csv(tmp_path_factory):
    return write_csv(
        tmp_path_factory.mktemp("equiv_csv") / "big.csv",
        header=_CSV_HEADER,
        units=_CSV_UNITS,
        rows=_csv_rows(N_ROWS),
    )


@pytest.fixture(scope="module")
def mem_csv(big_csv):
    return FastDataLoader(str(big_csv), has_unit=True, sep=",")


@pytest.fixture(scope="module")
def lazy_csv(big_csv, tmp_path_factory, lazy_parquet_factory_module):
    return lazy_parquet_factory_module(big_csv, has_unit=True, sep=",")


@pytest.fixture(scope="module")
def lazy_parquet_factory_module():
    """module 级共享工厂（转换 205 行 × 15 列一次够全模块用）。"""
    from src.data.parquet_converter import convert_to_parquet
    from src.data.parquet_lazy_loader import ParquetLazyLoader
    from src.data.temp_cache_dir import TempCacheDir

    loaders: list = []

    def make(src_path, **conv_kwargs):
        temp = TempCacheDir.create()
        convert_to_parquet(str(src_path), outdir=temp.path(), **conv_kwargs)
        loader = ParquetLazyLoader(str(src_path), temp)
        loaders.append(loader)
        return loader

    yield make
    for ld in loaders:
        ld.close()


def _assert_values_equal(a, b, *, ctx: str):
    """逐位相等；float 按 equal_nan，object/datetime 按 NA 位对齐 + 非 NA 逐元素。

    np.array_equal(..., equal_nan=True) 对 object 数组会抛
    "ufunc 'isnan' not supported"（numpy 2.4 实测），必须分流。
    """
    a = np.asarray(a)
    b = np.asarray(b)
    assert a.shape == b.shape, f"{ctx}: 形状不一致 {a.shape} vs {b.shape}"
    if a.dtype.kind == "f" and b.dtype.kind == "f":
        assert np.array_equal(a, b, equal_nan=True), f"{ctx}: 值不相等"
        return
    import pandas as pd

    a_na, b_na = pd.isna(a), pd.isna(b)
    assert (a_na == b_na).all(), f"{ctx}: NA 位不一致"
    va, vb = a[~a_na], b[~b_na]
    if va.size and va.dtype.kind == "M":
        assert (va == vb).all(), f"{ctx}: datetime 值不相等"
    else:
        assert list(va) == list(vb), f"{ctx}: 值不相等"


def _assert_series_equal(mem_s, lazy_s):
    name = mem_s.name
    assert str(mem_s.dtype) == str(lazy_s.dtype), (
        f"{name}: dtype {mem_s.dtype} != {lazy_s.dtype}"
    )
    _assert_values_equal(mem_s.to_numpy(), lazy_s.to_numpy(), ctx=name)


# ==========================================================================
# DoD 1：CSV 等价性
# ==========================================================================

class TestCsvEquivalence:

    def test_var_names_unique_rule(self, mem_csv, lazy_csv):
        # 重复列名占位规则：转速/转速_1，字面 A_1 保号
        expect = ["t", "转速", "转速_1", "A_1", "温度", "date_col", "time_hms",
                  "timestamp_ymd", "v", "w", "c", "h", "dirty", "mix", "big"]
        assert mem_csv.var_names == expect
        assert lazy_csv.var_names == mem_csv.var_names

    def test_metadata_bitwise(self, mem_csv, lazy_csv):
        assert lazy_csv.units == mem_csv.units
        # 通用等价夹具刻意不含低基数文本列（枚举列的有意差异在 DoD 3 单测）
        assert lazy_csv.df_validity == mem_csv.df_validity
        assert lazy_csv.datalength == mem_csv.datalength == N_ROWS
        assert lazy_csv.max_row_count == mem_csv.max_row_count
        assert lazy_csv.row_count == mem_csv.row_count
        assert lazy_csv.column_count == mem_csv.column_count
        assert lazy_csv.global_time_range == mem_csv.global_time_range
        assert lazy_csv.time_channels_info == mem_csv.time_channels_info
        assert lazy_csv.time_column_name == mem_csv.time_column_name is None

    def test_date_formats_three_fmts(self, mem_csv, lazy_csv):
        dfmts = mem_csv.time_channels_info
        assert dfmts == {
            "date_col": "%Y-%m-%d",
            "time_hms": "%H:%M:%S",
            "timestamp_ymd": "%Y/%m/%d",
        }
        assert lazy_csv.time_channels_info == dfmts

    def test_get_series_all_columns(self, mem_csv, lazy_csv):
        for name in mem_csv.var_names:
            _assert_series_equal(mem_csv.get_series(name), lazy_csv.get_series(name))

    def test_date_columns_string_dtype(self, mem_csv, lazy_csv):
        # D9：do_parse_date=False → 两侧都是字符串（pandas 自动推断 StringDtype）
        for name in ("date_col", "time_hms", "timestamp_ymd"):
            assert str(mem_csv.get_series(name).dtype) == str(
                lazy_csv.get_series(name).dtype
            )

    def test_validity_matrix(self, mem_csv, lazy_csv):
        v = lazy_csv.df_validity
        assert v["t"] == 1          # 变化数值
        assert v["c"] == 0          # 常量列
        assert v["v"] == -1         # 全空列
        assert v["h"] == -1         # 高基数文本（to_numeric raise）
        assert v["date_col"] == 1   # 日期列（_postprocess 跳过规则）
        assert v["mix"] == 1        # 溢出清理后仍非常量
        assert v["dirty"] == 1

    def test_get_value_from_name_all_columns(self, mem_csv, lazy_csv):
        for name in mem_csv.var_names:
            mx, my, mu, m_map = mem_csv.get_value_from_name(name)
            lx, ly, lu, l_map = lazy_csv.get_value_from_name(name)
            np.testing.assert_array_equal(mx, lx)
            assert mu == lu
            assert l_map == {} == m_map
            ma = my.to_numpy() if hasattr(my, "to_numpy") else np.asarray(my)
            la = ly.to_numpy() if hasattr(ly, "to_numpy") else np.asarray(ly)
            _assert_values_equal(ma, la, ctx=name)

    def test_overflow_literals_cleaned(self, mem_csv, lazy_csv):
        # "1e400"/"-1e400"：pandas→NaN、polars→inf→清理为 null；两侧都 NaN
        a = mem_csv.get_series("mix").to_numpy()
        b = lazy_csv.get_series("mix").to_numpy()
        assert not np.isinf(a).any() and not np.isinf(b).any()
        _assert_values_equal(a.astype(np.float64), b.astype(np.float64), ctx="mix")

    def test_dirty_tokens_are_na(self, mem_csv, lazy_csv):
        a = mem_csv.get_series("dirty").to_numpy()
        b = lazy_csv.get_series("dirty").to_numpy()
        # 脏 token 两侧逐位判 NA（值级相等已在 get_series 全列断言覆盖）
        assert (np.isnan(a) == np.isnan(b)).all()
        assert np.isnan(a).any() and np.isfinite(a).any()


def test_csv_empty_header_cell(tmp_path, lazy_parquet_factory):
    """空表头单元格（历史崩溃点 header-empty-cell-pandas3-crash）。

    main 上的既有行为是**双侧同崩**（pandas 3.0 astype(str) 保留 NaN 列名，
    _infer_schema 的 col.lower() 抛 AttributeError）。修复后两侧都按
    pandas 惯例归一为 "Unnamed: {i}"，等价面 = 归一结果逐位一致。
    """
    p = write_csv(
        tmp_path / "eh.csv",
        header=["A", "", "B", ""],
        units=["-", "-", "-", "-"],
        rows=[[f"{i}.5", f"{i}.0", f"{i * 2}.5", ""] for i in range(6)],
    )
    mem = FastDataLoader(str(p), has_unit=True, sep=",")
    lazy = lazy_parquet_factory(p, has_unit=True, sep=",")
    assert mem.var_names == ["A", "Unnamed: 1", "B", "Unnamed: 3"]
    assert lazy.var_names == mem.var_names
    for name in mem.var_names:
        _assert_series_equal(mem.get_series(name), lazy.get_series(name))


def test_csv_ragged_row_rejected(tmp_path):
    """字段过多的行：pandas on_bad_lines=skip 静默跳过 vs 转换器抛错回退
    （S2 负向护栏的端到端形态：宁可拒绝交付，不可静默错位）。"""
    p = tmp_path / "ragged.csv"
    p.write_text(
        "x,y\n1.0,2.0\n3.0,4.0,9.9\n5.0,6.0\n", encoding="utf-8"
    )
    mem = FastDataLoader(str(p), has_unit=False, sep=",")
    assert mem.datalength == 2  # pandas skip 了坏行
    from src.data.parquet_converter import convert_to_parquet
    from src.data.temp_cache_dir import TempCacheDir

    temp = TempCacheDir.create()
    try:
        with pytest.raises(ParquetConversionError):
            convert_to_parquet(str(p), has_unit=False, sep=",", outdir=temp.path())
    finally:
        temp.cleanup()


# ==========================================================================
# DoD 2：Excel 等价性
# ==========================================================================

_XLSX_HEADER = ["num", "intc", "boolc", "text_hc", "const", "emptyv", "emptyw",
                "date_col", "dt_col"]
_XLSX_UNITS = ["-", "-", "-", "-", "-", "-", "-", "-", "-"]


def _xlsx_rows(n: int) -> list[list]:
    rows = []
    for i in range(n):
        rows.append([
            i + 0.5,                                    # float 链路（钉小数）
            i,                                          # int64
            i % 2 == 0,                                 # bool
            f"hc{i}",                                   # 高基数文本（>200）
            3.5,                                        # 常量 float
            None,                                       # 全空列
            float(i),                                   # 数值列
            dt.date(2024, 1, (i % 28) + 1),             # 日期 → datetime64
            dt.datetime(2024, 1, 1, 10, i % 60, i % 60),
        ])
    return rows


@pytest.fixture(scope="module")
def big_xlsx(tmp_path_factory):
    return write_xlsx(
        tmp_path_factory.mktemp("equiv_xl") / "big.xlsx",
        header=_XLSX_HEADER,
        units=_XLSX_UNITS,
        rows=_xlsx_rows(N_ROWS),
    )


@pytest.fixture(scope="module")
def mem_xlsx(big_xlsx):
    return ExcelDataLoader(str(big_xlsx), sheet_name="Sheet1",
                           desc_rows=0, has_unit=True)


@pytest.fixture(scope="module")
def lazy_xlsx(big_xlsx, lazy_parquet_factory_module):
    return lazy_parquet_factory_module(
        big_xlsx, is_excel=True, sheet_name="Sheet1", desc_rows=0, has_unit=True
    )


class TestExcelEquivalence:

    def test_metadata_bitwise(self, mem_xlsx, lazy_xlsx):
        assert lazy_xlsx.var_names == mem_xlsx.var_names
        assert lazy_xlsx.units == mem_xlsx.units
        # 高基数文本列两侧同判（非枚举）；emptyv 全空 -1；const 常量 0
        assert lazy_xlsx.df_validity == mem_xlsx.df_validity
        assert lazy_xlsx.datalength == mem_xlsx.datalength == N_ROWS
        assert lazy_xlsx.column_count == mem_xlsx.column_count
        assert lazy_xlsx.time_channels_info == mem_xlsx.time_channels_info

    def test_get_series_all_columns(self, mem_xlsx, lazy_xlsx):
        for name in mem_xlsx.var_names:
            _assert_series_equal(mem_xlsx.get_series(name), lazy_xlsx.get_series(name))

    def test_datetime_roundtrip_dtype(self, mem_xlsx, lazy_xlsx):
        # D9：datetime64[s]/[us] 按原分辨率往返（meta dtype_str 还原）
        for name in ("date_col", "dt_col"):
            assert str(mem_xlsx.get_series(name).dtype) == str(
                lazy_xlsx.get_series(name).dtype
            )

    def test_get_value_from_name_all_columns(self, mem_xlsx, lazy_xlsx):
        for name in mem_xlsx.var_names:
            mx, my, mu, m_map = mem_xlsx.get_value_from_name(name)
            lx, ly, lu, l_map = lazy_xlsx.get_value_from_name(name)
            np.testing.assert_array_equal(mx, lx)
            assert mu == lu
            assert l_map == {} == m_map
            ma = my.to_numpy() if hasattr(my, "to_numpy") else np.asarray(my)
            la = ly.to_numpy() if hasattr(ly, "to_numpy") else np.asarray(ly)
            _assert_values_equal(ma, la, ctx=name)


# ==========================================================================
# DoD 3：枚举列有意差异（D8）—— 单独断言，不与通用等价混用
# ==========================================================================

def test_enum_intentional_diff_csv(tmp_path, lazy_parquet_factory):
    """CSV 低基数文本列（n_unique<=200 → 枚举码值化）。

    三条有意差异：
    1. validity：内存 -1（to_numeric raise） vs 新 loader 1（低基数变可选）
    2. get_series：两边都返回文本 category（值级相等；categories 顺序
       允许不同——内存侧排序序、新侧出现序）
    3. get_value_from_name：新 loader 返回码值 ndarray + 非空 text_map
    """
    # 数据区用「汉」类字（gb18030 baba 是非法 UTF-8，避开「状态」恰好
    # 合法 UTF-8 直读不转码的坑）
    p = write_csv(
        tmp_path / "enum.csv",
        header=["状态汉", "v"],
        units=["-", "-"],
        rows=[[f"汉{i % 3}", f"{i}.5"] for i in range(12)],
    )
    mem = FastDataLoader(str(p), has_unit=True, sep=",")
    lazy = lazy_parquet_factory(p, has_unit=True, sep=",")

    # 1) validity 有意差异
    assert mem.df_validity["状态汉"] == -1
    assert lazy.df_validity["状态汉"] == 1

    # 2) get_series 两边都是文本 category，值级相等
    ms, ls = mem.get_series("状态汉"), lazy.get_series("状态汉")
    assert str(ms.dtype) == "category"
    assert str(ls.dtype) == "category"
    assert ms.tolist() == ls.tolist() == [f"汉{i % 3}" for i in range(12)]
    # v 数值列不受影响（差异只发生在枚举列上）
    _assert_series_equal(mem.get_series("v"), lazy.get_series("v"))

    # 3) get_value_from_name：码值 + text_map vs 文本 + {}
    _, my, _, m_map = mem.get_value_from_name("状态汉")
    x, y, unit, text_map = lazy.get_value_from_name("状态汉")
    assert m_map == {}
    assert text_map == {0: "汉0", 1: "汉1", 2: "汉2"}
    assert isinstance(y, np.ndarray)
    np.testing.assert_array_equal(y[:3], [0, 1, 2])
    np.testing.assert_array_equal(x, np.arange(1, 13, dtype=np.float64))
    # 内存侧 y 是文本（可画性来自新 loader 的码值，这正是 D8 的意义）
    assert set(my.to_numpy().tolist()) == {"汉0", "汉1", "汉2"}


def test_enum_intentional_diff_excel(tmp_path, lazy_parquet_factory):
    """Excel 低基数文本列同规则（「文本列」→ 枚举码值化）。"""
    p = write_xlsx(
        tmp_path / "enum.xlsx",
        header=["mode", "v"],
        units=["-", "-"],
        rows=[[f"m{i % 3}", i + 0.5] for i in range(12)],
    )
    mem = ExcelDataLoader(str(p), sheet_name="Sheet1", desc_rows=0, has_unit=True)
    lazy = lazy_parquet_factory(
        p, is_excel=True, sheet_name="Sheet1", desc_rows=0, has_unit=True
    )

    assert mem.df_validity["mode"] == -1
    assert lazy.df_validity["mode"] == 1

    ms, ls = mem.get_series("mode"), lazy.get_series("mode")
    # 值级相等：两边都是文本（内存 str dtype、新侧 category——差异仅在
    # dtype 包装，值语义一致；文案口径见 test_var_info_parquet）
    assert ms.tolist() == ls.tolist() == [f"m{i % 3}" for i in range(12)]

    _, y, _, text_map = lazy.get_value_from_name("mode")
    assert text_map == {0: "m0", 1: "m1", 2: "m2"}
    np.testing.assert_array_equal(y[:3], [0, 1, 2])


# ==========================================================================
# 现场宽表形态：行尾空表头 + 私有区字符列名 + ** 缺测值
# ==========================================================================

@pytest.fixture(scope="module")
def wide_csv(tmp_path_factory):
    from tests.fixtures.data_factory import write_field_like_wide_csv

    return write_field_like_wide_csv(
        tmp_path_factory.mktemp("equiv_wide") / "wide.txt",
        pua_column_name=True,
        star_nulls=True,
    )


@pytest.fixture(scope="module")
def mem_wide(wide_csv):
    return FastDataLoader(str(wide_csv), has_unit=True, sep="\t")


@pytest.fixture(scope="module")
def lazy_wide(wide_csv, lazy_parquet_factory_module):
    return lazy_parquet_factory_module(wide_csv, has_unit=True, sep="\t")


class TestFieldLikeWideCsv:

    def test_var_names_normalized_bitwise(self, mem_wide, lazy_wide):
        # 行尾空表头 → 末列归一 "Unnamed: 5"；PUA 列名原样保留；全部 str
        expect = ["ENG01_CH00", "ENG01_CH01", "ENG01_CH02", "ENG01_CH03",
                  "\ue71aPUA_NAME", "Unnamed: 5"]
        assert mem_wide.var_names == expect
        assert lazy_wide.var_names == mem_wide.var_names
        assert all(isinstance(n, str) for n in mem_wide.var_names)

    def test_metadata_bitwise(self, mem_wide, lazy_wide):
        assert lazy_wide.units == mem_wide.units
        # PUA 列是低基数文本 → D8 枚举有意差异，单列断言见 test_pua_column_enum_path
        pua = "\ue71aPUA_NAME"
        assert {
            k: v for k, v in lazy_wide.df_validity.items() if k != pua
        } == {
            k: v for k, v in mem_wide.df_validity.items() if k != pua
        }
        assert lazy_wide.datalength == mem_wide.datalength == 40
        assert lazy_wide.column_count == mem_wide.column_count == 6

    def test_get_series_all_columns(self, mem_wide, lazy_wide):
        for name in mem_wide.var_names:
            _assert_series_equal(mem_wide.get_series(name), lazy_wide.get_series(name))

    def test_star_nulls_nan_both_sides(self, mem_wide, lazy_wide):
        a = mem_wide.get_series("ENG01_CH01").to_numpy()
        b = lazy_wide.get_series("ENG01_CH01").to_numpy()
        assert (np.isnan(a) == np.isnan(b)).all()
        assert np.isnan(a).any() and np.isfinite(a).any()

    def test_pua_column_enum_path(self, mem_wide, lazy_wide):
        # PUA 列是低基数文本（on/off）→ 枚举码值化（D8），两侧有效性有意差异
        assert mem_wide.df_validity["\ue71aPUA_NAME"] == -1
        assert lazy_wide.df_validity["\ue71aPUA_NAME"] == 1
        ms = mem_wide.get_series("\ue71aPUA_NAME")
        ls = lazy_wide.get_series("\ue71aPUA_NAME")
        assert ms.tolist() == ls.tolist() == ["on" if i % 2 else "off" for i in range(40)]
