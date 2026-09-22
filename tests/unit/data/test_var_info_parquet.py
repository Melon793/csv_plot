"""_from_parquet 文案等价（D15）与 compute_stats 统计等价（C2）测试。

P4 DoD 4/5 的唯一护栏：
- DoD 4：_from_parquet 与 _from_tabular 的 snapshot 逐字段相等（除身份
  字段 source_kind），夹具覆盖 float32/float64、全空列、category 文本列
  （高基数）、日期列、常量列。
- DoD 5：compute_stats 对同一列在两种 loader 下返回的 VarStats 逐字段
  相等；枚举列两边都返回「非数值类型」错误（不得变成对码值算出的数字）。
"""

from __future__ import annotations

import pytest

from src.data.loader import FastDataLoader
from src.data.var_info import VarStats, build_snapshot, compute_stats
from tests.fixtures.data_factory import write_csv

N_ROWS = 205  # 高基数列需要 > 200 unique（ENUM_LABEL_MAX=200）

_HEADER = ["t", "big", "v", "w", "c", "h", "date_col", "dirty"]
_UNITS = ["s", "-", "-", "-", "-", "-", "-", "-"]


def _rows(n: int) -> list[list]:
    rows = []
    for i in range(n):
        rows.append([
            f"{i * 0.1:.1f}",          # t → float32
            f"{1e39 + i:.5e}",         # big → float64（>3.4e38 不降档）
            "",                        # v 全空列
            f"{i * 1.0:.1f}",          # w 数值
            "7",                       # c 常量列
            f"h{i}",                   # h 高基数文本（205 unique）
            f"2024-01-{(i % 28) + 1:02d}",  # date_col %Y-%m-%d
            "NULL" if i % 3 == 0 else f"{i}.5",  # dirty 含 NA 的 float
        ])
    return rows


@pytest.fixture(scope="module")
def snap_csv(tmp_path_factory):
    return write_csv(
        tmp_path_factory.mktemp("vi_parquet") / "snap.csv",
        header=_HEADER,
        units=_UNITS,
        rows=_rows(N_ROWS),
    )


@pytest.fixture(scope="module")
def mem(snap_csv):
    return FastDataLoader(str(snap_csv), has_unit=True, sep=",")


@pytest.fixture(scope="module")
def lazy(snap_csv):
    from src.data.parquet_converter import convert_to_parquet
    from src.data.parquet_lazy_loader import ParquetLazyLoader
    from src.data.temp_cache_dir import TempCacheDir

    temp = TempCacheDir.create()
    convert_to_parquet(str(snap_csv), outdir=temp.path(), has_unit=True, sep=",")
    loader = ParquetLazyLoader(str(snap_csv), temp)
    yield loader
    loader.close()


_SNAPSHOT_FIELDS = (
    "name", "original_name", "dtype", "length", "unit", "validity",
    "is_numeric", "is_enum", "all_empty", "generation",
)


class TestSnapshotTextEquivalence:
    """DoD 4：文案等价（D15 的唯一护栏）。"""

    @pytest.mark.parametrize("col", ["t", "big", "v", "h", "date_col", "c"])
    def test_snapshot_fields_equal(self, mem, lazy, col):
        s_mem = build_snapshot(mem, col, generation=7)
        s_lazy = build_snapshot(lazy, col, generation=7)
        # 身份字段：分派正确（唯一的合法差异）
        assert s_mem.source_kind == "csv"  # FastDataLoader.LOADER_TYPE
        assert s_lazy.source_kind == "parquet"
        for f in _SNAPSHOT_FIELDS:
            got_mem, got_lazy = getattr(s_mem, f), getattr(s_lazy, f)
            assert got_lazy == got_mem, f"{col}.{f}: {got_lazy!r} != {got_mem!r}"
        assert s_lazy.sections == s_mem.sections

    def test_snapshot_dtype_matrix(self, lazy):
        """dtype 文案矩阵：文本层归一 object，数值保持精度名。"""
        by_col = {c: build_snapshot(lazy, c).dtype for c in lazy.var_names}
        assert by_col["t"] == "float32"
        assert by_col["big"] == "float64"
        assert by_col["v"] == "object"        # 全空列（category 0 cats 归一）
        assert by_col["h"] == "object"        # 高基数文本（categories str→O）
        assert by_col["date_col"] == "object"  # 日期字符串列（StringDtype→O）
        assert by_col["c"] == "float32"

    def test_all_empty_column_copy(self, mem, lazy):
        """全空列专属文案（复刻 _from_tabular:255-300 的分支）。"""
        s_mem = build_snapshot(mem, "v")
        s_lazy = build_snapshot(lazy, "v")
        assert s_lazy.all_empty is True and s_mem.all_empty is True
        assert s_lazy.sections["列信息"] == s_mem.sections["列信息"]
        # 说明行含行数与「不支持统计与绘图」
        desc = dict(s_lazy.sections["列信息"])["说明"]
        assert f"共 {N_ROWS} 行" in desc and "不支持统计与绘图" in desc

    def test_text_column_copy(self, mem, lazy):
        """高基数文本列 → 「非数值列」文案（不是全空列文案）。"""
        s_mem = build_snapshot(mem, "h")
        s_lazy = build_snapshot(lazy, "h")
        assert s_lazy.sections["列信息"] == s_mem.sections["列信息"]
        desc = dict(s_lazy.sections["列信息"])["说明"]
        assert desc.startswith("非数值列")

    def test_date_column_copy(self, mem, lazy):
        s_mem = build_snapshot(mem, "date_col")
        s_lazy = build_snapshot(lazy, "date_col")
        rows = dict(s_lazy.sections["列信息"])
        assert rows["是否时间格式列"] == "是"
        assert rows["时间格式"] == "%Y-%m-%d"
        assert s_lazy.sections == s_mem.sections

    def test_file_info_rows_equal(self, mem, lazy):
        s_mem = build_snapshot(mem, "t")
        s_lazy = build_snapshot(lazy, "t")
        assert s_lazy.sections["文件信息"] == s_mem.sections["文件信息"]

    def test_missing_variable_raises(self, lazy):
        with pytest.raises(KeyError):
            build_snapshot(lazy, "不存在")


_STATS_FIELDS = (
    "min", "max", "mean", "std",
    "nan_count", "inf_count", "finite_count",
    "computed", "error",
)


class TestStatsEquivalence:
    """DoD 5：统计等价（C2：单趟 numpy 口径，非 MDF 分块累加）。"""

    @pytest.mark.parametrize(
        "col", ["t", "big", "c", "dirty", "v", "h", "date_col"]
    )
    def test_stats_fields_equal(self, mem, lazy, col):
        st_mem = compute_stats(mem, col)
        st_lazy = compute_stats(lazy, col)
        for f in _STATS_FIELDS:
            got_mem, got_lazy = getattr(st_mem, f), getattr(st_lazy, f)
            assert got_lazy == got_mem, f"{col}.{f}: {got_lazy!r} != {got_mem!r}"

    def test_numeric_stats_values(self, lazy):
        """数值列给出真实统计（computed=True、四个量非空）。"""
        st = compute_stats(lazy, "t")
        assert st.computed is True
        assert st.min == 0.0 and st.max == pytest.approx(20.4)
        assert st.error == ""

    def test_dirty_nan_counts(self, mem, lazy):
        st = compute_stats(lazy, "dirty")
        n_null = sum(1 for i in range(N_ROWS) if i % 3 == 0)
        assert st.nan_count == n_null
        assert st.finite_count == N_ROWS - n_null
        assert st.inf_count == 0

    def test_all_empty_error(self, mem, lazy):
        """全空列在 CSV 链是 category(0 cats) → to_numpy 为 object 数组 →
        两侧同为「非数值类型」错误（今天的口径，「全 NaN」错误只属于
        float 全 NaN 列）。"""
        st = compute_stats(lazy, "v")
        assert st.computed is False
        assert st.error == compute_stats(mem, "v").error
        assert st.error == "非数值类型（object），不适用统计"

    def test_text_error(self, lazy):
        st = compute_stats(lazy, "h")
        assert st.computed is False
        assert "非数值类型" in st.error

    def test_date_error(self, lazy):
        st = compute_stats(lazy, "date_col")
        assert st.computed is False
        assert "非数值类型" in st.error


def test_stats_enum_both_non_numeric(tmp_path, lazy_parquet_factory):
    """DoD 5 特别条款：枚举列两边都是「非数值类型」错误。

    get_series 对枚举列必须还原文本（D8）——若返回码值，这里会对码值
    算出 min/max/mean（无意义数字被当成统计结果，比崩更糟）。
    """
    p = write_csv(
        tmp_path / "enum_stats.csv",
        header=["状态汉", "v"],
        units=["-", "-"],
        rows=[[f"汉{i % 3}", f"{i}.5"] for i in range(12)],
    )
    mem = FastDataLoader(str(p), has_unit=True, sep=",")
    lazy = lazy_parquet_factory(p, has_unit=True, sep=",")

    st_mem = compute_stats(mem, "状态汉")
    st_lazy = compute_stats(lazy, "状态汉")
    for st in (st_mem, st_lazy):
        assert st.computed is False
        assert "非数值类型" in st.error
        assert st.min is None and st.max is None and st.mean is None
    assert st_lazy.error == st_mem.error
    # 对照组：同文件数值列两侧完整统计
    sv = compute_stats(lazy, "v")
    assert sv.computed is True and sv.finite_count == 12
