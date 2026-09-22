"""ParquetLazyLoader 单元测试（P4，§4.4 契约的逐条验证）。

等价性（与 FastDataLoader/ExcelDataLoader 的逐位对比）在
test_parquet_equivalence.py；文案/统计等价在 test_var_info_parquet.py。
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from src.data.parquet_lazy_loader import ParquetLazyLoader
from src.data.temp_cache_dir import TempCacheDir
from tests.fixtures.data_factory import write_csv


@pytest.fixture()
def tiny_csv(tmp_path):
    return write_csv(
        tmp_path / "tiny.csv",
        header=["x", "y", "lbl"],
        units=["-", "m", "-"],
        rows=[[f"{i}.5", f"{i * 2}.0", f"L{i % 3}"] for i in range(8)],
    )


@pytest.fixture()
def lazy(tmp_path, tiny_csv, lazy_parquet_factory):
    return lazy_parquet_factory(tiny_csv, has_unit=True, sep=",")


# --------------------------------------------------------------------------
# 契约属性
# --------------------------------------------------------------------------

class TestContractProperties:

    def test_identity_and_lazy_flags(self, lazy):
        assert lazy.LOADER_TYPE == "parquet"
        assert lazy.IS_LAZY is True
        assert lazy.df is None  # 硬性：数据不在内存

    def test_shape_properties_from_meta(self, lazy):
        # 全部来自 meta.json，不读 parquet（datalength=8、列数=3）
        assert lazy.datalength == 8
        assert lazy.max_row_count == 8
        assert lazy.row_count == 8
        assert lazy.column_count == 3
        assert lazy.var_names == ["x", "y", "lbl"]

    def test_time_semantics_csv(self, lazy):
        assert lazy.time_column_name is None
        assert lazy.time_axis_label == "Index"
        assert lazy.global_time_range == (1.0, 8.0)
        assert lazy.baseline_density == 1.0
        tv = lazy.time_values
        assert tv.name == "index"
        np.testing.assert_array_equal(tv.to_numpy(), np.arange(1, 9))

    def test_units_and_validity(self, lazy):
        assert lazy.units == {"x": "-", "y": "m", "lbl": "-"}
        # lbl 是低基数文本 → 枚举列 validity=1（D8 有意正向差异）
        assert lazy.df_validity == {"x": 1, "y": 1, "lbl": 1}

    def test_path_points_to_source(self, lazy, tiny_csv):
        assert lazy.path == str(tiny_csv)
        assert lazy.file_size == Path(tiny_csv).stat().st_size


# --------------------------------------------------------------------------
# 访问与缓存
# --------------------------------------------------------------------------

class TestAccess:

    def test_get_series_dtypes(self, lazy):
        sx = lazy.get_series("x")
        assert str(sx.dtype) == "float32"
        assert sx.name == "x"
        # 枚举列还原文本（不是码值）
        sl = lazy.get_series("lbl")
        assert str(sl.dtype) == "category"
        assert set(sl.cat.categories) == {"L0", "L1", "L2"}
        assert sl.tolist() == [f"L{i % 3}" for i in range(8)]

    def test_get_series_missing_raises(self, lazy):
        with pytest.raises(KeyError):
            lazy.get_series("不存在")

    def test_get_value_from_name_non_enum(self, lazy):
        x, y, unit, text_map = lazy.get_value_from_name("y")
        np.testing.assert_array_equal(x, np.arange(1, 9, dtype=np.float64))
        assert str(y.dtype) == "float32"
        assert unit == "m"
        assert text_map == {}

    def test_get_value_from_name_enum_codes_and_map(self, lazy):
        x, y, unit, text_map = lazy.get_value_from_name("lbl")
        np.testing.assert_array_equal(x, np.arange(1, 9, dtype=np.float64))
        assert text_map == {0: "L0", 1: "L1", 2: "L2"}
        # y 是码值序列（出现序编码）
        y_list = list(y)
        # L0/L1/L2 出现序：L0(第0行)、L1(第1行)、L2(第2行)
        assert y_list[:3] == [0, 1, 2]
        assert all(v in (0, 1, 2) for v in y_list)
        assert unit == "-"

    def test_meta_returns_copy(self, lazy):
        m = lazy.meta("x")
        assert m["dtype_str"] == "float32"
        m["dtype_str"] = "被污染"
        assert lazy.meta("x")["dtype_str"] == "float32"

    def test_meta_missing_raises(self, lazy):
        with pytest.raises(KeyError):
            lazy.meta("不存在")

    def test_column_cache_hit(self, lazy, monkeypatch):
        """第二次 get_series 命中缓存，不再读 parquet。"""
        import polars as pl

        calls = []
        orig = pl.read_parquet

        def counting(path, *args, **kwargs):
            calls.append(path)
            return orig(path, *args, **kwargs)

        monkeypatch.setattr(pl, "read_parquet", counting)
        lazy.get_series("x")
        lazy.get_series("x")
        lazy.get_value_from_name("x")
        assert len(calls) == 1

    def test_threaded_get_series(self, lazy):
        """并发 get_series 冒烟：RLock 串行化下全部成功。"""
        errors = []

        def worker(name):
            try:
                for _ in range(5):
                    s = lazy.get_series(name)
                    assert len(s) == 8
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(n,)) for n in ("x", "y", "lbl")
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors


# --------------------------------------------------------------------------
# 生命周期
# --------------------------------------------------------------------------

class TestLifecycle:

    def test_close_removes_dir_and_idempotent(self, tiny_csv, lazy_parquet_factory):
        lazy = lazy_parquet_factory(tiny_csv, has_unit=True, sep=",")
        temp_path = Path(lazy._cache_dir.path())
        assert temp_path.exists()
        lazy.close()
        assert not temp_path.exists()
        lazy.close()  # 幂等：不抛

    def test_closed_access_raises_keyerror(self, tiny_csv, lazy_parquet_factory):
        lazy = lazy_parquet_factory(tiny_csv, has_unit=True, sep=",")
        lazy.close()
        with pytest.raises(KeyError):
            lazy.get_series("x")
        with pytest.raises(KeyError):
            lazy.get_value_from_name("x")
        with pytest.raises(KeyError):
            lazy.meta("x")

    def test_release_memory_keeps_dir_and_rereads(self, lazy):
        lazy.get_series("x")
        # 一列两种形态入缓存：polars 原始（"r:"）+ pandas 物化（"s:"）
        assert len(lazy._cache) == 2
        lazy.release_memory()
        assert len(lazy._cache) == 0
        # 目录还在，数据可重读
        s = lazy.get_series("x")
        assert len(s) == 8
        assert Path(lazy._cache_dir.path()).exists()

    def test_missing_artifact_raises(self, tmp_path):
        """目录里没有转换产物 → 构造失败（防呆）。"""
        temp = TempCacheDir.create()
        try:
            with pytest.raises(FileNotFoundError):
                ParquetLazyLoader("/nonexistent/src.csv", temp)
        finally:
            temp.cleanup()

    def test_bad_meta_raises(self, tmp_path):
        """meta.json 缺关键字段 → ValueError。"""
        import json

        src = tmp_path / "s.csv"
        src.write_text("x\n1\n", encoding="utf-8")
        temp = TempCacheDir.create()
        try:
            (Path(temp.path()) / "meta.json").write_text(
                json.dumps({"rows": 1}), encoding="utf-8"
            )
            (Path(temp.path()) / "data.parquet").write_bytes(b"")
            with pytest.raises(ValueError):
                ParquetLazyLoader(str(src), temp)
        finally:
            temp.cleanup()
