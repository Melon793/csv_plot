"""BaseDataLoader 单元测试：列名去重、validity 判定、inf 清理、属性接口。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.base_loader import BaseDataLoader


def _make_loader(df: pd.DataFrame) -> BaseDataLoader:
    loader = BaseDataLoader()
    loader._df = df
    return loader


class TestMakeUnique:
    def test_duplicates_get_suffix(self):
        assert BaseDataLoader._make_unique(["a", "a", "b", "a"]) == [
            "a", "a_1", "b", "a_2",
        ]

    def test_no_duplicates_unchanged(self):
        assert BaseDataLoader._make_unique(["x", "y", "z"]) == ["x", "y", "z"]


class TestPostprocessValidity:
    def test_float_varying_is_valid(self):
        loader = _make_loader(pd.DataFrame({"v": [1.0, 2.0, 3.0]}))
        assert loader._postprocess_columns() == {"v": 1}

    def test_float_constant_is_const(self):
        loader = _make_loader(pd.DataFrame({"v": [5.0, 5.0, 5.0]}))
        assert loader._postprocess_columns() == {"v": 0}

    def test_all_nan_is_invalid(self):
        loader = _make_loader(pd.DataFrame({"v": [np.nan, np.nan]}))
        assert loader._postprocess_columns() == {"v": -1}

    def test_int_constant_is_const(self):
        loader = _make_loader(pd.DataFrame({"v": [7, 7, 7]}))
        assert loader._postprocess_columns() == {"v": 0}

    def test_int_varying_is_valid(self):
        loader = _make_loader(pd.DataFrame({"v": [1, 1, 2]}))
        assert loader._postprocess_columns() == {"v": 1}

    def test_non_numeric_string_is_invalid(self):
        loader = _make_loader(pd.DataFrame({"v": ["abc", "def"]}))
        assert loader._postprocess_columns() == {"v": -1}

    def test_date_format_column_always_valid(self):
        loader = _make_loader(pd.DataFrame({"time": [1.0, 1.0]}))
        loader.date_formats = {"time": "%H:%M:%S"}
        assert loader._postprocess_columns() == {"time": 1}


class TestInfCleanup:
    def test_inf_replaced_by_nan_and_downcast(self):
        arr = np.array([1.0, np.inf, 3.0, -np.inf])
        loader = _make_loader(pd.DataFrame({"v": arr}))
        loader._postprocess_columns(downcast=True)
        result = loader._df["v"]
        assert result.dtype == np.float32
        assert np.isnan(result.iloc[1])
        assert np.isnan(result.iloc[3])
        assert result.iloc[0] == np.float32(1.0)

    def test_no_downcast_keeps_float64_dtype(self):
        arr = np.array([1.0, np.inf, 3.0])
        loader = _make_loader(pd.DataFrame({"v": arr}))
        validity = loader._postprocess_columns(downcast=False)
        # downcast=False 时不清理 inf、不转换 dtype
        assert loader._df["v"].dtype == np.float64
        assert validity["v"] == 1


class TestProperties:
    def _loaded(self) -> BaseDataLoader:
        loader = _make_loader(
            pd.DataFrame({"time": [0.0, 0.1], "a": [1.0, 2.0]})
        )
        loader.time_column_name = "time"
        loader._units = {"time": "s", "a": "V"}
        loader._df_validity = {"time": 1, "a": 1}
        return loader

    def test_var_names_excludes_time_column(self):
        assert self._loaded().var_names == ["a"]

    def test_df_validity_excludes_time_column(self):
        assert self._loaded().df_validity == {"a": 1}

    def test_time_axis_label_with_unit(self):
        assert self._loaded().time_axis_label == "time (s)"

    def test_time_axis_label_without_time(self):
        loader = _make_loader(pd.DataFrame({"a": [1.0]}))
        assert loader.time_axis_label == "Index"

    def test_time_values_falls_back_to_index(self):
        loader = _make_loader(pd.DataFrame({"a": [1.0, 2.0]}))
        values = loader.time_values
        assert values.tolist() == [1, 2]

    def test_time_values_from_time_column(self):
        assert self._loaded().time_values.tolist() == [0.0, 0.1]

    def test_release_memory_empties_dataframe(self):
        loader = self._loaded()
        loader.release_memory()
        assert loader.df.empty
