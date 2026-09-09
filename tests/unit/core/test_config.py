"""core.config 单元测试：float32 安全检测、全局 X limits、安全回调。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.core.config import (
    DEFAULT_PADDING_VAL_X,
    FLOAT32_REPRESENTABLE_MAX,
    _evaluate_float32_safety,
    compute_global_x_limits,
    safe_callback,
    safe_qt_op,
)


class TestEvaluateFloat32Safety:
    def test_none_is_unsafe(self):
        assert _evaluate_float32_safety(None) == (False, None)

    def test_empty_is_safe(self):
        assert _evaluate_float32_safety([]) == (True, 0.0)

    def test_normal_values_safe(self):
        is_safe, abs_max = _evaluate_float32_safety([1.0, -5.0, 3.0])
        assert is_safe is True
        assert abs_max == pytest.approx(5.0)

    def test_huge_values_unsafe(self):
        values = [1.0, FLOAT32_REPRESENTABLE_MAX * 10]
        is_safe, abs_max = _evaluate_float32_safety(values)
        assert is_safe is False
        assert abs_max == pytest.approx(FLOAT32_REPRESENTABLE_MAX * 10)

    def test_all_nan_unsafe(self):
        is_safe, abs_max = _evaluate_float32_safety([np.nan, np.nan])
        assert is_safe is False
        assert abs_max is None

    def test_pandas_series_input(self):
        is_safe, abs_max = _evaluate_float32_safety(pd.Series([2.0, -8.0]))
        assert is_safe is True
        assert abs_max == pytest.approx(8.0)

    def test_inf_ignored_in_abs_max(self):
        is_safe, abs_max = _evaluate_float32_safety([np.inf, 3.0])
        assert is_safe is True
        assert abs_max == pytest.approx(3.0)


class _CsvLoaderStub:
    LOADER_TYPE = "csv"

    def __init__(self, datalength: int):
        self.datalength = datalength


class _MdfLoaderStub:
    LOADER_TYPE = "mdf"

    def __init__(self, time_range: tuple[float, float]):
        self.global_time_range = time_range


class TestComputeGlobalXLimits:
    def test_none_loader_returns_none(self):
        assert compute_global_x_limits(None) is None

    def test_csv_loader_range(self):
        result = compute_global_x_limits(_CsvLoaderStub(100))
        min_x, max_x, lim_min, lim_max = result
        assert min_x == pytest.approx(1.0)
        assert max_x == pytest.approx(100.0)
        span = max_x - min_x
        assert lim_min == pytest.approx(min_x - DEFAULT_PADDING_VAL_X * span)
        assert lim_max == pytest.approx(max_x + DEFAULT_PADDING_VAL_X * span)

    def test_factor_and_offset_applied(self):
        min_x, max_x, _, _ = compute_global_x_limits(
            _CsvLoaderStub(10), factor=2.0, offset=5.0
        )
        assert min_x == pytest.approx(5.0 + 2.0 * 1.0)
        assert max_x == pytest.approx(5.0 + 2.0 * 10.0)

    def test_single_row_expands_range(self):
        """min == max 时自动扩展，避免零宽范围"""
        min_x, max_x, _, _ = compute_global_x_limits(_CsvLoaderStub(1))
        assert min_x < max_x

    def test_mdf_loader_uses_global_time_range(self):
        min_x, max_x, _, _ = compute_global_x_limits(_MdfLoaderStub((0.0, 60.0)))
        assert min_x == pytest.approx(0.0)
        assert max_x == pytest.approx(60.0)

    def test_empty_loader_returns_none(self):
        assert compute_global_x_limits(_CsvLoaderStub(0)) is None


class TestSafeQtOp:
    def test_normal_return_value(self):
        assert safe_qt_op(lambda: 42) == 42

    def test_deleted_cpp_object_swallowed(self):
        def raise_deleted():
            raise RuntimeError("Internal C++ object already deleted")

        assert safe_qt_op(raise_deleted) is None

    def test_other_runtime_error_propagates(self):
        def raise_other():
            raise RuntimeError("some other error")

        with pytest.raises(RuntimeError):
            safe_qt_op(raise_other)

    def test_attribute_error_swallowed(self):
        def raise_attr():
            raise AttributeError("no such attr")

        assert safe_qt_op(raise_attr) is None


class TestSafeCallback:
    def test_normal_return_value(self):
        @safe_callback
        def fn(x):
            return x + 1

        assert fn(1) == 2

    def test_deleted_cpp_object_returns_none(self):
        @safe_callback
        def fn():
            raise RuntimeError("wrapped C/C++ object has been deleted")

        assert fn() is None

    def test_generic_exception_suppressed(self):
        @safe_callback
        def fn():
            raise ValueError("boom")

        assert fn() is None
