"""curve_transform 投影纯函数单元测试（编辑器强化轮 C1 / D3）。"""

import numpy as np
import pytest

from src.core.curve_transform import (
    NON_NUMERIC_FORMATS,
    format_transform,
    project_y,
    reproject,
)
from src.core.data_types import CurveInfo, InvalidCurveRef


def _ci(y_raw, **kw):
    """最小 CurveInfo（curve 传 None，纯数据层测试不碰场景）。"""
    return CurveInfo(
        var_name="v", curve=None,
        x_data=np.arange(len(y_raw), dtype=float),
        y_data=np.asarray(y_raw, dtype=float),
        y_raw=np.asarray(y_raw, dtype=float),
        **kw,
    )


class TestProjectY:
    def test_default_identity_returns_same_object(self):
        raw = np.array([1.0, 2.0, 3.0])
        assert project_y(raw, 1.0, 0.0) is raw

    def test_linear_arithmetic(self):
        raw = np.array([1.0, 2.0, 3.0])
        out = project_y(raw, 2.0, -10.0)
        np.testing.assert_allclose(out, [-8.0, -6.0, -4.0])

    def test_zero_factor_allowed(self):
        raw = np.array([1.0, 2.0])
        np.testing.assert_allclose(project_y(raw, 0.0, 5.0), [5.0, 5.0])

    def test_negative_factor(self):
        raw = np.array([1.0, 2.0])
        np.testing.assert_allclose(project_y(raw, -1.0, 0.0), [-1.0, -2.0])

    @pytest.mark.parametrize("fmt", sorted(NON_NUMERIC_FORMATS))
    def test_non_numeric_formats_skip(self, fmt):
        raw = np.array([1.0, 2.0])
        assert project_y(raw, 2.0, 10.0, fmt) is raw

    def test_none_raw(self):
        assert project_y(None, 2.0, 10.0) is None

    def test_float32_dtype_preserved(self):
        raw = np.array([1.0, 2.0], dtype=np.float32)
        out = project_y(raw, 1.5, 0.5)
        assert out.dtype == np.float32
        np.testing.assert_allclose(out, [2.0, 3.5], rtol=1e-6)

    def test_float64_precision_then_cast(self):
        # 大 offset 下 float32 直接算会量化；float64 中间计算保精度
        raw = np.array([1.0], dtype=np.float32)
        out = project_y(raw, 1.0, 1e8 + 0.5)
        assert out.dtype == np.float32
        # cast 回 float32 后 ULP 量化是预期口径，断言不抛且有限
        assert np.isfinite(out).all()


class TestReproject:
    def test_idempotent_no_double_projection(self):
        ci = _ci([1.0, 2.0, 3.0], factor=2.0, offset=10.0)
        reproject(ci)
        first = ci.y_data.copy()
        reproject(ci)
        np.testing.assert_allclose(ci.y_data, first)
        np.testing.assert_allclose(ci.y_data, [12.0, 14.0, 16.0])

    def test_change_factor_reprojects_from_raw(self):
        ci = _ci([1.0, 2.0], factor=2.0, offset=0.0)
        reproject(ci)
        ci.factor = 3.0
        reproject(ci)
        np.testing.assert_allclose(ci.y_data, [3.0, 6.0])

    def test_enum_ci_untouched(self):
        ci = _ci([1.0, 2.0], factor=2.0, offset=10.0, y_format="enum")
        reproject(ci)
        np.testing.assert_allclose(ci.y_data, [1.0, 2.0])


class TestFormatTransform:
    def test_identity(self):
        assert format_transform(1.0, 0.0) == "y"

    def test_factor_only(self):
        assert format_transform(2.0, 0.0) == "2·y"

    def test_offset_only(self):
        assert format_transform(1.0, -10.0) == "y − 10"
        assert format_transform(1.0, 2.5) == "y + 2.5"

    def test_combined(self):
        assert format_transform(2.0, -10.0) == "2·y − 10"


class TestDataclassFields:
    def test_y_raw_fallback_to_y_data(self):
        ci = CurveInfo(
            var_name="v", curve=None,
            x_data=np.arange(3, dtype=float),
            y_data=np.array([1.0, 2.0, 3.0]),
        )
        assert ci.y_raw is ci.y_data
        assert ci.display_name == ""
        assert ci.factor == 1.0 and ci.offset == 0.0

    def test_invalid_ref_defaults_and_carry(self):
        ref = InvalidCurveRef(var_name="v")
        assert ref.display_name == "" and ref.factor == 1.0 and ref.offset == 0.0
        ref2 = InvalidCurveRef(
            var_name="v", display_name="别名", factor=2.0, offset=-1.0
        )
        assert (ref2.display_name, ref2.factor, ref2.offset) == ("别名", 2.0, -1.0)