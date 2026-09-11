"""metadata 单元测试：VarMetadata、枚举转换判定与枚举映射提取。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.data.metadata import (
    INVALID,
    UNKNOWN,
    VALID,
    VarMetadata,
    enum_conversion_types,
    extract_enum_map,
    is_enum_conversion,
    is_mdf3_version,
)


class TestVarMetadata:
    def test_defaults(self):
        m = VarMetadata(name="v", unit="V")
        assert m.group_index == 0
        assert m.is_enum is False
        assert m.validity == UNKNOWN
        assert m.enum_map is None
        # 采样率语义修正：标称间隔（秒）与平均有效采样率（Hz）分为两个字段，
        # 旧的 sampling_rate_hz 语义错误（MDF3 存秒间隔、MDF4 恒为 None）已删除
        assert m.nominal_raster_s is None
        assert m.effective_rate_hz is None
        # original_name 空串表示未发生聚合改名
        assert m.original_name == ""
        assert m.sample_count == 0
        assert m.time_min == 0.0
        assert m.time_max == 0.0

    def test_validity_constants(self):
        assert VALID == 1
        assert INVALID == -1


class TestIsMdf3Version:
    """MDF 2.x/3.x 与 4.x 的 conversion_type 数值语义完全不同，判定必须分版本。"""

    @pytest.mark.parametrize("version", ["3.00", "3.30", "2.00", "2.10"])
    def test_mdf3_family(self, version):
        assert is_mdf3_version(version) is True

    @pytest.mark.parametrize("version", ["4.00", "4.10", "4.20"])
    def test_mdf4_family(self, version):
        assert is_mdf3_version(version) is False

    @pytest.mark.parametrize("version", [None, ""])
    def test_missing_version_falls_back_to_mdf4(self, version):
        # 版本缺失时按 MDF4 取表：生产路径唯一调用点必然能拿到 _mdf.version，
        # 此分支仅为防御，需保证不抛异常
        assert is_mdf3_version(version) is False
        assert enum_conversion_types(version) == enum_conversion_types("4.10")


class TestEnumConversionTypes:
    def test_mdf4_table(self):
        """MDF4 枚举集合 = 输出为**文本**的转换。

        依据 asammdf 8.8.9 的 CONVERSION_TYPE_TO_STRING：
        7 TABX / 8 RTABX / 9 TTAB / 10 TRANS / 11 BITFIELD。
        注意 6 RTAB 不在内 —— 它是范围表但输出仍为数值。
        """
        assert enum_conversion_types("4.10") == frozenset({7, 8, 9, 10, 11})

    def test_mdf3_table(self):
        # TABX 文本表 / RTABX 范围文本表
        assert enum_conversion_types("3.00") == frozenset({11, 12})

    def test_mdf4_rtab_is_excluded(self):
        """回归防护：RTAB(6) 是范围**数值**表，不属于枚举。

        RTAB 的转换块内只有 lower_i/upper_i 而没有 text_i，
        extract_enum_map() 对它必定返回 None；若误判为枚举，
        get_series() 会走 raw=True 返回原始码值，且无任何文本标签可用，
        绘图静默展示错误数值（与 MDF3 的 EXPO/RAT 误判同一后果链）。
        对比：TAB(5)/TABI(4) 同样输出数值，也均不在集合内。
        """
        assert 6 not in enum_conversion_types("4.10")
        assert 5 not in enum_conversion_types("4.10")
        assert 4 not in enum_conversion_types("4.10")
        # RTABX(8) 是范围**文本**表，必须保留
        assert 8 in enum_conversion_types("4.10")


class TestIsEnumConversion:
    def test_none_is_not_enum(self):
        assert is_enum_conversion(None, "4.10") is False

    @pytest.mark.parametrize("ct", [7, 8, 9, 10, 11])
    def test_mdf4_enum_conversion_types(self, ct):
        conv = SimpleNamespace(conversion_type=ct)
        assert is_enum_conversion(conv, "4.10") is True

    @pytest.mark.parametrize("ct", [0, 1, 2, 3, 4, 5, 6])
    def test_mdf4_non_enum_conversion_types(self, ct):
        """NON/LIN/RAT/ALG/TABI/TAB/RTAB 均输出数值，不得判为枚举。"""
        conv = SimpleNamespace(conversion_type=ct)
        assert is_enum_conversion(conv, "4.10") is False

    @pytest.mark.parametrize("ct", [11, 12])
    def test_mdf3_enum_conversion_types(self, ct):
        conv = SimpleNamespace(conversion_type=ct)
        assert is_enum_conversion(conv, "3.00") is True

    @pytest.mark.parametrize("ct", [7, 9, 10])
    def test_mdf3_numeric_conversions_are_not_enum(self, ct):
        """回归防护：MDF3 的 EXPO(7)/RAT(9)/FORMULA(10) 是纯数值转换。

        误判为枚举会使 get_series() 走 raw=True 返回原始码值，且因 enum_map
        为 None 不会触发 enum 格式化，UI 静默显示错误数值无任何提示。
        实测某 λ 滤波时间变量因此显示 655 而真实物理值为 1.00054962。
        """
        conv = SimpleNamespace(conversion_type=ct)
        assert is_enum_conversion(conv, "3.00") is False

    def test_same_ct_value_has_opposite_meaning_across_versions(self):
        """ct=7 在 MDF4 是 TABX（枚举）、在 MDF3 是 EXPO（纯数值）。"""
        conv = SimpleNamespace(conversion_type=7)
        assert is_enum_conversion(conv, "4.10") is True
        assert is_enum_conversion(conv, "3.00") is False

    def test_non_enum_conversion_type(self):
        conv = SimpleNamespace(conversion_type=0)
        assert is_enum_conversion(conv, "3.00") is False

    def test_missing_conversion_type_attribute(self):
        conv = SimpleNamespace()
        assert is_enum_conversion(conv, "4.10") is False


class _TextAttrConversion:
    """模拟 asammdf 的 text_i / param_val_i 属性型枚举转换"""

    def __init__(self, entries: list[tuple[float, str]], ref_param_nr: int | None = None):
        self.conversion_type = 9
        self.val_to_text = None
        if ref_param_nr is not None:
            self.ref_param_nr = ref_param_nr
        for i, (val, text) in enumerate(entries):
            setattr(self, f"text_{i}", text)
            setattr(self, f"param_val_{i}", val)


class TestExtractEnumMap:
    def test_none_returns_none(self):
        assert extract_enum_map(None) is None

    def test_val_to_text_path(self):
        conv = SimpleNamespace(val_to_text={0: b"Off", 1: "On"})
        result = extract_enum_map(conv)
        assert result == {0: "Off", 1: "On"}

    def test_text_attr_path(self):
        conv = _TextAttrConversion([(0.0, "Off"), (1.0, "On")])
        result = extract_enum_map(conv)
        assert result == {0: "Off", 1: "On"}

    def test_text_attr_with_bytes_labels(self):
        conv = _TextAttrConversion([(0.0, b"Off\x00"), (1.0, b"On\x00")])
        result = extract_enum_map(conv)
        assert result == {0: "Off", 1: "On"}

    def test_empty_conversion_returns_none(self):
        conv = SimpleNamespace(conversion_type=9, val_to_text=None)
        assert extract_enum_map(conv) is None
