"""metadata 单元测试：VarMetadata、枚举转换判定与枚举映射提取。"""

from __future__ import annotations

from types import SimpleNamespace

from src.data.metadata import (
    INVALID,
    UNKNOWN,
    VALID,
    VarMetadata,
    extract_enum_map,
    is_enum_conversion,
)


class TestVarMetadata:
    def test_defaults(self):
        m = VarMetadata(name="v", unit="V")
        assert m.group_index == 0
        assert m.is_enum is False
        assert m.validity == UNKNOWN
        assert m.enum_map is None

    def test_validity_constants(self):
        assert VALID == 1
        assert INVALID == -1


class TestIsEnumConversion:
    def test_none_is_not_enum(self):
        assert is_enum_conversion(None) is False

    def test_enum_conversion_types(self):
        for ct in (7, 9, 10, 11):
            conv = SimpleNamespace(conversion_type=ct)
            assert is_enum_conversion(conv) is True

    def test_non_enum_conversion_type(self):
        conv = SimpleNamespace(conversion_type=0)
        assert is_enum_conversion(conv) is False

    def test_missing_conversion_type_attribute(self):
        conv = SimpleNamespace()
        assert is_enum_conversion(conv) is False


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
