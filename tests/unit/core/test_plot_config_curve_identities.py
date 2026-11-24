"""PlotConfig.curve_identities 序列化/容错（模板身份匹配 C1 / D2 / R9）。"""

import pytest

from src.core.plot_config import (
    PlotConfig,
    PlotSessionConfig,
    parse_curve_identities,
)


MDF_ENTRY = {"original_name": "Press", "kind": "mdf", "device": "Dev1", "ecu": "ECU_A"}
BLF_ENTRY = {
    "original_name": "EngineSpeed",
    "kind": "blf",
    "frame": "EngineData",
    "can_id": 0x1A0,
    "extended": False,
    "bus_channel": 0,
}


class TestRoundtrip:
    def test_old_doc_without_key_reads_empty(self):
        cfg = PlotConfig.from_dict({"curves": ["a"]})
        assert cfg.curve_identities == {}
        assert "curve_identities" not in cfg.to_dict()

    def test_empty_dict_not_written(self):
        cfg = PlotConfig(curves=["a"], curve_identities={})
        assert "curve_identities" not in cfg.to_dict()

    def test_mdf_roundtrip(self):
        cfg = PlotConfig(curves=["Press_G0"], curve_identities={"Press_G0": dict(MDF_ENTRY)})
        d = cfg.to_dict()
        assert d["curve_identities"] == {"Press_G0": MDF_ENTRY}
        back = PlotConfig.from_dict(d)
        assert back.curve_identities == cfg.curve_identities

    def test_blf_roundtrip(self):
        cfg = PlotConfig(
            curves=["EngineData.EngineSpeed_G0"],
            curve_identities={"EngineData.EngineSpeed_G0": dict(BLF_ENTRY)},
        )
        back = PlotConfig.from_dict(cfg.to_dict())
        assert back.curve_identities == cfg.curve_identities

    def test_session_roundtrip_carries_identities(self):
        """auto-save 与模板共用 PlotSessionConfig：身份字段随全档往返。"""
        cfg = PlotSessionConfig(
            plots=[PlotConfig(curves=["Press_G0"], curve_identities={"Press_G0": dict(MDF_ENTRY)})]
        )
        back = PlotSessionConfig.from_dict(cfg.to_dict())
        assert back.plots[0].curve_identities == {"Press_G0": MDF_ENTRY}
        assert back.pages[0].plots[0].curve_identities == {"Press_G0": MDF_ENTRY}

    def test_curves_field_untouched(self):
        """D2：不改 curves 本身——身份是旁挂字段。"""
        cfg = PlotConfig(curves=["Press_G0"], curve_identities={"Press_G0": dict(MDF_ENTRY)})
        assert cfg.to_dict()["curves"] == ["Press_G0"]


class TestGuards:
    def test_non_dict_raw(self):
        assert parse_curve_identities(None) == {}
        assert parse_curve_identities([1, 2]) == {}
        assert parse_curve_identities("x") == {}

    def test_non_dict_entry_dropped(self):
        assert parse_curve_identities({"a": "x", "b": dict(MDF_ENTRY)}) == {
            "b": MDF_ENTRY
        }

    def test_missing_original_name_dropped(self):
        assert parse_curve_identities({"a": {"kind": "mdf", "device": "Dev1"}}) == {}
        assert parse_curve_identities(
            {"a": {"original_name": "  ", "kind": "mdf"}}
        ) == {}
        assert parse_curve_identities({"a": {"original_name": 5, "kind": "mdf"}}) == {}

    def test_unknown_kind_dropped(self):
        assert parse_curve_identities(
            {"a": {"original_name": "Press", "kind": "csv"}}
        ) == {}
        assert parse_curve_identities({"a": {"original_name": "Press"}}) == {}

    def test_mdf_without_si_kept(self):
        """真实 MDF3 常无 SI：只有 original_name+kind 也保留（L1 退 L2）。"""
        out = parse_curve_identities({"a": {"original_name": "Press", "kind": "mdf"}})
        assert out == {"a": {"original_name": "Press", "kind": "mdf"}}

    def test_mdf_blank_si_dropped_field(self):
        out = parse_curve_identities(
            {"a": {"original_name": "Press", "kind": "mdf", "device": " ", "ecu": 3}}
        )
        assert out == {"a": {"original_name": "Press", "kind": "mdf"}}

    def test_blf_partial_fingerprint_dropped(self):
        """D12：BLF 四元组必须齐全，半截指纹整条丢弃。"""
        for missing in ("frame", "can_id", "extended", "bus_channel"):
            entry = dict(BLF_ENTRY)
            del entry[missing]
            assert parse_curve_identities({"a": entry}) == {}, missing

    def test_blf_bool_int_rejected(self):
        """照 cell_index P2-44：bool 拒收（int(True)→1 会静默错配）。"""
        entry = dict(BLF_ENTRY, can_id=True)
        assert parse_curve_identities({"a": entry}) == {}
        entry = dict(BLF_ENTRY, bus_channel=False)
        assert parse_curve_identities({"a": entry}) == {}
        entry = dict(BLF_ENTRY, extended=1)
        assert parse_curve_identities({"a": entry}) == {}

    def test_blf_garbage_number_rejected(self):
        assert parse_curve_identities({"a": dict(BLF_ENTRY, can_id="abc")}) == {}
        assert parse_curve_identities({"a": dict(BLF_ENTRY, can_id=-1)}) == {}
        assert parse_curve_identities({"a": dict(BLF_ENTRY, bus_channel=-2)}) == {}

    def test_blf_numeric_string_coerced(self):
        out = parse_curve_identities({"a": dict(BLF_ENTRY, can_id="416")})
        assert out["a"]["can_id"] == 416

    @pytest.mark.parametrize(
        ("field", "value", "extended"),
        [
            ("can_id", 416.9, False),
            ("bus_channel", 1.9, False),
            ("can_id", True, False),
            ("bus_channel", False, False),
            ("can_id", -1, False),
            ("bus_channel", -1, False),
            ("can_id", 0x800, False),
            ("can_id", 0x20000000, True),
        ],
        ids=[
            "fractional-can-id",
            "fractional-bus-channel",
            "bool-can-id",
            "bool-bus-channel",
            "negative-can-id",
            "negative-bus-channel",
            "standard-can-id-overflow",
            "extended-can-id-overflow",
        ],
    )
    def test_r9_blf_invalid_integer_fields_rejected(self, field, value, extended):
        """R9：非整数值、bool、负值和 CAN ID 越界必须整条拒收，不得静默取整。"""
        entry = dict(BLF_ENTRY, extended=extended)
        entry[field] = value
        assert parse_curve_identities({"a": entry}) == {}

    @pytest.mark.parametrize("field", ["can_id", "bus_channel"])
    @pytest.mark.parametrize("value", [float("inf"), float("-inf")])
    def test_r9_blf_infinity_rejected_without_raising(self, field, value):
        """R9：int(inf) 的 OverflowError 不得逃出容错解析入口。"""
        entry = dict(BLF_ENTRY)
        entry[field] = value
        assert parse_curve_identities({"a": entry}) == {}

    @pytest.mark.parametrize(
        ("field", "value", "expected"),
        [
            ("can_id", "416", 416),
            ("can_id", "2047", 0x7FF),
            ("can_id", "536870911", 0x1FFFFFFF),
            ("bus_channel", "2", 2),
        ],
    )
    def test_r9_blf_integer_strings_and_boundaries_accepted(
        self, field, value, expected
    ):
        entry = dict(
            BLF_ENTRY,
            extended=(expected == 0x1FFFFFFF),
        )
        entry[field] = value
        parsed = parse_curve_identities({"a": entry})
        assert parsed["a"][field] == expected

    def test_unknown_keys_ignored(self):
        out = parse_curve_identities({"a": dict(MDF_ENTRY, acq_name="NormalGroup", color="red")})
        assert out == {"a": MDF_ENTRY}

    def test_none_key_dropped(self):
        assert parse_curve_identities({None: dict(MDF_ENTRY)}) == {}

    def test_from_dict_malformed_identities_no_raise(self):
        cfg = PlotConfig.from_dict(
            {"curves": ["a"], "curve_identities": "garbage"}
        )
        assert cfg.curve_identities == {}


class TestUnitField:
    """D9 一致性校验需要模板侧单位（C2 增补）。"""

    def test_unit_roundtrip(self):
        entry = dict(MDF_ENTRY, unit="bar")
        out = parse_curve_identities({"a": entry})
        assert out == {"a": entry}

    def test_dash_unit_dropped(self):
        out = parse_curve_identities({"a": dict(MDF_ENTRY, unit="-")})
        assert out == {"a": MDF_ENTRY}

    def test_blank_unit_dropped(self):
        out = parse_curve_identities({"a": dict(MDF_ENTRY, unit="  ")})
        assert out == {"a": MDF_ENTRY}

    def test_non_str_unit_dropped(self):
        out = parse_curve_identities({"a": dict(MDF_ENTRY, unit=5)})
        assert out == {"a": MDF_ENTRY}

    def test_blf_unit_roundtrip(self):
        entry = dict(BLF_ENTRY, unit="rpm")
        out = parse_curve_identities({"a": entry})
        assert out == {"a": entry}