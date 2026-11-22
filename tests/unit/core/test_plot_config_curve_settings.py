"""PlotConfig.curve_settings 序列化/容错（编辑器强化轮 C5 / D5/D6）。"""

from src.core.plot_config import PlotConfig, parse_curve_settings


class TestRoundtrip:
    def test_old_doc_without_key_reads_empty(self):
        cfg = PlotConfig.from_dict({"curves": ["a"]})
        assert cfg.curve_settings == {}
        assert "curve_settings" not in cfg.to_dict()

    def test_non_default_roundtrip(self):
        cfg = PlotConfig(
            curves=["a"],
            curve_settings={"a": {"display_name": "扭矩A", "factor": 2.0, "offset": -10.0}},
        )
        d = cfg.to_dict()
        assert d["curve_settings"] == {
            "a": {"display_name": "扭矩A", "factor": 2.0, "offset": -10.0}
        }
        back = PlotConfig.from_dict(d)
        assert back.curve_settings == cfg.curve_settings

    def test_default_only_entry_dropped_on_parse(self):
        cfg = PlotConfig.from_dict(
            {"curves": ["a"], "curve_settings": {"a": {"factor": 1.0, "offset": 0.0}}}
        )
        assert cfg.curve_settings == {}

    def test_partial_keys(self):
        cfg = PlotConfig.from_dict(
            {"curves": ["a"], "curve_settings": {"a": {"offset": 5.0}}}
        )
        assert cfg.curve_settings == {"a": {"offset": 5.0}}


class TestGuards:
    def test_non_dict_raw(self):
        assert parse_curve_settings(None) == {}
        assert parse_curve_settings([1, 2]) == {}
        assert parse_curve_settings("x") == {}

    def test_bool_rejected(self):
        assert parse_curve_settings({"a": {"factor": True}}) == {}
        assert parse_curve_settings({"a": {"offset": False}}) == {}

    def test_garbage_number_rejected(self):
        assert parse_curve_settings({"a": {"factor": "abc"}}) == {}
        assert parse_curve_settings({"a": {"factor": float("nan")}}) == {}
        assert parse_curve_settings({"a": {"offset": float("inf")}}) == {}

    def test_numeric_string_coerced(self):
        assert parse_curve_settings({"a": {"factor": "2"}}) == {"a": {"factor": 2.0}}

    def test_unknown_keys_ignored(self):
        out = parse_curve_settings({"a": {"color": "red", "factor": 2.0}})
        assert out == {"a": {"factor": 2.0}}

    def test_non_dict_entry_dropped(self):
        assert parse_curve_settings({"a": "x", "b": {"factor": 2.0}}) == {
            "b": {"factor": 2.0}
        }

    def test_empty_display_name_dropped(self):
        assert parse_curve_settings({"a": {"display_name": "  "}}) == {}

    def test_int_key_strified(self):
        assert parse_curve_settings({1: {"factor": 2.0}}) == {"1": {"factor": 2.0}}