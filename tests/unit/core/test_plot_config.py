"""core.plot_config 单元测试：配置序列化往返与模板异常体系。"""

from __future__ import annotations

from src.core.plot_config import (
    PlotConfig,
    PlotPageConfig,
    PlotSessionConfig,
    TemplateError,
    TemplateNameConflictError,
    TemplateNotFoundError,
    TemplateStorageError,
    TemplateValidationError,
)


class TestPlotConfig:
    def test_roundtrip(self):
        cfg = PlotConfig(curves=["a", "b"])
        assert PlotConfig.from_dict(cfg.to_dict()) == cfg

    def test_from_dict_missing_curves_defaults_empty(self):
        assert PlotConfig.from_dict({}).curves == []

    def test_invalid_curves_defaults_empty(self):
        assert PlotConfig(curves=["a"]).invalid_curves == []
        assert PlotConfig.from_dict({}).invalid_curves == []

    def test_invalid_curves_roundtrip(self):
        cfg = PlotConfig(curves=["a"], invalid_curves=["ghost"])
        assert PlotConfig.from_dict(cfg.to_dict()) == cfg
        assert cfg.to_dict()["invalid_curves"] == ["ghost"]

    def test_to_dict_omits_invalid_key_when_empty(self):
        """invalid_curves 为空时不写键：保持 v1 视图与旧档字节兼容（§8.4）"""
        assert PlotConfig(curves=["a"]).to_dict() == {"curves": ["a"]}

    def test_from_dict_ignores_unknown_keys(self):
        """旧版软件读新模板：未知键忽略；新版读旧档：缺省容错"""
        cfg = PlotConfig.from_dict({"curves": ["a"], "future_key": 1})
        assert cfg.curves == ["a"]
        assert cfg.invalid_curves == []

    def test_cell_index_defaults_none_and_omitted_in_to_dict(self):
        """P1-L：cell_index 缺省 None，为空不写键（旧档字节兼容）。"""
        cfg = PlotConfig(curves=["a"])
        assert cfg.cell_index is None
        assert "cell_index" not in cfg.to_dict()

    def test_cell_index_roundtrip(self):
        cfg = PlotConfig(curves=["a"], cell_index=3)
        assert cfg.to_dict()["cell_index"] == 3
        assert PlotConfig.from_dict(cfg.to_dict()).cell_index == 3

    def test_cell_index_from_dict_bad_values_fall_back_none(self):
        assert PlotConfig.from_dict({"cell_index": "abc"}).cell_index is None
        assert PlotConfig.from_dict({"cell_index": None}).cell_index is None
        assert PlotConfig.from_dict({}).cell_index is None


class TestPlotSessionConfig:
    def test_created_at_auto_filled(self):
        cfg = PlotSessionConfig()
        assert cfg.created_at != ""

    def test_roundtrip(self):
        cfg = PlotSessionConfig(
            layout_rows=2,
            layout_cols=3,
            time_factor=0.5,
            time_offset=10.0,
            plots=[PlotConfig(curves=["a"]), PlotConfig(curves=["b", "c"])],
        )
        restored = PlotSessionConfig.from_dict(cfg.to_dict())
        assert restored.layout_rows == 2
        assert restored.layout_cols == 3
        assert restored.time_factor == 0.5
        assert restored.time_offset == 10.0
        assert restored.plots[1].curves == ["b", "c"]
        assert restored.created_at == cfg.created_at

    def test_from_dict_defaults(self):
        cfg = PlotSessionConfig.from_dict({})
        assert cfg.layout_rows == 1
        assert cfg.layout_cols == 1
        assert cfg.time_factor == 1.0
        assert cfg.time_offset == 0.0
        assert cfg.plots == []
        assert cfg.pages == []
        assert cfg.active_page == 0
        assert cfg.cursor_mode is None
        assert cfg.saved_mark_range is None


class TestPlotSessionConfigV2:
    """模板 schema v2：pages 优先、v1 兼容、int 键还原（设计文档 §6.5）。"""

    @staticmethod
    def _v2_config() -> PlotSessionConfig:
        return PlotSessionConfig(
            time_factor=2.0,
            time_offset=5.0,
            pages=[
                PlotPageConfig(
                    title="图 1",
                    layout_rows=2,
                    layout_cols=2,
                    row_height_factors={0: 100, 1: 50},
                    plots=[PlotConfig(curves=["a"]), PlotConfig(curves=["b", "c"])],
                ),
                PlotPageConfig(
                    title="分析 B",
                    layout_rows=1,
                    layout_cols=3,
                    plots=[PlotConfig(curves=["d"])],
                ),
            ],
            active_page=1,
            cursor_mode="2 anchored cursor",
            saved_mark_range=[1.0, 2.5],
        )

    def test_is_v2_judges_by_pages_key_only(self):
        assert PlotSessionConfig.is_v2({"pages": [{}]}) is True
        assert PlotSessionConfig.is_v2({"plots": [{}]}) is False
        assert PlotSessionConfig.is_v2({"pages": []}) is False
        assert PlotSessionConfig.is_v2({}) is False

    def test_v1_normalized_to_single_page(self):
        cfg = PlotSessionConfig(
            layout_rows=2, layout_cols=3,
            plots=[PlotConfig(curves=["a", "b"])],
        )
        pages = cfg.normalized_pages()
        assert len(pages) == 1
        assert pages[0].layout_rows == 2
        assert pages[0].layout_cols == 3
        assert pages[0].plots[0].curves == ["a", "b"]

    def test_v2_roundtrip_keeps_pages_and_view_state(self):
        cfg = self._v2_config()
        restored = PlotSessionConfig.from_dict(cfg.to_dict())

        assert len(restored.pages) == 2
        assert restored.pages[1].title == "分析 B"
        assert restored.pages[0].plots[1].curves == ["b", "c"]
        assert restored.active_page == 1
        assert restored.cursor_mode == "2 anchored cursor"
        assert restored.saved_mark_range == [1.0, 2.5]
        assert restored.time_factor == 2.0
        assert restored.time_offset == 5.0

    def test_to_dict_dual_writes_first_page_as_v1_view(self):
        data = self._v2_config().to_dict()
        assert data["layout_rows"] == 2
        assert data["layout_cols"] == 2
        assert data["plots"] == [
            {"curves": ["a"]},
            {"curves": ["b", "c"]},
        ]
        assert len(data["pages"]) == 2

    def test_row_height_factors_string_keys_restored_to_int(self):
        data = {
            "pages": [{
                "layout_rows": 2,
                "layout_cols": 1,
                "row_height_factors": {"0": "100", "1": 60},
                "plots": [],
            }]
        }
        page = PlotSessionConfig.from_dict(data).normalized_pages()[0]
        assert page.row_height_factors == {0: 100, 1: 60}

    def test_optional_view_state_absent_stays_none(self):
        restored = PlotSessionConfig.from_dict({"pages": [{"plots": []}]})
        assert restored.cursor_mode is None
        assert restored.saved_mark_range is None


class TestTemplateExceptions:
    def test_hierarchy(self):
        for exc_type in (
            TemplateNotFoundError,
            TemplateNameConflictError,
            TemplateValidationError,
            TemplateStorageError,
        ):
            assert issubclass(exc_type, TemplateError)


class TestCellRowCol:
    """P1-7（第 5 轮）：(row, col) 坐标可选字段，旧档零回归。"""

    def test_roundtrip_keeps_row_col(self):
        cfg = PlotConfig(curves=["a"], cell_index=4, cell_row=1, cell_col=1)
        data = cfg.to_dict()
        assert data["cell_row"] == 1 and data["cell_col"] == 1
        restored = PlotConfig.from_dict(data)
        assert restored == cfg

    def test_old_dict_without_row_col_falls_back_to_none(self):
        restored = PlotConfig.from_dict({"curves": ["a"], "cell_index": 3})
        assert restored.cell_index == 3
        assert restored.cell_row is None and restored.cell_col is None
        assert "cell_row" not in restored.to_dict()

    def test_malformed_row_col_tolerated_as_none(self):
        restored = PlotConfig.from_dict(
            {"curves": [], "cell_row": "abc", "cell_col": {}}
        )
        assert restored.cell_row is None and restored.cell_col is None

    def test_row_col_written_only_when_both_present(self):
        assert "cell_row" not in PlotConfig(curves=[], cell_row=1).to_dict()
        assert "cell_col" not in PlotConfig(curves=[], cell_col=1).to_dict()


class TestMalformedConfigTolerance:
    """评审 B1（第 5 轮批次 4）：手写畸形档在 from_dict 层收口。"""

    def test_curves_null_or_scalar_becomes_empty_list(self):
        assert PlotConfig.from_dict({"curves": None}).curves == []
        assert PlotConfig.from_dict({"curves": "speed"}).curves == []

    def test_non_dict_plots_are_skipped(self):
        page = PlotPageConfig.from_dict(
            {"plots": ["speed", {"curves": ["rpm"]}, 5]}
        )
        assert [p.curves for p in page.plots] == [["rpm"]]
        session = PlotSessionConfig.from_dict({"plots": ["speed", {"curves": []}]})
        assert len(session.plots) == 1

    def test_non_dict_factors_tolerated(self):
        page = PlotPageConfig.from_dict({"row_height_factors": "abc", "plots": []})
        assert page.row_height_factors == {}

    def test_invalid_int_still_raises_valueerror_for_entry_catch(self):
        import pytest
        with pytest.raises(ValueError):
            PlotPageConfig.from_dict({"layout_rows": "abc", "plots": []})

    def test_curves_elements_are_stringified(self):
        """复审 BL-3：数字通道名 str 化后可匹配；unhashable/null 不得逃槽。"""
        cfg = PlotConfig.from_dict({"curves": [123, None, ["x"], {"n": 1}]})
        assert cfg.curves[0] == "123"
        assert None not in cfg.curves
        assert all(isinstance(c, str) for c in cfg.curves)

    def test_invalid_curves_scalar_becomes_empty(self):
        assert PlotConfig.from_dict({"invalid_curves": "abc"}).invalid_curves == []
        assert PlotConfig.from_dict({"invalid_curves": 5}).invalid_curves == []

    def test_v1_top_level_layout_invalid_raises_in_from_dict(self):
        import pytest
        with pytest.raises(ValueError):
            PlotSessionConfig.from_dict({"layout_rows": "abc", "plots": []})
        # null 键存在时 or 1 兜底（与页级同口径）
        assert PlotSessionConfig.from_dict({"layout_rows": None}).layout_rows == 1
