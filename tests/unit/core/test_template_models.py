"""core.template_models 单元测试：元数据/模板序列化与变量统计。"""

from __future__ import annotations

from src.core.plot_config import PlotConfig, PlotPageConfig, PlotSessionConfig
from src.core.template_models import (
    PlotTemplate,
    TemplateMetadata,
    count_template_pages,
    count_template_plots,
    count_template_variables,
    extract_variables_from_config,
)


class TestTemplateMetadata:
    def test_timestamps_auto_filled(self):
        m = TemplateMetadata(id="abc123", name="t1")
        assert m.created_at != ""
        assert m.updated_at != ""

    def test_roundtrip(self):
        m = TemplateMetadata(id="abc123", name="t1", description="desc")
        restored = TemplateMetadata.from_dict(m.to_dict())
        assert restored == m


class TestPlotTemplate:
    def test_roundtrip(self):
        tpl = PlotTemplate(
            metadata=TemplateMetadata(id="id1", name="n1"),
            config={"plots": [{"curves": ["a"]}]},
        )
        restored = PlotTemplate.from_dict(tpl.to_dict())
        assert restored.metadata.id == "id1"
        assert restored.config == tpl.config


class TestCountTemplateVariables:
    def test_counts_unique_across_plots(self):
        config = {
            "plots": [
                {"curves": ["a", "b"]},
                {"curves": ["b", "c"]},
            ]
        }
        assert count_template_variables(config) == 3

    def test_none_plots_handled(self):
        assert count_template_variables({"plots": None}) == 0
        assert count_template_variables({}) == 0

    def test_v2_counts_across_all_pages(self):
        config = {
            "plots": [{"curves": ["a"]}],
            "pages": [
                {"plots": [{"curves": ["a", "b"]}]},
                {"plots": [{"curves": ["b", "c"]}, {"curves": []}]},
            ],
        }
        assert count_template_variables(config) == 3

    def test_count_template_plots_and_pages(self):
        v2 = {
            "pages": [
                {"plots": [{"curves": ["a"]}, {"curves": []}]},
                {"plots": [{"curves": ["b", "c"]}]},
            ]
        }
        assert count_template_plots(v2) == 2
        assert count_template_pages(v2) == 2
        assert count_template_plots({"plots": [{"curves": ["a"]}]}) == 1
        assert count_template_pages({"plots": [{"curves": ["a"]}]}) == 1
        assert count_template_pages({}) == 1


class TestExtractVariablesFromConfig:
    def test_dict_form(self):
        config = {"plots": [{"curves": ["a", "b"]}, None]}
        assert extract_variables_from_config(config) == {"a", "b"}

    def test_object_form(self):
        cfg = PlotSessionConfig(
            plots=[PlotConfig(curves=["x", "y"]), PlotConfig(curves=["y"])]
        )
        assert extract_variables_from_config(cfg) == {"x", "y"}

    def test_object_form_v2_reads_all_pages(self):
        cfg = PlotSessionConfig(
            pages=[
                PlotPageConfig(plots=[PlotConfig(curves=["x"])]),
                PlotPageConfig(plots=[PlotConfig(curves=["y", "z"])]),
            ]
        )
        assert extract_variables_from_config(cfg) == {"x", "y", "z"}
