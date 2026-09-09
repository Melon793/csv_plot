"""core.template_models 单元测试：元数据/模板序列化与变量统计。"""

from __future__ import annotations

from src.core.plot_config import PlotConfig, PlotSessionConfig
from src.core.template_models import (
    PlotTemplate,
    TemplateMetadata,
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


class TestExtractVariablesFromConfig:
    def test_dict_form(self):
        config = {"plots": [{"curves": ["a", "b"]}, None]}
        assert extract_variables_from_config(config) == {"a", "b"}

    def test_object_form(self):
        cfg = PlotSessionConfig(
            plots=[PlotConfig(curves=["x", "y"]), PlotConfig(curves=["y"])]
        )
        assert extract_variables_from_config(cfg) == {"x", "y"}
