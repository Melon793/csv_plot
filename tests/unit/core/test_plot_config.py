"""core.plot_config 单元测试：配置序列化往返与模板异常体系。"""

from __future__ import annotations

from src.core.plot_config import (
    PlotConfig,
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


class TestTemplateExceptions:
    def test_hierarchy(self):
        for exc_type in (
            TemplateNotFoundError,
            TemplateNameConflictError,
            TemplateValidationError,
            TemplateStorageError,
        ):
            assert issubclass(exc_type, TemplateError)
