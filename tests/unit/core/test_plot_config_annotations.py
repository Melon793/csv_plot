"""模板配置里的标注字段：向后兼容 + 往返。

核心风险是**向后兼容**：v0.4.3 之前生成、已经躺在用户模板目录里的 YAML
没有 ``annotations`` 键。新增字段如果写成必填，老模板会整份读不出来 ——
那比"没有标注功能"严重得多。本文件同时锁住数据结构和真实 YAML 文件两条路。

真实读写链路的端到端往返（建标注 → 存模板 → 清空 → 套模板）见
tests/e2e/test_annotation_roundtrip.py。
"""

from __future__ import annotations

import yaml
import pytest

from src.core.plot_config import PlotConfig, PlotSessionConfig
from src.core.storage import TemplateStorage

# 一条最小合法标注（字段取自 annotation_models 的默认值形态）
_SAMPLE_ANNOTATION = {
    "id": "aa11bb22",
    "kind": "rect",
    "points": [[0.0, 0.0], [2.0, 3.0]],
    "text": "",
    "stroke": "#E24B4A",
    "stroke_width": 2,
    "fill": None,
    "fill_alpha": 40,
    "font_size": 12,
    "anchor": "data",
    "visible": True,
}


class TestPlotConfigAnnotations:
    def test_defaults_to_empty(self):
        assert PlotConfig().annotations == []

    def test_to_dict_carries_annotations(self):
        cfg = PlotConfig(curves=["a"], annotations=[_SAMPLE_ANNOTATION])
        assert cfg.to_dict()["annotations"] == [_SAMPLE_ANNOTATION]

    def test_roundtrip_preserves_annotations(self):
        cfg = PlotConfig(curves=["a"], annotations=[_SAMPLE_ANNOTATION])
        restored = PlotConfig.from_dict(cfg.to_dict())
        assert restored == cfg
        assert restored.annotations == [_SAMPLE_ANNOTATION]

    def test_legacy_dict_without_key_loads_as_empty(self):
        """v0.4.3 之前的字典形态：没有 annotations 键"""
        cfg = PlotConfig.from_dict({"curves": ["a", "b"]})
        assert cfg.curves == ["a", "b"]
        assert cfg.annotations == []

    @pytest.mark.parametrize("bad", [None, "abc", 42, {"a": 1}])
    def test_non_list_annotations_degrade_to_empty(self, bad):
        # 手改过 / 被别的工具写坏的模板：当作没有标注，而不是抛异常
        cfg = PlotConfig.from_dict({"curves": ["a"], "annotations": bad})
        assert cfg.curves == ["a"]
        assert cfg.annotations == []

    def test_annotations_are_copied_not_aliased(self):
        src = [_SAMPLE_ANNOTATION]
        cfg = PlotConfig.from_dict({"annotations": src})
        assert cfg.annotations == src and cfg.annotations is not src

    def test_annotations_may_outlive_curves(self):
        """只有标注、没有曲线的子图必须能表达（否则用户辛苦标的图一存就丢）"""
        cfg = PlotConfig.from_dict({"curves": [], "annotations": [_SAMPLE_ANNOTATION]})
        assert cfg.curves == [] and len(cfg.annotations) == 1


class TestPlotSessionConfigAnnotations:
    def test_roundtrip_keeps_per_plot_annotations(self):
        cfg = PlotSessionConfig(
            layout_rows=1,
            layout_cols=2,
            plots=[
                PlotConfig(curves=["a"], annotations=[_SAMPLE_ANNOTATION]),
                PlotConfig(curves=["b"]),
            ],
        )
        restored = PlotSessionConfig.from_dict(cfg.to_dict())
        assert restored.plots[0].annotations == [_SAMPLE_ANNOTATION]
        assert restored.plots[1].annotations == []

    def test_legacy_session_dict_loads(self):
        legacy = {
            "created_at": "2025-01-01T00:00:00",
            "layout_rows": 2,
            "layout_cols": 2,
            "time_factor": 1.0,
            "time_offset": 0.0,
            "plots": [{"curves": ["speed", "rpm"]}, {"curves": ["flag"]}],
        }
        cfg = PlotSessionConfig.from_dict(legacy)
        assert cfg.layout_rows == 2
        assert cfg.plots[0].curves == ["speed", "rpm"]
        assert all(p.annotations == [] for p in cfg.plots)


class TestLegacyTemplateFileOnDisk:
    """真实 YAML 文件路径：用户模板目录里的老文件必须照常读出来。

    不直接调 from_dict —— 那样绕开了 yaml.safe_load 与
    TemplateStorage._validate_template_data，而"读不出来"最常见的原因恰恰
    是这两层。
    """

    def _write_legacy_yaml(self, path) -> None:
        legacy = {
            "metadata": {
                "id": "0a1b2c3d",
                "name": "老模板",
                "description": "v0.4.3 之前生成",
                "created_at": "2025-01-01T00:00:00",
                "updated_at": "2025-01-01T00:00:00",
                "source_file": None,
            },
            "config": {
                "created_at": "2025-01-01T00:00:00",
                "layout_rows": 1,
                "layout_cols": 1,
                "time_factor": 1.0,
                "time_offset": 0.0,
                # 注意：这里的 plot 字典没有 annotations 键
                "plots": [{"curves": ["speed", "rpm"]}],
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(legacy, f, default_flow_style=False, allow_unicode=True, indent=2)

    def test_template_without_annotations_key_still_reads(self, tmp_path):
        storage = TemplateStorage(tmp_path)
        legacy_file = tmp_path / "legacy.yaml"
        self._write_legacy_yaml(legacy_file)

        template = storage.read_template_from_file(legacy_file)
        assert template is not None, "老模板整份读不出来 = 向后兼容被破坏"

        config = PlotSessionConfig.from_dict(template.config)
        assert config.plots[0].curves == ["speed", "rpm"]
        assert config.plots[0].annotations == []

    def test_legacy_template_survives_write_back(self, tmp_path):
        """老模板读出来再存回去：曲线不动、标注补成空表（升级路径）"""
        storage = TemplateStorage(tmp_path)
        legacy_file = tmp_path / "legacy.yaml"
        self._write_legacy_yaml(legacy_file)
        template = storage.read_template_from_file(legacy_file)
        assert template is not None

        config = PlotSessionConfig.from_dict(template.config)
        template.config = config.to_dict()
        # 模拟"原地升级"：老文件被按模板名生成的新文件取代
        legacy_file.unlink()
        assert storage.write_template(template) is True

        # 用新实例读，绕开存储层缓存 —— 否则这里验的是内存对象不是磁盘内容
        again = TemplateStorage(tmp_path).read_template("0a1b2c3d")
        assert again is not None
        assert again.config["plots"][0]["curves"] == ["speed", "rpm"]
        assert again.config["plots"][0]["annotations"] == []

    def test_new_template_with_annotations_round_trips_through_disk(self, tmp_path):
        storage = TemplateStorage(tmp_path)
        source = tmp_path / "with_ann.yaml"
        with open(source, "w", encoding="utf-8") as f:
            yaml.dump(
                {
                    "metadata": {"id": "ffeeddcc", "name": "带标注"},
                    "config": PlotSessionConfig(
                        plots=[PlotConfig(curves=["a"], annotations=[_SAMPLE_ANNOTATION])]
                    ).to_dict(),
                },
                f,
                default_flow_style=False,
                allow_unicode=True,
                indent=2,
            )

        template = storage.read_template_from_file(source)
        assert template is not None
        config = PlotSessionConfig.from_dict(template.config)
        assert config.plots[0].annotations == [_SAMPLE_ANNOTATION]
