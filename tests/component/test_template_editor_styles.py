"""模板编辑器「编辑→保存」不得清空外观束（评审 P0 护栏）。

编辑器打开模板用 _generate_yaml_with_comments 白名单再生成全文、保存以
文本为准；漏写 curve_styles 等于静默清空跨会话外观（R16 同类）。
"""
from __future__ import annotations

import yaml

from src.ui.dialogs.template_editor_dialog import TemplateEditorDialog


def test_editor_regeneration_preserves_curve_styles():
    config = {
        "pages": [
            {
                "title": "图 1",
                "layout_rows": 1,
                "layout_cols": 1,
                "plots": [
                    {
                        "curves": ["speed"],
                        "curve_styles": {
                            "speed": {
                                "color": "#D62728",
                                "line_preset": "dash",
                                "symbol_policy": "on",
                            }
                        },
                    }
                ],
            }
        ]
    }
    text = TemplateEditorDialog._generate_yaml_with_comments(config)
    restored = yaml.safe_load(text)
    styles = restored["pages"][0]["plots"][0]["curve_styles"]
    # YAML 1.1 布尔陷阱：on 必须带引号落盘，safe_load 回读仍是字符串
    assert styles == config["pages"][0]["plots"][0]["curve_styles"]


def test_editor_regeneration_without_styles_unchanged_keys():
    config = {
        "pages": [
            {
                "title": "图 1",
                "layout_rows": 1,
                "layout_cols": 1,
                "plots": [{"curves": ["speed"], "cell_index": 0}],
            }
        ]
    }
    text = TemplateEditorDialog._generate_yaml_with_comments(config)
    plot = yaml.safe_load(text)["pages"][0]["plots"][0]
    assert "curve_styles" not in plot
    assert plot["curves"] == ["speed"]
    assert plot["cell_index"] == 0
