"""模板 curve_settings 导出/套用/占位保真 + 编辑器 YAML 再生成保键（C5 / D5/D6）。"""

import types

import pandas as pd
import pytest
import yaml

from src.core.plot_config import PlotConfig, PlotSessionConfig
from src.ui.plot_config_manager import PlotConfigManager
from src.ui.dialogs.template_editor_dialog import TemplateEditorDialog


@pytest.fixture()
def pw_env(plot_factory, qapp):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    pw = plot_factory(df)
    host = pw.window()
    host.loader = types.SimpleNamespace(var_names=["a", "b"], units={})
    host.data_validity = {}
    return pw, host


class TestCollect:
    def test_collect_non_default_only(self, pw_env):
        pw, _ = pw_env
        assert pw.add_variable_to_plot("a")
        assert pw.add_variable_to_plot("b")
        pw.curves["a"].display_name = "扭矩A"
        pw.curves["a"].factor = 2.0
        pw.curves["a"].offset = -10.0
        out = PlotConfigManager._collect_curve_settings(pw)
        assert out == {"a": {"display_name": "扭矩A", "factor": 2.0, "offset": -10.0}}
        assert "b" not in out

    def test_collect_includes_invalid(self, pw_env):
        pw, _ = pw_env
        from src.core.data_types import InvalidCurveRef
        pw.invalid_curves["ghost"] = InvalidCurveRef(
            var_name="ghost", display_name="旧名", factor=3.0
        )
        out = PlotConfigManager._collect_curve_settings(pw)
        assert out["ghost"] == {"display_name": "旧名", "factor": 3.0}


class TestApply:
    def test_apply_transform_recomputes_y_limits(self, pw_env):
        """X3/F4：模板先按 raw 画完再套换算，套用后 limits 必须覆盖投影值。"""
        pw, host = pw_env
        cfg = PlotConfig(
            curves=["a"],
            curve_settings={"a": {"factor": 2.0, "offset": -10.0}},
        )
        PlotConfigManager()._apply_plot_config(pw, cfg, host, 0)
        lo, hi = pw.view_box.state["limits"]["yLimits"]
        # a 原始 [1,2,3] → 投影 [-8,-6,-4]
        assert lo <= -8.0 and hi >= -4.0
        view_lo, view_hi = pw.view_box.viewRange()[1]
        assert view_lo <= -8.0 + 1e-6 and view_hi >= -4.0 - 1e-6

    def test_hidden_page_transform_limits_recomputed(self, pw_env):
        """隐藏页 scene=False：finalize 按投影 y_data 重算 limits。"""
        pw, host = pw_env
        cfg = PlotConfig(
            curves=["a"],
            curve_settings={"a": {"factor": 2.0, "offset": -10.0}},
        )
        PlotConfigManager()._apply_plot_config(pw, cfg, host, 0, scene=False)
        lo, hi = pw.view_box.state["limits"]["yLimits"]
        assert lo <= -8.0 and hi >= -4.0

    def test_apply_restores_name_and_transform(self, pw_env):
        pw, host = pw_env
        cfg = PlotConfig(
            curves=["a", "b"],
            curve_settings={"a": {"display_name": "扭矩A", "factor": 2.0, "offset": -10.0}},
        )
        PlotConfigManager()._apply_plot_config(pw, cfg, host, 0)
        ci = pw.curves["a"]
        assert ci.display_name == "扭矩A"
        assert ci.factor == 2.0 and ci.offset == -10.0
        import numpy as np
        np.testing.assert_allclose(ci.y_data, 2.0 * ci.y_raw - 10.0)
        assert "扭矩A" in pw._multi_curve_manager._build_legend_html()
        # b 无设置 → 默认
        assert pw.curves["b"].display_name == ""
        assert pw.curves["b"].factor == 1.0

    def test_apply_invalid_carries_settings(self, pw_env):
        pw, host = pw_env
        cfg = PlotConfig(
            curves=["ghost"],
            curve_settings={"ghost": {"display_name": "旧名", "factor": 3.0, "offset": 1.0}},
        )
        PlotConfigManager()._apply_plot_config(pw, cfg, host, 0)
        ref = pw.invalid_curves["ghost"]
        assert ref.display_name == "旧名"
        assert ref.factor == 3.0 and ref.offset == 1.0

    def test_apply_silent_no_status_spam(self, pw_env):
        pw, host = pw_env
        msgs = []
        host._show_status_message = lambda m: msgs.append(m)
        cfg = PlotConfig(
            curves=["a"], curve_settings={"a": {"factor": 2.0}}
        )
        PlotConfigManager()._apply_plot_config(pw, cfg, host, 0)
        assert not any("换算" in m for m in msgs)


class TestEditorYamlKeepsSettings:
    def test_regenerate_preserves_curve_settings(self):
        config = {
            "plots": [
                {
                    "curves": ["a"],
                    "curve_settings": {
                        "a": {"display_name": "扭矩A", "factor": 2.0, "offset": -10.0}
                    },
                }
            ]
        }
        text = TemplateEditorDialog._generate_yaml_with_comments(config)
        assert "curve_settings:" in text
        assert 'display_name: "扭矩A"' in text
        assert "factor: 2.0" in text
        # 再生成 → 解析 → 不丢键（编辑→保存往返）
        parsed = yaml.safe_load(text)
        sess = PlotSessionConfig.from_dict(parsed)
        plot = sess.normalized_pages()[0].plots[0]
        assert plot.curve_settings == {
            "a": {"display_name": "扭矩A", "factor": 2.0, "offset": -10.0}
        }

    def test_regenerate_without_settings_omits_key(self):
        config = {"plots": [{"curves": ["a"]}]}
        text = TemplateEditorDialog._generate_yaml_with_comments(config)
        assert "curve_settings:" not in text