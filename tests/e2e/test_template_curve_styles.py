"""曲线外观束（curve_styles）跨会话链路 e2e（真实 MainWindow）。

覆盖：模板 保存→清场→套用 四字段恢复；auto-save YAML 往返；重复套用
颜色确定性；老档（无束）色板轮转零回归；空格子+束组合落位不前移；
束名缺失→灰占位；套用后 reload saved_state 接力。
"""
from __future__ import annotations

from PySide6.QtCore import Qt

from tests.fixtures.data_factory import make_simple_rows, write_csv
from tests.fixtures.waits import settle

from src.core.auto_save_manager import AutoSaveManager
from src.core.config import DEFAULT_CURVE_COLORS, INVALID_CURVE_COLOR
from src.core.data_types import (
    DRAW_MODE_SOURCE_USER,
    DRAW_MODE_STEP_POST,
    LINE_PRESET_DASH,
    SYMBOL_POLICY_ON,
)
from src.core.plot_config import PlotConfig, PlotPageConfig, PlotSessionConfig


def _pw0(mw):
    return mw.current_page.plot_widgets[0].plot_widget


def _styled_window(mw):
    """plot0 绘 speed/rpm 并钉住四字段外观（speed 色+预设，rpm 阶梯+数据点）。"""
    pw = _pw0(mw)
    for name in ("speed", "rpm"):
        if name not in pw.curves:
            assert pw.add_variable_to_plot(name)
    mcm = pw._multi_curve_manager
    assert mcm.apply_curve_color("speed", "#D62728")
    assert mcm.apply_curve_line_preset("speed", LINE_PRESET_DASH)
    assert mcm.apply_curve_draw_mode("rpm", DRAW_MODE_STEP_POST)
    assert mcm.apply_curve_symbol_policy("rpm", SYMBOL_POLICY_ON)
    settle()
    return pw


def _assert_styles(pw):
    speed = pw.curves["speed"]
    rpm = pw.curves["rpm"]
    assert speed.color == "#D62728"
    assert speed.line_preset == LINE_PRESET_DASH
    assert speed.curve.opts["pen"].style() == Qt.PenStyle.DashLine
    assert rpm.draw_mode == DRAW_MODE_STEP_POST
    assert rpm.draw_mode_source == DRAW_MODE_SOURCE_USER
    assert rpm.symbol_policy == SYMBOL_POLICY_ON


def test_export_apply_roundtrip_restores_styles(loaded_window, qtbot):
    mw = loaded_window
    _styled_window(mw)
    cfg = mw.plot_config_manager.export_current_config(mw)
    styles = cfg.pages[0].plots[0].curve_styles
    assert styles["speed"] == {"color": "#D62728", "line_preset": LINE_PRESET_DASH}
    assert styles["rpm"]["draw_mode"] == DRAW_MODE_STEP_POST
    assert styles["rpm"]["symbol_policy"] == SYMBOL_POLICY_ON

    assert mw.plot_config_manager.apply_config(mw, cfg) is True
    qtbot.wait(100)
    settle()
    _assert_styles(_pw0(mw))


def test_auto_save_yaml_roundtrip_restores_styles(loaded_window, qtbot, tmp_path):
    mw = loaded_window
    _styled_window(mw)
    cfg = mw.plot_config_manager.export_current_config(mw)
    asm = AutoSaveManager(storage_path=tmp_path / "auto_dir")
    asm.auto_save(cfg)
    loaded = asm.load_auto_save()
    assert loaded is not None

    assert mw.plot_config_manager.apply_config(mw, loaded) is True
    qtbot.wait(100)
    settle()
    _assert_styles(_pw0(mw))


def test_repeat_apply_color_deterministic(loaded_window, qtbot):
    mw = loaded_window
    _styled_window(mw)
    cfg = mw.plot_config_manager.export_current_config(mw)
    for _ in range(2):
        assert mw.plot_config_manager.apply_config(mw, cfg) is True
        qtbot.wait(100)
        settle()
        assert _pw0(mw).curves["speed"].color == "#D62728"
        assert _pw0(mw).curves["rpm"].color == DEFAULT_CURVE_COLORS[1]


def test_legacy_config_without_styles_rotates_palette(loaded_window, qtbot):
    mw = loaded_window
    cfg = PlotSessionConfig(pages=[PlotPageConfig(
        title="图 1", layout_rows=1, layout_cols=1,
        plots=[PlotConfig(curves=["speed", "rpm"])],
    )])
    assert mw.plot_config_manager.apply_config(mw, cfg) is True
    qtbot.wait(100)
    settle()
    pw = _pw0(mw)
    assert pw.curves["speed"].color == DEFAULT_CURVE_COLORS[0]
    assert pw.curves["rpm"].color == DEFAULT_CURVE_COLORS[1]
    assert pw.curves["speed"].line_preset == "auto"


def test_empty_first_cell_keeps_position_with_styles(loaded_window, qtbot):
    mw = loaded_window
    page = mw.current_page
    page.rows_max = 4
    page.cols_max = 4
    page.ensure_matrix(1, 2)
    pw0 = page.plot_widgets[0].plot_widget
    pw1 = page.plot_widgets[1].plot_widget
    pw0.clear_plot_item(clear_invalid=True)
    assert pw1.add_variable_to_plot("speed")
    assert pw1._multi_curve_manager.apply_curve_line_preset("speed", LINE_PRESET_DASH)
    settle()

    cfg = mw.plot_config_manager.export_current_config(mw)
    assert mw.plot_config_manager.apply_config(mw, cfg) is True
    qtbot.wait(100)
    settle()
    page = mw.current_page
    assert not page.plot_widgets[0].plot_widget.curves  # 洞保持
    pw1 = page.plot_widgets[1].plot_widget
    assert "speed" in pw1.curves
    assert pw1.curves["speed"].line_preset == LINE_PRESET_DASH


def test_styles_name_missing_from_data_demotes_gray(loaded_window, qtbot):
    mw = loaded_window
    cfg = PlotSessionConfig(pages=[PlotPageConfig(
        title="图 1", layout_rows=1, layout_cols=1,
        plots=[PlotConfig(
            curves=["speed", "ghost_x"],
            curve_styles={"ghost_x": {"color": "#D62728", "line_preset": LINE_PRESET_DASH}},
        )],
    )])
    assert mw.plot_config_manager.apply_config(mw, cfg) is True
    qtbot.wait(100)
    settle()
    pw = _pw0(mw)
    assert "ghost_x" in pw.invalid_curves
    assert pw.invalid_curves["ghost_x"].color == INVALID_CURVE_COLOR


def test_hidden_page_scene_false_apply_restores_styles(loaded_window, qtbot):
    """隐藏页套用分支（scene=False）：_batch_adding 窗口内恢复外观束。"""
    mw = loaded_window
    _styled_window(mw)
    cfg = mw.plot_config_manager.export_current_config(mw)
    plot_cfg = cfg.pages[0].plots[0]
    pw = _pw0(mw)
    mgr = mw.plot_config_manager
    assert mgr._apply_plot_config(pw, plot_cfg, mw, 0, scene=False) is True
    qtbot.wait(100)
    settle()
    _assert_styles(pw)


def test_partial_bundle_mixes_palette_rotation(loaded_window, qtbot):
    """束只覆盖 speed：rpm 无束条目走色板轮转（部分束混排）。"""
    mw = loaded_window
    cfg = PlotSessionConfig(pages=[PlotPageConfig(
        title="图 1", layout_rows=1, layout_cols=1,
        plots=[PlotConfig(
            curves=["speed", "rpm"],
            curve_styles={"speed": {"color": "#D62728", "line_preset": LINE_PRESET_DASH}},
        )],
    )])
    assert mw.plot_config_manager.apply_config(mw, cfg) is True
    qtbot.wait(100)
    settle()
    pw = _pw0(mw)
    assert pw.curves["speed"].color == "#D62728"
    assert pw.curves["speed"].line_preset == LINE_PRESET_DASH
    assert pw.curves["rpm"].color == DEFAULT_CURVE_COLORS[1]
    assert pw.curves["rpm"].line_preset == "auto"


def test_reload_after_apply_keeps_styles(loaded_window, qtbot, tmp_path):
    mw = loaded_window
    _styled_window(mw)
    cfg = mw.plot_config_manager.export_current_config(mw)
    assert mw.plot_config_manager.apply_config(mw, cfg) is True
    qtbot.wait(100)
    settle()

    write_csv(
        tmp_path / "e2e_demo.csv",
        header=["time", "speed", "rpm", "flag"],
        units=["s", "km/h", "rpm", "-"],
        rows=make_simple_rows(50),
    )
    version_before = mw._data_version
    mw.file_loader_manager.reload_data()
    qtbot.waitUntil(lambda: mw._data_version > version_before, timeout=5000)
    qtbot.waitUntil(lambda: not mw._is_loading_new_data, timeout=5000)
    settle()
    _assert_styles(_pw0(mw))
