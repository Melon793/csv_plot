"""第 5 轮批次 4 回归：模板格子坐标跨 stride + 畸形模板容错（真实 MainWindow）。

P1-7：cell_index 是导出端 built stride 的扁平索引；扩后缩/页 1 满容量建矩阵/
页重排会让目标页 stride 不同 → 错位/越界丢弃。row/col 与 stride 无关。
P1-8：手写/损坏模板的非法 int 字段不得让异常逃出 Qt 槽。
"""
from __future__ import annotations

import types

import pytest

from src.core.plot_config import PlotConfig, PlotPageConfig, PlotSessionConfig
from src.ui.plot_config_manager import PlotConfigManager


def _stub_container(var_names):
    return types.SimpleNamespace(
        isHidden=lambda: False,
        plot_widget=types.SimpleNamespace(
            curve_strategy=types.SimpleNamespace(
                get_curve_names=lambda: list(var_names),
                get_invalid_curve_names=lambda: [],
            )
        ),
    )


def test_cell_row_col_survives_target_stride_change(loaded_window, qtbot):
    """导出端 built stride=3（2×3 缩回可见 2×2），目标页 stride=4：
    曲线必须落在目标页的 (1,0)/(1,1)，而不是按旧扁平索引错位/丢弃。"""
    mw = loaded_window
    page = mw.current_page
    # e2e 环境页容量上限默认为 1，显式放开以构造多格网格
    page.rows_max = 4
    page.cols_max = 4
    # 目标页 built stride 固定为 4（只扩不缩），与导出端 3 不同
    page.ensure_matrix(2, 4)
    stride = page.grid_cols
    assert stride == 4

    # 导出端替身页：built 2×3（stride 3），可见 2×2，曲线在 (1,0)=3、(1,1)=4
    src = types.SimpleNamespace(
        title="p", rows_current=2, cols_current=2, grid_cols=3,
        row_height_factors={},
        plot_widgets=[
            _stub_container([]), _stub_container([]), _stub_container([]),
            _stub_container(["speed"]), _stub_container(["rpm"]),
            _stub_container([]),
        ],
        cell_visible=lambda i: i in {0, 1, 3, 4},
    )
    page_cfg = PlotConfigManager._export_page(src)
    assert [(p.curves[0], p.cell_index, p.cell_row, p.cell_col)
            for p in page_cfg.plots] == [
        ("speed", 3, 1, 0), ("rpm", 4, 1, 1),
    ]
    session = PlotSessionConfig(pages=[page_cfg])

    assert mw.plot_config_manager.apply_config(mw, session) is True
    qtbot.wait(100)
    # 新口径：row*stride+col → 4/5；旧口径 cell_index 3/4 会丢弃/错位
    # （3→(0,3) 不可见被丢弃，4→(1,0) 错位）
    assert "speed" in page.plot_widgets[1 * stride + 0].plot_widget.curves
    assert "rpm" in page.plot_widgets[1 * stride + 1].plot_widget.curves
    assert "speed" not in page.plot_widgets[3].plot_widget.curves


def test_legacy_cell_index_only_config_still_applies(loaded_window, qtbot):
    """旧档零回归：只有 cell_index（无 row/col）时沿用旧落位口径。"""
    mw = loaded_window
    page = mw.current_page
    page.rows_max = 4
    page.cols_max = 4
    page.ensure_matrix(2, 2)
    stride = page.grid_cols
    page_cfg = PlotPageConfig(
        title="legacy", layout_rows=2, layout_cols=2,
        plots=[PlotConfig(curves=["speed"], cell_index=stride + 1)],
    )
    session = PlotSessionConfig(pages=[page_cfg])
    assert mw.plot_config_manager.apply_config(mw, session) is True
    qtbot.wait(100)
    assert "speed" in page.plot_widgets[stride + 1].plot_widget.curves


def test_malformed_template_config_warns_instead_of_raising(loaded_window, monkeypatch):
    """P1-8：非法 layout_rows 的模板应用时弹警告，异常不得逃出槽。"""
    mw = loaded_window
    warnings = []
    monkeypatch.setattr(
        "src.ui.main_window.QMessageBox.warning",
        staticmethod(lambda *a, **k: warnings.append(a)),
    )
    bad = types.SimpleNamespace(
        config={"pages": [{"layout_rows": "abc", "plots": []}]},
        metadata=types.SimpleNamespace(name="bad", id="bad", description=""),
    )
    mw._check_and_apply_template(bad, "bad", "bad")
    assert warnings, "畸形模板必须降级为警告"

    # apply_template 入口同口径：解析失败返回 False
    monkeypatch.setattr(
        mw.plot_config_manager._template_manager, "get_template",
        lambda tid: bad,
    )
    assert mw.plot_config_manager.apply_template(mw, "bad") is False


def test_non_dict_plots_and_null_curves_do_not_escape_slot(loaded_window, monkeypatch):
    """评审 B1：plots: ["speed"]（AttributeError 族）与 curves: null
    （try 外 TypeError 族）都不得逃出应用槽。"""
    mw = loaded_window
    warnings = []
    monkeypatch.setattr(
        "src.ui.main_window.QMessageBox.warning",
        staticmethod(lambda *a, **k: warnings.append(a)),
    )
    # 非 dict plot 被过滤 → 空模板匹配度 1.0 → 直接 apply（清空可见格），
    # 不走低匹配对话框；靶点是「不抛异常且不误报损坏」
    monkeypatch.setattr(mw, "_show_match_low_dialog", lambda *a, **k: None)
    bad = types.SimpleNamespace(
        config={"pages": [{"layout_rows": 1, "layout_cols": 1, "plots": ["speed"]}]},
        metadata=types.SimpleNamespace(name="bad2", id="bad2", description=""),
    )
    mw._check_and_apply_template(bad, "bad2", "bad2")
    assert not warnings, "容错跳过不得误报模板损坏"
    # curves: null 解析为空列表，check_template_match 不得抛 TypeError
    bad3 = types.SimpleNamespace(
        config={"pages": [{"plots": [{"curves": None}]}]},
        metadata=types.SimpleNamespace(name="bad3", id="bad3", description=""),
    )
    monkeypatch.setattr(
        mw.plot_config_manager._template_manager, "get_template",
        lambda tid: bad3,
    )
    # 靶点是「异常不逃槽」：空曲线模板应用结果可为 False（无曲线可套），
    # 但必须正常返回布尔而非抛 TypeError
    result = mw.plot_config_manager.apply_template(mw, "bad3")
    assert isinstance(result, bool)


@pytest.mark.parametrize(
    "plot_config",
    [
        PlotConfig(curves=["speed"], cell_index=99),
        PlotConfig(curves=["speed"], cell_index=-1),
        PlotConfig(curves=["speed"], cell_row=0, cell_col=3),
    ],
)
def test_all_targets_out_of_bounds_keeps_old_plot(
    loaded_window, monkeypatch, plot_config
):
    mw = loaded_window
    page = mw.current_page
    page.rows_max = 4
    page.cols_max = 4
    page.ensure_matrix(2, 2)
    widget = page.plot_widgets[0].plot_widget
    assert widget.plot_variable("speed") is True
    clears = []
    original_clear = widget.clear_plot_item

    def record_clear(*args, **kwargs):
        clears.append(True)
        return original_clear(*args, **kwargs)

    monkeypatch.setattr(widget, "clear_plot_item", record_clear)
    session = PlotSessionConfig(pages=[PlotPageConfig(
        layout_rows=1, layout_cols=1, plots=[plot_config],
    )])
    assert mw.plot_config_manager.apply_config(mw, session) is False
    assert not clears, "全越界必须在清场前失败"
    assert "speed" in widget.curves


def test_col_beyond_target_capacity_is_skipped_not_wrapped(loaded_window, qtbot):
    """评审 B2：目标机 cols_max=3 < 模板 4 列时，(0,3) 不得回绕到 (1,0)
    覆盖本该落在那里的曲线；越界显式跳过，合法 (1,0) 正常落位。"""
    mw = loaded_window
    page = mw.current_page
    page.rows_max = 4
    page.cols_max = 3
    page.ensure_matrix(2, 3)
    stride = page.grid_cols
    assert stride == 3

    # 导出端 2×4（stride 4）：speed 在 (0,3)=3，rpm 在 (1,1)=5。
    # rpm 刻意不放 (1,0)：否则修复前 speed 回绕到 (1,0) 会被后导出的 rpm
    # clear+覆盖，终态与修复后相同，用例失去判别力（复审 BL-2）。
    src = types.SimpleNamespace(
        title="p", rows_current=2, cols_current=4, grid_cols=4,
        row_height_factors={},
        plot_widgets=[
            _stub_container([]), _stub_container([]), _stub_container([]),
            _stub_container(["speed"]), _stub_container([]),
            _stub_container(["rpm"]), _stub_container([]), _stub_container([]),
        ],
        cell_visible=lambda i: i in {0, 1, 2, 3, 4, 5, 6, 7},
    )
    page_cfg = PlotConfigManager._export_page(src)
    session = PlotSessionConfig(pages=[page_cfg])
    assert mw.plot_config_manager.apply_config(mw, session) is True
    qtbot.wait(100)
    assert "rpm" in page.plot_widgets[1 * stride + 1].plot_widget.curves
    all_curves = [
        name
        for c in page.plot_widgets
        for name in c.plot_widget.curves
    ]
    assert "speed" not in all_curves, "越界 col 必须跳过，不得回绕错落到 (1,0)"