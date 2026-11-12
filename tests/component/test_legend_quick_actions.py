"""图例快速操作（快速改色 / 中键删除）component 测试（施工方案 §5）。

分步落地：
- Step 2：apply_curve_color 下沉 MultiCurveManager（编辑器改薄壳委托）；
- Step 3a：图例锚点拆分 color:/// + 调色板菜单；
- Step 3b：中键单击删除（含 view 层双击护栏）。

复用 plot_factory（tests/component/conftest.py）与右键菜单测试的
FakeMenu 替身模式（见 test_legend_context_menu.py 模块注释）。
"""

import types

import numpy as np
import pytest

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QColorDialog

from src.ui.plot_variable_editor import PlotVariableEditorDialog

RED_NAME = QColor("red").name()  # "#ff0000"


# ---------------------------------------------------------------------------
# Step 2：apply_curve_color 下沉（刷新链与迁移前逐项一致，T14）
# ---------------------------------------------------------------------------

def test_apply_curve_color_updates_pen_and_refreshes(plot_factory):
    pw = plot_factory()
    assert pw.plot_variable("a")
    ci = pw.curves["a"]
    old_width = ci.curve.opts.get("pen").widthF()
    ci.curve._cached_pen_key = "thin_blue"  # 预置缓存：改色后必须被清除

    legend_calls = []
    orig_update = pw.update_legend_label
    pw.update_legend_label = lambda html: legend_calls.append(html)

    assert pw._multi_curve_manager.apply_curve_color("a", "red") is True

    assert ci.color == "red"
    pen = ci.curve.opts.get("pen")
    assert pen.color().name() == RED_NAME
    assert pen.widthF() == old_width, "改色必须保留原线宽"
    assert not hasattr(ci.curve, "_cached_pen_key"), "缓存 pen key 应被清除"
    assert legend_calls, "legend 应重建"


def test_apply_curve_color_updates_symbols(plot_factory):
    pw = plot_factory()
    assert pw.plot_variable("a")
    ci = pw.curves["a"]
    ci.curve._has_symbols = True

    assert pw._multi_curve_manager.apply_curve_color("a", "red") is True

    assert ci.curve.opts.get("symbolPen").color().name() == RED_NAME
    assert ci.curve.opts.get("symbolBrush").color().name() == RED_NAME


def test_apply_curve_color_missing_var_returns_false(plot_factory):
    pw = plot_factory()
    assert pw.plot_variable("a")
    assert pw._multi_curve_manager.apply_curve_color("ghost", "red") is False
    assert pw.curves["a"].color != "red"


def test_cursor_refresh_chain_runs(plot_factory):
    """游标刷新链：清游标项 + 重置节流时间戳（vline 可见时刷标签）。"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    pw._last_cursor_update_time = 123456.0

    cleared = []
    orig_clear = pw._clear_cursor_items
    pw._clear_cursor_items = lambda: (cleared.append(1), orig_clear())[0]

    assert pw._multi_curve_manager.apply_curve_color("a", "red") is True
    assert cleared == [1]
    assert pw._last_cursor_update_time == 0


# ---------------------------------------------------------------------------
# Step 2：编辑器薄壳委托（行为零变化，表格色块按返回值刷新）
# ---------------------------------------------------------------------------

@pytest.fixture()
def editor_env(plot_factory, qapp):
    """带宿主 loader 的编辑器对话框（对齐 test_plot_variable_editor_refresh 夹具）"""
    pw = plot_factory()
    assert pw.plot_variable("a")
    host = pw.window()
    host.loader = types.SimpleNamespace(var_names=["a"], units={})
    host.data_validity = {}
    dlg = PlotVariableEditorDialog(pw)
    qapp.processEvents()
    yield dlg, pw
    dlg.deleteLater()


def test_editor_apply_color_delegates_to_manager(editor_env):
    dlg, pw = editor_env
    assert dlg._apply_color_to_curve("a", "red") is True
    assert pw.curves["a"].color == "red"
    # 不存在的变量：返回 False，无任何状态变化
    assert dlg._apply_color_to_curve("ghost", "red") is False
    assert "ghost" not in pw.curves


def test_editor_set_variable_color_updates_table_on_success(
    editor_env, monkeypatch
):
    dlg, pw = editor_env
    dlg.var_table.selectRow(0)
    var_item = dlg.var_table.item(0, 1)
    assert var_item is not None
    assert var_item.data(Qt.ItemDataRole.UserRole) == "a"

    monkeypatch.setattr(
        QColorDialog, "getColor", staticmethod(lambda *a, **k: QColor("#123456"))
    )
    dlg.set_variable_color()

    assert pw.curves["a"].color == "#123456"
    color_widget = dlg.var_table.cellWidget(0, 2)
    assert color_widget is not None
    assert "#123456" in color_widget.styleSheet()

    # 无效颜色（isValid 为 False）：颜色与表格样式均不变
    monkeypatch.setattr(
        QColorDialog, "getColor", staticmethod(lambda *a, **k: QColor())
    )
    dlg.set_variable_color()
    assert pw.curves["a"].color == "#123456"
