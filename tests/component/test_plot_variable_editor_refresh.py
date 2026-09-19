"""绘图变量编辑器 reload 后就地刷新数据源（V6.0 P1-2 防回归）。

对话框在 __init__ 里 setWindowFlag(Qt.WindowType.Tool) → 它是顶层窗口，
`self.window()` 返回**自身**而非主窗口，因此旧实现里 `getattr(main_window,
"loader", None)` 恒为 None，整条「reload 后就地更新搜索栏」链路静默空转
（调用侧 file_loader_manager 用 findChildren + try/except debug 掩盖了失败）。
正确写法与同文件 setup_ui 一致：经 plot_widget 取宿主窗口。
"""

import types

import pytest

from src.ui.plot_variable_editor import PlotVariableEditorDialog


@pytest.fixture()
def editor_with_host(plot_factory, qapp):
    pw = plot_factory()
    host = pw.window()
    host.loader = types.SimpleNamespace(var_names=["a", "b"], units={})
    host.data_validity = {}
    dlg = PlotVariableEditorDialog(pw)
    qapp.processEvents()
    return dlg, host, pw


def test_dialog_is_top_level_tool_window(editor_with_host):
    """前提固定：对话框自身即顶层窗口，self.window() 取不到主窗口"""
    dlg, _, _ = editor_with_host
    assert dlg.window() is dlg
    assert dlg.search_bar is not None


def test_setup_ui_seeds_search_bar_from_host_window(editor_with_host):
    dlg, host, _ = editor_with_host
    assert dlg.search_bar._all_var_names == ["a", "b"]


def test_refresh_data_source_picks_up_reloaded_variables(editor_with_host):
    """reload 后主窗口 loader 换列 → 已打开的编辑器搜索栏必须同步"""
    dlg, host, _ = editor_with_host

    host.loader = types.SimpleNamespace(var_names=["x", "y", "z"], units={})
    host.data_validity = {"x": 1, "y": 0, "z": -2}

    dlg.refresh_data_source()

    assert dlg.search_bar._all_var_names == ["x", "y", "z"]
    assert dlg.search_bar._validity == {"x": 1, "y": 0, "z": -2}
