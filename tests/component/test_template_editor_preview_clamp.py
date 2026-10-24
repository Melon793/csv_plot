"""模板编辑器预览的规模钳制（V6.0 P0-7 防回归）。

YAML 编辑区是自由文本，`_update_preview` 挂在 `textChanged` 上逐键触发。
旧实现对 `layout_rows × layout_cols` 无任何上限，用户多打一个 0 就会同步
创建百万级 QFrame/QLabel 冻结 UI 线程（甚至 OOM），只能强杀进程 —— 保存
路径的 PlotSessionConfig 校验发生在预览之后，兜不住这条路径。
"""

import pytest

from src.core.template_manager import TemplateManager
from src.ui.dialogs.template_editor_dialog import TemplateEditorDialog


@pytest.fixture()
def editor(qapp, tmp_path):
    dlg = TemplateEditorDialog(TemplateManager(tmp_path / "templates"))
    yield dlg
    dlg.deleteLater()


def _cell_count(dlg):
    return dlg._preview_layout.count()


def test_normal_grid_still_renders(editor):
    """阈值内的常规布局照常渲染（4 格）"""
    editor._yaml_edit.setPlainText(
        "layout_rows: 2\nlayout_cols: 2\nplots: []\n"
    )
    assert _cell_count(editor) == 4
    assert "cells: 2 × 2" in editor._stats_label.text()


def test_oversized_grid_is_not_rendered(editor):
    """超限布局：不创建任何控件，只降级提示（旧实现会同步建 100 万个控件）"""
    editor._yaml_edit.setPlainText(
        "layout_rows: 2\nlayout_cols: 2\nplots: []\n"
    )
    assert _cell_count(editor) == 4

    editor._yaml_edit.setPlainText(
        "layout_rows: 1000\nlayout_cols: 1000\nplots: []\n"
    )
    assert _cell_count(editor) == 0
    assert "布局过大或非法" in editor._stats_label.text()


def test_grid_barely_over_limit_is_rejected(editor):
    """刚好超过 PREVIEW_MAX_CELLS 即拒绝，阈值边界明确"""
    over = TemplateEditorDialog.PREVIEW_MAX_CELLS + 1
    editor._yaml_edit.setPlainText(
        f"layout_rows: {over}\nlayout_cols: 1\nplots: []\n"
    )
    assert _cell_count(editor) == 0

    editor._yaml_edit.setPlainText(
        f"layout_rows: {TemplateEditorDialog.PREVIEW_MAX_CELLS}\n"
        "layout_cols: 1\nplots: []\n"
    )
    assert _cell_count(editor) == TemplateEditorDialog.PREVIEW_MAX_CELLS


@pytest.mark.parametrize(
    "yaml_text",
    [
        "layout_rows: abc\nlayout_cols: 2\nplots: []\n",
        "layout_rows: 0\nlayout_cols: 2\nplots: []\n",
        "layout_rows: -3\nlayout_cols: 2\nplots: []\n",
        "layout_rows: 100000000000000000000\nlayout_cols: 2\nplots: []\n",
    ],
)
def test_illegal_grid_values_are_rejected_without_exception(editor, yaml_text):
    """非法/极端取值一律降级提示，不抛异常也不渲染控件"""
    editor._yaml_edit.setPlainText(yaml_text)
    assert _cell_count(editor) == 0
    assert "布局过大或非法" in editor._stats_label.text()


def test_v2_oversized_page_only_blocks_that_page(editor):
    """多页 YAML：只渲染当前页；切到超限页才清空并提示。"""
    editor._yaml_edit.setPlainText(
        "pages:\n"
        "  - title: page 1\n"
        "    layout_rows: 2\n"
        "    layout_cols: 2\n"
        "    plots: []\n"
        "  - title: page 2\n"
        "    layout_rows: 1000\n"
        "    layout_cols: 1000\n"
        "    plots: []\n"
    )
    assert editor._page_combo.count() == 2
    assert _cell_count(editor) == 4
    assert "cells: 2 × 2" in editor._stats_label.text()

    editor._page_combo.setCurrentIndex(1)
    assert _cell_count(editor) == 0
    assert "第 2 页" in editor._stats_label.text()
    assert "布局过大或非法" in editor._stats_label.text()

    editor._page_combo.setCurrentIndex(0)
    assert _cell_count(editor) == 4


def test_non_dict_page_entry_falls_back_without_crash(editor):
    """pages 里只有 None/非 dict 页：与 from_dict 一样回退单页，不崩。"""
    editor._yaml_edit.setPlainText("pages:\n  - null\n")
    assert editor._page_combo.count() == 1
    assert _cell_count(editor) == 1
    assert "pages: 1" in editor._stats_label.text()
