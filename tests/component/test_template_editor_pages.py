"""模板编辑器：v2 多页 YAML、按页预览与保存往返。"""

from __future__ import annotations

import yaml
import pytest
from PySide6.QtWidgets import QMessageBox

from src.core.plot_config import (
    PlotConfig,
    PlotPageConfig,
    PlotSessionConfig,
)
from src.core.template_manager import TemplateManager
from src.ui.dialogs.template_editor_dialog import TemplateEditorDialog


@pytest.fixture()
def manager(qapp, tmp_path):
    return TemplateManager(storage_path=tmp_path / "templates")


@pytest.fixture()
def boxes(monkeypatch):
    """记录 warning/critical 文案，避免真弹窗阻塞。"""
    shown: list[tuple[str, str]] = []

    def fake(parent, title, text, *args, **kwargs):
        shown.append((title, text))
        return QMessageBox.StandardButton.Ok

    for kind in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, kind, staticmethod(fake))
    return shown


def _two_page_config() -> PlotSessionConfig:
    return PlotSessionConfig(
        time_factor=2.0,
        time_offset=1.0,
        pages=[
            PlotPageConfig(
                title="page 1",
                layout_rows=2,
                layout_cols=2,
                row_height_factors={0: 40},
                plots=[
                    PlotConfig(curves=["speed", "rpm"]),
                    PlotConfig(curves=["throttle"]),
                ],
            ),
            PlotPageConfig(
                title="对比页: #1\nline2",
                layout_rows=1,
                layout_cols=2,
                plots=[PlotConfig(curves=["rpm"])],
            ),
        ],
        active_page=1,
        cursor_mode="2 anchored cursor",
        saved_mark_range=[1.0, 2.0],
    )


def _editor(manager, config=None, edit_template_id=None) -> TemplateEditorDialog:
    return TemplateEditorDialog(
        manager,
        current_config=config,
        edit_template_id=edit_template_id,
    )


def _add_and_save(editor: TemplateEditorDialog, name: str) -> None:
    editor._name_edit.setText(name)
    editor._save_btn.click()


def test_yaml_keeps_all_pages_and_view_state(manager):
    editor = _editor(manager, _two_page_config())
    try:
        payload = yaml.safe_load(editor._yaml_edit.toPlainText())
        assert "layout_rows" not in payload, "v2 编辑只写 pages，避免两份真相"
        assert "plots" not in payload
        assert payload["active_page"] == 1
        assert payload["time_factor"] == 2.0
        assert payload["time_offset"] == 1.0
        assert payload["cursor_mode"] == "2 anchored cursor"
        assert payload["saved_mark_range"] == [1.0, 2.0]

        pages = payload["pages"]
        assert [page["title"] for page in pages] == [
            "page 1",
            "对比页: #1\nline2",
        ]
        assert pages[0]["row_height_factors"] == {0: 40}
        assert pages[0]["plots"][0]["curves"] == ["speed", "rpm"]
        assert pages[1]["plots"][0]["curves"] == ["rpm"]
    finally:
        editor.deleteLater()


def test_preview_combo_defaults_to_active_page_and_switches(manager):
    editor = _editor(manager, _two_page_config())
    try:
        assert editor._page_combo.count() == 2
        assert editor._page_combo.currentIndex() == 1
        assert "pages: 2" in editor._stats_label.text()
        assert "当前 对比页: #1 line2" in editor._stats_label.text()
        assert "cells: 1 × 2" in editor._stats_label.text()
        assert editor._preview_layout.count() == 2

        editor._page_combo.setCurrentIndex(0)
        assert editor._preview_layout.count() == 4
        assert "cells: 2 × 2" in editor._stats_label.text()
        assert "当前 page 1" in editor._stats_label.text()

        # 页下拉必须在 _preview_layout 之外：清预览只清下拉项，不删控件
        assert editor._preview_widget.isAncestorOf(editor._page_combo) is False
        editor._clear_preview()
        assert editor._page_combo.count() == 0
        assert editor._page_combo.parent() is not None
    finally:
        editor.deleteLater()


def test_typing_preserves_selected_page(manager, qapp):
    editor = _editor(manager, _two_page_config())
    try:
        editor._page_combo.setCurrentIndex(0)
        editor._yaml_edit.append("# edit")
        qapp.processEvents()
        assert editor._page_combo.currentIndex() == 0
        assert editor._preview_layout.count() == 4
    finally:
        editor.deleteLater()


def test_save_preserves_all_pages_and_optional_state(manager):
    editor = _editor(manager, _two_page_config())
    try:
        _add_and_save(editor, "multi-page")
        saved = next(
            template
            for template in manager.get_all_templates()
            if template.metadata.name == "multi-page"
        )
        assert len(saved.config["pages"]) == 2
        assert [page["title"] for page in saved.config["pages"]] == [
            "page 1",
            "对比页: #1\nline2",
        ]
        assert saved.config["active_page"] == 1
        assert saved.config["pages"][0]["row_height_factors"] == {0: 40}
        assert saved.config["cursor_mode"] == "2 anchored cursor"
        assert saved.config["saved_mark_range"] == [1.0, 2.0]
        # to_dict 仍保留 v1 回退视图（旧版软件可打开）
        assert saved.config["plots"] == saved.config["pages"][0]["plots"]
    finally:
        editor.deleteLater()


def test_edit_existing_v2_template_shows_pages(manager):
    template = manager.save_template(_two_page_config(), "existing")
    editor = _editor(manager, edit_template_id=template.metadata.id)
    try:
        assert editor._page_combo.count() == 2
        assert editor._page_combo.currentIndex() == 1
        assert "pages:" in editor._yaml_edit.toPlainText()
        _add_and_save(editor, "existing-renamed")
        renamed = next(
            item
            for item in manager.get_all_templates()
            if item.metadata.name == "existing-renamed"
        )
        assert len(renamed.config["pages"]) == 2
        assert renamed.config["active_page"] == 1
    finally:
        editor.deleteLater()


def test_blank_template_saves_as_single_page_v2(manager):
    editor = _editor(manager)
    try:
        _add_and_save(editor, "blank")
        saved = next(
            template
            for template in manager.get_all_templates()
            if template.metadata.name == "blank"
        )
        assert len(saved.config["pages"]) == 1
        assert saved.config["active_page"] == 0
        assert len(saved.config["pages"][0]["plots"]) == 4
    finally:
        editor.deleteLater()


def test_save_rejects_missing_layout_and_invalid_pages(manager, boxes):
    editor = _editor(manager)
    try:
        editor._name_edit.setText("bad-missing")
        editor._yaml_edit.setPlainText("time_factor: 1\n")
        editor._save_btn.click()
        assert boxes[-1][1].startswith("配置缺少布局信息")
        assert not manager.exists("bad-missing")

        editor._name_edit.setText("bad-empty-pages")
        editor._yaml_edit.setPlainText("pages: []\n")
        editor._save_btn.click()
        assert boxes[-1][1].startswith("配置缺少布局信息")

        editor._name_edit.setText("bad-layout")
        editor._yaml_edit.setPlainText(
            "pages:\n"
            "  - title: p1\n"
            "    layout_rows: abc\n"
            "    layout_cols: 1\n"
            "    plots: []\n"
        )
        editor._save_btn.click()
        assert "第 1 页 layout_rows 必须是正整数" in boxes[-1][1]

        editor._name_edit.setText("bad-page-type")
        editor._yaml_edit.setPlainText("pages:\n  - just-a-string\n")
        editor._save_btn.click()
        assert "第 1 页配置不是字典" in boxes[-1][1]
    finally:
        editor.deleteLater()


def test_save_clamps_out_of_range_active_page(manager):
    config = _two_page_config()
    config.active_page = 99
    editor = _editor(manager, config)
    try:
        _add_and_save(editor, "clamped")
        saved = next(
            template
            for template in manager.get_all_templates()
            if template.metadata.name == "clamped"
        )
        assert saved.config["active_page"] == 1
    finally:
        editor.deleteLater()


def _placeholder_config() -> PlotSessionConfig:
    """一页两格：一格真曲线 + 占位，一格纯占位。"""
    return PlotSessionConfig(
        pages=[
            PlotPageConfig(
                title="page 1",
                layout_rows=1,
                layout_cols=2,
                plots=[
                    PlotConfig(curves=["speed"], invalid_curves=["ghost"]),
                    PlotConfig(curves=[], invalid_curves=["only_placeholder"]),
                ],
            )
        ]
    )


def test_yaml_keeps_invalid_curve_placeholders(manager):
    """失效占位是布局资产：编辑器重新生成的 YAML 必须带上 invalid_curves。

    回归点：生成器只写 curves 时，「编辑模板 → 保存」会静默清空全部占位，
    而编辑器预览/统计全程不显示占位，用户无从察觉。
    """
    editor = _editor(manager, _placeholder_config())
    try:
        text = editor._yaml_edit.toPlainText()
        assert "invalid_curves" in text
        plots = yaml.safe_load(text)["pages"][0]["plots"]
        assert plots[0]["curves"] == ["speed"]
        assert plots[0]["invalid_curves"] == ["ghost"]
        # 纯占位格子不得因 curves 为空提前 continue 而丢掉占位
        assert plots[1]["curves"] == []
        assert plots[1]["invalid_curves"] == ["only_placeholder"]
        # 解析回来的配置能重建出同样的占位（读侧 from_dict 一直支持）
        rebuilt = PlotSessionConfig.from_dict(yaml.safe_load(text))
        assert [
            plot.invalid_curves
            for plot in rebuilt.normalized_pages()[0].plots
        ] == [["ghost"], ["only_placeholder"]]
    finally:
        editor.deleteLater()


def test_edit_and_save_without_edits_preserves_placeholders(manager, boxes):
    """完整事故路径：存含占位模板 → 编辑器打开 → 一字不改直接保存 → 占位仍在。"""
    template = manager.save_template(_placeholder_config(), "with-placeholder")
    editor = _editor(manager, edit_template_id=template.metadata.id)
    try:
        _add_and_save(editor, "with-placeholder")
    finally:
        editor.deleteLater()
    assert not boxes, f"保存不应弹窗：{boxes}"
    saved = next(
        item
        for item in manager.get_all_templates()
        if item.metadata.name == "with-placeholder"
    )
    plots = saved.config["pages"][0]["plots"]
    assert plots[0]["curves"] == ["speed"]
    assert plots[0]["invalid_curves"] == ["ghost"]
    assert plots[1]["invalid_curves"] == ["only_placeholder"]
