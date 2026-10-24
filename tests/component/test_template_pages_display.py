"""模板管理器：页数列与跨页统计（模板 v2）。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.core.plot_config import (
    PlotConfig,
    PlotPageConfig,
    PlotSessionConfig,
)
from src.core.template_manager import TemplateManager
from src.ui.dialogs.template_manager_dialog import TemplateManagerDialog


@pytest.fixture()
def manager(qapp, tmp_path):
    return TemplateManager(storage_path=tmp_path / "templates")


def _page(title: str, rows: int, cols: int, *curve_lists: list[str]) -> PlotPageConfig:
    return PlotPageConfig(
        title=title,
        layout_rows=rows,
        layout_cols=cols,
        plots=[PlotConfig(curves=curves) for curves in curve_lists],
    )


def _save_v2(manager, name: str, *, active_page: int = 0, page_count: int = 2):
    pages = [
        _page(f"page {index + 1}", 1, 1, [f"v{index + 1}"])
        for index in range(page_count)
    ]
    return manager.save_template(
        PlotSessionConfig(pages=pages, active_page=active_page), name
    )


def _row_named(dlg: TemplateManagerDialog, name: str) -> int:
    for row in range(dlg._table.rowCount()):
        if dlg._table.item(row, 0).text() == name:
            return row
    raise AssertionError(f"模板 {name!r} 不在列表里")


def test_table_has_page_column_and_cross_page_counts(manager):
    _save_v2(manager, "two-pages", active_page=1)
    dlg = TemplateManagerDialog(manager)
    try:
        headers = [
            dlg._table.horizontalHeaderItem(column).text()
            for column in range(dlg._table.columnCount())
        ]
        assert headers == ["名称", "页数", "变量数", "Plot 数", "更新时间"]

        row = _row_named(dlg, "two-pages")
        assert [dlg._table.item(row, col).text() for col in range(1, 4)] == [
            "2",
            "2",
            "2",
        ]
    finally:
        dlg.deleteLater()


def test_plot_count_ignores_empty_slots_and_counts_all_pages(manager):
    config = PlotSessionConfig(
        pages=[
            PlotPageConfig(
                title="page 1",
                layout_rows=2,
                layout_cols=1,
                plots=[PlotConfig(curves=["a", "b"]), PlotConfig(curves=[])],
            ),
            PlotPageConfig(
                title="page 2",
                layout_rows=1,
                layout_cols=2,
                plots=[PlotConfig(curves=["b", "c"])],
            ),
        ]
    )
    manager.save_template(config, "counts")
    dlg = TemplateManagerDialog(manager)
    try:
        row = _row_named(dlg, "counts")
        assert dlg._table.item(row, 1).text() == "2"  # 页数
        assert dlg._table.item(row, 2).text() == "3"  # 变量去重 a/b/c
        assert dlg._table.item(row, 3).text() == "2"  # 非空子图，跨两页
    finally:
        dlg.deleteLater()


def test_v1_template_still_reports_one_page(manager):
    manager.save_template(
        PlotSessionConfig(plots=[PlotConfig(curves=["speed"])]), "legacy"
    )
    dlg = TemplateManagerDialog(manager)
    try:
        row = _row_named(dlg, "legacy")
        assert dlg._table.item(row, 1).text() == "1"
        assert dlg._table.item(row, 2).text() == "1"
        assert dlg._table.item(row, 3).text() == "1"
    finally:
        dlg.deleteLater()


def test_details_show_active_page_and_page_summaries(manager):
    _save_v2(manager, "detail", active_page=1, page_count=2)
    dlg = TemplateManagerDialog(manager)
    try:
        row = _row_named(dlg, "detail")
        dlg._table.selectRow(row)
        text = dlg._details_label.text()
        assert text.startswith("选中: detail")
        assert "页数: 2（当前页: page 2）" in text
        assert "1. page 1" in text
        assert "2. page 2" in text
    finally:
        dlg.deleteLater()


def test_active_page_is_clamped_and_safe_for_bad_values(manager):
    dlg = TemplateManagerDialog(manager)
    try:
        template = SimpleNamespace(
            metadata=SimpleNamespace(
                name="demo",
                description="",
                created_at="2026-01-01T00:00:00",
                updated_at="2026-01-01T00:00:00",
            ),
            config={
                "pages": [
                    {"title": "page 1", "layout_rows": 1, "layout_cols": 1},
                    {"title": "page 2", "layout_rows": 1, "layout_cols": 1},
                ],
                "active_page": "not-a-number",
            },
        )
        text = dlg._details_text(template)
        assert "页数: 2（当前页: page 1）" in text

        template.config["active_page"] = 99
        text = dlg._details_text(template)
        assert "页数: 2（当前页: page 2）" in text
    finally:
        dlg.deleteLater()


def test_details_truncate_long_page_list(manager):
    _save_v2(manager, "many-pages", active_page=0, page_count=7)
    dlg = TemplateManagerDialog(manager)
    try:
        row = _row_named(dlg, "many-pages")
        dlg._table.selectRow(row)
        text = dlg._details_label.text()
        assert "… 共 7 页" in text
        assert "7. page 7" not in text, "只列前 6 页"
    finally:
        dlg.deleteLater()
