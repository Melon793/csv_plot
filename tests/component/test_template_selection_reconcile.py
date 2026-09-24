"""模板管理器：列表重建后选中态必须回到视图上（删除/过滤/复制三条同源表现）。

`_refresh_template_list()` 只换 item 不动 selectionModel，Qt 的选中又按行号存，
所以重建时 `itemSelectionChanged` 一次都不会发。高亮、`_selected_template_id`、
按钮 enabled 三份状态因此会各说各话：
  1. 删掉首行 → 高亮留在行 0（装的已是第二个模板），id 却被清成 None，按钮还亮着
     → 用户看到"亮着但点不动"，必须点别处再点回来才能连删。
  2. 搜索把选中项滤掉 → id 指向视图里根本不存在的模板，按钮亮着 → 能删掉看不见的模板。
  3. 复制把副本插到行 0 → 高亮在副本上，id 还是原模板 → 后续动作作用在没高亮的行。
收口点是 `_reconcile_selection_with_view()`：id 还在视图里就让高亮跟着 id 挪，
不在了就清空选中并置灰。
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

import src.core.template_models as tm
from src.core.plot_config import PlotConfig, PlotSessionConfig
from src.core.template_manager import TemplateManager
from src.ui.dialogs import template_manager_dialog as tmd
from src.ui.dialogs.template_manager_dialog import TemplateManagerDialog

YES = QMessageBox.StandardButton.Yes
NO = QMessageBox.StandardButton.No


def _stub_msgbox(monkeypatch, answer=YES):
    """确认框在测试里不该真弹；替身只负责给答案。"""

    class _Fake:
        StandardButton = QMessageBox.StandardButton

    _Fake.question = staticmethod(lambda *a, **k: answer)
    _Fake.warning = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok)
    _Fake.information = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok)
    _Fake.critical = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(tmd, "QMessageBox", _Fake)
    return _Fake


def _ids_in_view(dlg) -> list[str | None]:
    return [
        dlg._table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        if dlg._table.item(row, 0)
        else None
        for row in range(dlg._table.rowCount())
    ]


def _selected_rows(dlg) -> list[int]:
    return [i.row() for i in dlg._table.selectionModel().selectedRows()]


def _gated_buttons(dlg) -> tuple:
    return (
        dlg._load_btn,
        dlg._edit_btn,
        dlg._duplicate_btn,
        dlg._delete_btn,
        dlg._export_btn,
    )


@pytest.fixture()
def three_templates(qapp, tmp_path, monkeypatch):
    """updated_at 倒序 → 视图行序为 c, b, a（最近更新的在顶部）。

    Windows 上 datetime.now() 的粒度约 15.6 ms，三次连续保存会盖出同一个
    updated_at；sorted(reverse=True) 是稳定排序，并列时退回写入顺序 a/b/c，
    行序整个翻转。把时钟换成每次 now() 前进一秒的替身，"谁更晚"就与平台无关。
    """

    class _TickingDatetime(datetime):
        tick = 0

        @classmethod
        def now(cls, tz=None):
            cls.tick += 1
            return datetime(2026, 1, 1) + timedelta(seconds=cls.tick)

    monkeypatch.setattr(tm, "datetime", _TickingDatetime)
    manager = TemplateManager(storage_path=tmp_path / "templates")
    ids = {}
    for name in ("a-top", "b-second", "c-third"):
        tpl = manager.save_template(
            PlotSessionConfig(plots=[PlotConfig(curves=["a"])]), name
        )
        ids[name] = tpl.metadata.id
    dlg = TemplateManagerDialog(manager)
    return dlg, manager, ids


class TestDeleteTopRow:
    """用户报的那一条：删掉顶部模板后第二次删除没反应。"""

    def test_delete_leaves_no_highlight_behind(self, three_templates, monkeypatch):
        dlg, _, ids = three_templates
        _stub_msgbox(monkeypatch)
        dlg._table.selectRow(0)

        dlg._on_delete_clicked()

        assert _selected_rows(dlg) == [], (
            "第二个模板升上来占住行 0，但高亮必须跟着被删的那个模板一起消失，"
            "不能留在一个没人选中的行上"
        )
        assert dlg._selected_template_id is None
        assert dlg._details_label.text().startswith("选中: -"), "详情不该还挂着已删模板"

    def test_buttons_grey_instead_of_lying_clickable(
        self, three_templates, monkeypatch
    ):
        """置灰才是这句"没反应"的正确表达：按钮亮着却什么都不做是在骗人。"""
        dlg, _, ids = three_templates
        _stub_msgbox(monkeypatch)
        dlg._table.selectRow(0)
        assert all(b.isEnabled() for b in _gated_buttons(dlg))

        dlg._on_delete_clicked()

        assert not any(b.isEnabled() for b in _gated_buttons(dlg)), (
            "删除后没有选中项，五个动作必须一起置灰"
        )

    def test_second_delete_cannot_eat_the_promoted_row(
        self, three_templates, monkeypatch
    ):
        dlg, manager, ids = three_templates
        _stub_msgbox(monkeypatch)
        dlg._table.selectRow(0)
        dlg._on_delete_clicked()
        survivor = set(_ids_in_view(dlg))

        dlg._delete_btn.click()

        assert survivor == set(_ids_in_view(dlg)), "第二次点击不得动到升上来的模板"

    def test_clicking_promoted_row_restores_delete(self, three_templates, monkeypatch):
        """正向对照：走正常点选就能接着删，证明修的不是"永久卡死"。"""
        dlg, manager, ids = three_templates
        _stub_msgbox(monkeypatch)
        dlg._table.selectRow(0)
        dlg._on_delete_clicked()

        dlg._table.selectRow(0)

        assert dlg._selected_template_id == ids["b-second"]
        assert dlg._delete_btn.isEnabled()
        dlg._delete_btn.click()
        assert manager.get_template(ids["b-second"]) is None


class TestFilterHidesSelection:
    """比原报更严重的一条：过滤能选中一个看不见的模板。"""

    def test_filtering_out_selection_clears_ghost_target(
        self, three_templates, monkeypatch
    ):
        dlg, manager, ids = three_templates
        _stub_msgbox(monkeypatch)
        dlg._table.selectRow(0)
        assert dlg._selected_template_id == ids["c-third"]

        dlg._search_edit.setText("a-top")

        assert ids["c-third"] not in _ids_in_view(dlg), "前提：选中项已被滤掉"
        assert dlg._selected_template_id is None
        assert not any(b.isEnabled() for b in _gated_buttons(dlg)), (
            "否则点删除删的是用户屏幕上根本看不见的模板"
        )

    def test_ghost_target_survives_a_delete_click(self, three_templates, monkeypatch):
        dlg, manager, ids = three_templates
        _stub_msgbox(monkeypatch)
        dlg._table.selectRow(0)
        dlg._search_edit.setText("a-top")

        dlg._delete_btn.click()

        assert manager.get_template(ids["c-third"]) is not None, "被滤掉的模板不得被删"

    def test_filter_that_keeps_selection_re_homes_the_highlight(
        self, three_templates, monkeypatch
    ):
        """id 仍在视图里时高亮要挪到它真正的行，而不是停在原行号。"""
        dlg, _, ids = three_templates
        _stub_msgbox(monkeypatch)
        dlg._table.selectRow(2)  # a-top 在行 2
        assert _selected_rows(dlg) == [2]

        dlg._search_edit.setText("a-")

        assert dlg._selected_template_id == ids["a-top"]
        assert _selected_rows(dlg) == [0], "滤后 a-top 是唯一行，高亮要跟过去"


class TestDuplicateShiftsRows:
    """副本插到行 0 会让高亮和作用对象错位。"""

    def test_highlight_follows_id_after_duplicate(
        self, three_templates, monkeypatch
    ):
        dlg, _, ids = three_templates
        _stub_msgbox(monkeypatch)
        dlg._table.selectRow(0)

        dlg._on_duplicate_clicked()

        names = [
            dlg._table.item(r, 0).text() for r in range(dlg._table.rowCount())
        ]
        assert dlg._selected_template_id == ids["c-third"]
        assert _selected_rows(dlg) == [names.index("c-third")], (
            "高亮必须落在 id 所在行，否则删除/编辑作用的是没高亮的那一行"
        )
        assert dlg._details_label.text().startswith("选中: c-third")


class TestInPlaceEdit:
    """行数与行号都没变的一次重建：selectRow 不发信号，详情面板要自己跟上。"""

    def test_rename_keeps_details_label_current(self, three_templates):
        dlg, manager, ids = three_templates
        dlg._table.selectRow(0)
        assert dlg._details_label.text().startswith("选中: c-third")

        manager.save_template(
            PlotSessionConfig(plots=[PlotConfig(curves=["a"])]),
            "c-third-RENAMED",
            template_id=ids["c-third"],
        )

        assert dlg._selected_template_id == ids["c-third"]
        assert _selected_rows(dlg) == [0]
        assert dlg._details_label.text().startswith("选中: c-third-RENAMED"), (
            "列表已经换了名、详情还挂着旧名，等于界面上同时存在两个真相"
        )
