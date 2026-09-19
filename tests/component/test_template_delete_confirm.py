"""P2-18：不可逆操作「删除模板」的确认框，默认按钮必须是「否」。

`QMessageBox.question` 不传 `defaultButton` 时，默认落在 buttons 里的第一个按钮上，
也就是「Yes」—— 回车/空格直接删掉模板。项目里其余确认框（关窗、清空列、图例移动）
都已显式 `StandardButton.No`，此处是漏网的一处。
"""

from __future__ import annotations

import pytest

from PySide6.QtWidgets import QMessageBox

from src.core.plot_config import PlotConfig, PlotSessionConfig
from src.core.template_manager import TemplateManager
from src.ui.dialogs import template_manager_dialog as tmd
from src.ui.dialogs.template_manager_dialog import TemplateManagerDialog

YES = QMessageBox.StandardButton.Yes
NO = QMessageBox.StandardButton.No


def _stub(kind, calls, answer=None):
    def stub(*args, **kwargs):
        calls.append(args if answer is not None else (kind,) + args)
        return answer if answer is not None else QMessageBox.StandardButton.Ok

    return stub


def _patch_msgbox(monkeypatch, answer):
    """替身只记录调用参数：默认按钮只能从实参里看出来，弹窗本身不该在测试里弹。"""
    calls = []

    class _Fake:
        StandardButton = QMessageBox.StandardButton

    # 类体看不到外层函数的局部名，逐个挂属性赋值
    _Fake.question = staticmethod(_stub("question", calls, answer))
    _Fake.warning = staticmethod(_stub("warning", calls))
    _Fake.information = staticmethod(_stub("information", calls))
    _Fake.critical = staticmethod(_stub("critical", calls))

    monkeypatch.setattr(tmd, "QMessageBox", _Fake)
    return calls


@pytest.fixture()
def dialog_with_template(qapp, tmp_path):
    manager = TemplateManager(storage_path=tmp_path / "templates")
    tpl = manager.save_template(
        PlotSessionConfig(plots=[PlotConfig(curves=["a"])]), "要删的模板"
    )
    dlg = TemplateManagerDialog(manager)
    dlg._selected_template_id = tpl.metadata.id
    return dlg, manager, tpl.metadata.id


class TestDeleteTemplateConfirmation:
    def test_default_button_is_no(self, dialog_with_template, monkeypatch):
        dlg, manager, tid = dialog_with_template
        calls = _patch_msgbox(monkeypatch, NO)

        dlg._on_delete_clicked()

        assert len(calls) == 1, f"只该弹一次确认: {calls}"
        args = calls[0]
        assert args[1] == "删除模板"
        assert args[3] == (YES | NO)
        assert len(args) == 5, "确认框必须显式给出 defaultButton"
        assert args[4] is NO, "不可逆删除的默认按钮必须是「否」"
        assert manager.get_template(tid) is not None, "答「否」不得删掉模板"

    def test_answering_yes_really_deletes(self, dialog_with_template, monkeypatch):
        """正向对照：本用例确实走到了删除分支，而不是在测一个没被触发的路径。"""
        dlg, manager, tid = dialog_with_template
        _patch_msgbox(monkeypatch, YES)

        dlg._on_delete_clicked()

        assert manager.get_template(tid) is None
        assert dlg._selected_template_id is None

    def test_no_selection_warns_without_confirming(self, dialog_with_template, monkeypatch):
        dlg, manager, _ = dialog_with_template
        dlg._selected_template_id = None
        calls = _patch_msgbox(monkeypatch, YES)

        dlg._on_delete_clicked()

        assert [c[0] for c in calls] == ["warning"], "未选中时只提示，不该弹确认"
