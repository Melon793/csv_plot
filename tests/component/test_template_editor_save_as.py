"""P1-11：点「另存为」后若保存没成，编辑目标已被抹掉且无回滚

旧 `_on_saveas_clicked` 先 `self._edit_template_id = None` 再转调保存：名称
为空 / YAML 报错 / 冲突框选「取消」都会 return，但编辑目标已经没了 —— 用户
接着点「保存」时静默变成新建一个同配置模板。
"""

import pytest
from PySide6.QtWidgets import QMessageBox

from src.core.plot_config import PlotConfig, PlotSessionConfig
from src.core.template_manager import TemplateManager
from src.ui.dialogs.template_editor_dialog import TemplateEditorDialog


def _config(curves):
    return PlotSessionConfig(plots=[PlotConfig(curves=list(curves))])


@pytest.fixture()
def manager(qapp, tmp_path):
    return TemplateManager(storage_path=tmp_path / "templates")


@pytest.fixture()
def boxes(monkeypatch):
    """吞掉三态弹窗并记下标题"""
    shown = []
    for kind in ("warning", "information", "critical"):
        monkeypatch.setattr(
            QMessageBox,
            kind,
            staticmethod(lambda parent, title, *a, **k: shown.append(title)),
        )
    return shown


def _editor(manager, template_id):
    """顶层构造即可：临时 QWidget() 当 parent 会让子对话框立刻一起被回收"""
    return TemplateEditorDialog(manager, edit_template_id=template_id)


def test_cancelled_save_as_keeps_edit_target(manager, boxes, monkeypatch):
    first = manager.save_template(_config(["a"]), "tpl1", "d1")
    manager.save_template(_config(["b"]), "tpl2", "d2")
    dlg = _editor(manager, first.metadata.id)
    monkeypatch.setattr(
        TemplateEditorDialog,
        "_ask_conflict_resolution",
        lambda self, name, edit_id=None: "cancel",
    )

    dlg._name_edit.setText("tpl2")
    dlg._saveas_btn.click()  # 改名成 tpl2 → 冲突 → 取消，本次什么都没写

    assert dlg._edit_template_id == first.metadata.id, "编辑目标被另存为抹掉了"
    assert [t.metadata.name for t in manager.get_all_templates()] == ["tpl2", "tpl1"]


def test_save_after_cancelled_save_as_still_targets_original(
    manager, boxes, monkeypatch
):
    """上一条的回滚必须真接在「保存」上：改名保存的是原模板，不是又新建一个"""
    first = manager.save_template(_config(["a"]), "tpl1", "d1")
    manager.save_template(_config(["b"]), "tpl2", "d2")
    dlg = _editor(manager, first.metadata.id)
    monkeypatch.setattr(
        TemplateEditorDialog,
        "_ask_conflict_resolution",
        lambda self, name, edit_id=None: "cancel",
    )
    dlg._name_edit.setText("tpl2")
    dlg._saveas_btn.click()

    dlg._name_edit.setText("tpl3")
    dlg._save_btn.click()

    names = sorted(t.metadata.name for t in manager.get_all_templates())
    assert names == ["tpl2", "tpl3"], "「保存」新建了模板，说明编辑目标已被抹掉"
    assert manager.get_template(first.metadata.id).metadata.name == "tpl3"


def test_save_as_success_creates_new_template_and_leaves_original(manager, boxes):
    first = manager.save_template(_config(["a"]), "tpl1", "d1")
    dlg = _editor(manager, first.metadata.id)

    dlg._name_edit.setText("copy")
    dlg._saveas_btn.click()

    assert manager.exists("copy")
    assert manager.get_template(first.metadata.id).metadata.name == "tpl1"
    # 写盘成功后才切换编辑目标：后续保存落在新模板上
    copy_id = next(
        t.metadata.id for t in manager.get_all_templates() if t.metadata.name == "copy"
    )
    assert dlg._edit_template_id == copy_id


def test_save_success_is_quiet_and_hands_the_wording_over(manager, boxes):
    """成功框已撤：点保存不该再有「成功」弹窗，动词交给状态栏那一句去拼。"""
    emitted = []
    dlg = _editor(manager, None)
    dlg.template_saved.connect(emitted.append)

    dlg._name_edit.setText("quiet")
    dlg._save_btn.click()

    assert "成功" not in boxes, f"成功框还在: {boxes}"
    assert emitted, "撤弹窗不能把 template_saved 一起撤掉"
    assert dlg.saved_summary == "模板已保存"
    assert manager.exists("quiet")


def test_edit_save_reports_update_not_save(manager, boxes):
    """编辑已有模板时动词必须是「模板已更新」，否则状态栏会误报新建了一个。"""
    first = manager.save_template(_config(["a"]), "tpl1", "d1")
    dlg = _editor(manager, first.metadata.id)

    dlg._save_btn.click()

    assert "成功" not in boxes
    assert dlg.saved_summary == "模板已更新"
