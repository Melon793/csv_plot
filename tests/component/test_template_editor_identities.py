"""模板编辑器白名单再生成携带 curve_identities（C3 / D11 / BL-3 回归）。

漏写等于「编辑 → 保存」清空全部身份（与 invalid_curves/curve_settings
同类事故）。孤儿身份条目（用户手改曲线名留下）保留不删。
"""

from __future__ import annotations

import yaml
import pytest
from PySide6.QtWidgets import QMessageBox

from src.core.plot_config import PlotConfig, PlotPageConfig, PlotSessionConfig
from src.core.template_manager import TemplateManager
from src.ui.dialogs.template_editor_dialog import TemplateEditorDialog
from tests.fixtures.waits import flush_deferred_deletes


MDF_ENTRY = {
    "original_name": "Press", "kind": "mdf",
    "device": "Dev1", "ecu": "ECU_A", "unit": "bar",
}
BLF_ENTRY = {
    "original_name": "EngineSpeed", "kind": "blf", "frame": "EngineData",
    "can_id": 0x1A0, "extended": False, "bus_channel": 0, "unit": "rpm",
}


@pytest.fixture()
def manager(qapp, tmp_path):
    return TemplateManager(storage_path=tmp_path / "templates")


@pytest.fixture(autouse=True)
def boxes(monkeypatch):
    def fake(parent, title, text, *args, **kwargs):
        return QMessageBox.StandardButton.Ok

    for kind in ("warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, kind, staticmethod(fake))


def _config_with_identities() -> PlotSessionConfig:
    return PlotSessionConfig(
        pages=[PlotPageConfig(plots=[PlotConfig(
            curves=["Press_G0", "EngineData.EngineSpeed_G0"],
            curve_identities={
                "Press_G0": dict(MDF_ENTRY),
                "EngineData.EngineSpeed_G0": dict(BLF_ENTRY),
                # 孤儿条目：用户手改曲线名留下，引擎按精确键查，保留不删
                "orphan": {"original_name": "Ghost", "kind": "mdf"},
            },
        )])]
    )


def _editor(manager, qtbot, config=None, edit_template_id=None):
    dlg = TemplateEditorDialog(
        manager, current_config=config, edit_template_id=edit_template_id
    )
    qtbot.addWidget(dlg)
    return dlg


def test_generated_yaml_carries_identities(manager, qtbot):
    editor = _editor(manager, qtbot, _config_with_identities())
    try:
        payload = yaml.safe_load(editor._yaml_edit.toPlainText())
        identities = payload["pages"][0]["plots"][0]["curve_identities"]
        assert identities["Press_G0"] == MDF_ENTRY
        assert identities["EngineData.EngineSpeed_G0"] == BLF_ENTRY
        assert identities["orphan"] == {"original_name": "Ghost", "kind": "mdf"}
    finally:
        editor.deleteLater()
        flush_deferred_deletes()


def test_edit_save_roundtrip_keeps_identities(manager, qtbot):
    """编辑 → 保存 → 身份仍在（BL-3 核心回归）。"""
    saved_ids: list[str] = []
    editor = _editor(manager, qtbot, _config_with_identities())
    try:
        editor.template_saved.connect(saved_ids.append)
        editor._name_edit.setText("身份模板")
        editor._save_btn.click()
        assert saved_ids, "保存必须成功"
        template = manager.get_template(saved_ids[-1])
        plot = template.config["pages"][0]["plots"][0]
        assert plot["curve_identities"]["Press_G0"] == MDF_ENTRY
        assert plot["curve_identities"]["EngineData.EngineSpeed_G0"] == BLF_ENTRY
    finally:
        editor.deleteLater()
        flush_deferred_deletes()

    # 再编辑已保存模板 → 再生成 → 再保存：身份不丢
    editor2 = _editor(manager, qtbot, edit_template_id=saved_ids[-1])
    try:
        payload = yaml.safe_load(editor2._yaml_edit.toPlainText())
        assert payload["pages"][0]["plots"][0]["curve_identities"]["Press_G0"] == MDF_ENTRY
        editor2.template_saved.connect(saved_ids.append)
        editor2._save_btn.click()
        template = manager.get_template(saved_ids[-1])
        plot = template.config["pages"][0]["plots"][0]
        assert plot["curve_identities"]["EngineData.EngineSpeed_G0"] == BLF_ENTRY
        assert "orphan" in plot["curve_identities"]
    finally:
        editor2.deleteLater()
        flush_deferred_deletes()


def test_no_identities_no_key(manager, qtbot):
    cfg = PlotSessionConfig(pages=[PlotPageConfig(plots=[PlotConfig(curves=["speed"])])])
    editor = _editor(manager, qtbot, cfg)
    try:
        payload = yaml.safe_load(editor._yaml_edit.toPlainText())
        assert "curve_identities" not in payload["pages"][0]["plots"][0]
    finally:
        editor.deleteLater()
        flush_deferred_deletes()