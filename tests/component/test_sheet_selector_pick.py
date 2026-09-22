"""P1-12：SheetSelector 单 Sheet 快捷路径失效

旧实现在构造函数里调 `accept()`，但 `QDialog::exec()` 起始会重置 result 并重新
show，快捷路径拦不住弹框。改为 `pick_sheet()` 在构造对话框**之前**短路。
"""

import pytest
from PySide6.QtWidgets import QDialog, QWidget
from shiboken6 import isValid

from tests.fixtures.data_factory import write_xlsx
from tests.fixtures.waits import flush_deferred_deletes
from src.ui.dialogs.sheet_selector import SheetSelectorDialog


@pytest.fixture()
def single_sheet_xlsx(tmp_path):
    return str(
        write_xlsx(
            tmp_path / "one.xlsx",
            header=["time", "speed"],
            rows=[[0.0, 1.0]],
            sheet_name="Data",
        )
    )


@pytest.fixture()
def multi_sheet_xlsx(tmp_path):
    return str(
        write_xlsx(
            tmp_path / "two.xlsx",
            header=["time", "speed"],
            rows=[[0.0, 1.0]],
            sheet_name="First",
            extra_sheet_names=["Second"],
        )
    )


def test_single_sheet_never_opens_dialog(qapp, single_sheet_xlsx, monkeypatch):
    """回归主诉求：单 Sheet 不再要求用户多点一次「确定」"""

    def boom(self):
        raise AssertionError("单 Sheet 不应弹出 SheetSelectorDialog")

    monkeypatch.setattr(SheetSelectorDialog, "exec", boom)
    assert SheetSelectorDialog.pick_sheet(single_sheet_xlsx) == "Data"


def test_multi_sheet_returns_chosen_name(qapp, multi_sheet_xlsx, monkeypatch):
    def fake_exec(self):
        self.list_widget.setCurrentRow(1)
        self._on_accept()
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(SheetSelectorDialog, "exec", fake_exec)
    assert SheetSelectorDialog.pick_sheet(multi_sheet_xlsx) == "Second"


def test_cancel_returns_none(qapp, multi_sheet_xlsx, monkeypatch):
    monkeypatch.setattr(
        SheetSelectorDialog, "exec", lambda self: QDialog.DialogCode.Rejected
    )
    assert SheetSelectorDialog.pick_sheet(multi_sheet_xlsx) is None


def test_dialog_is_destroyed_after_pick(qapp, multi_sheet_xlsx, monkeypatch):
    """带 parent 的 exec 对话框不能留在主窗口上（P1-10 同源，这里由 pick_sheet 兜住）"""
    parent = QWidget()
    monkeypatch.setattr(
        SheetSelectorDialog, "exec", lambda self: QDialog.DialogCode.Rejected
    )

    SheetSelectorDialog.pick_sheet(multi_sheet_xlsx, parent)
    flush_deferred_deletes()

    assert [d for d in parent.findChildren(SheetSelectorDialog) if isValid(d)] == []
    parent.deleteLater()
    flush_deferred_deletes()


def test_broken_file_still_warns_and_returns_none(
    qapp, tmp_path, monkeypatch, multi_sheet_xlsx
):
    """读失败时 sheet 列表为空，不能走快捷路径，也要保留原有警告弹窗"""
    broken = tmp_path / "broken.xlsx"
    broken.write_bytes(b"not an excel file")

    class NoPopupBox:
        warnings = []

        @staticmethod
        def warning(parent, title, text):
            NoPopupBox.warnings.append(title)

    monkeypatch.setattr(
        "src.ui.dialogs.sheet_selector.QMessageBox", NoPopupBox
    )
    monkeypatch.setattr(
        SheetSelectorDialog, "exec", lambda self: QDialog.DialogCode.Rejected
    )

    assert SheetSelectorDialog.pick_sheet(str(broken)) is None
    assert NoPopupBox.warnings == ["无法读取 Excel 文件"]
