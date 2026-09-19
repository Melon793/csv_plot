"""P1-10：带 parent 的 exec 对话框用完不销毁 → 实例累积 + 信号回调倍增

`TemplateManagerDialog` 与长寿命的 `TemplateManager` 保持信号连接，只要 C++
侧还挂在主窗口上，之后每次模板变更就会触发 N 份 `_refresh_template_list`。
"""

from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QDialog, QWidget
from shiboken6 import isValid

from src.core.template_manager import TemplateManager
from src.ui.dialogs.help import HelpDialog
from src.ui.dialogs.template_manager_dialog import TemplateManagerDialog
from src.ui.layout_manager import LayoutManager
from src.ui.main_window import MainWindow


def flush_deferred_deletes(qapp):
    """deleteLater 要等 DeferredDelete 投递完才真销毁，processEvents 不够"""
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()


class _FakeConfigManager:
    def __init__(self, template_manager):
        self.template_manager = template_manager


class FakeMainWindow(QWidget):
    """只覆盖 open_template_manager / show_help 用到的那几个成员"""

    def __init__(self, template_manager):
        super().__init__()
        self.plot_config_manager = _FakeConfigManager(template_manager)
        self.layout_manager = LayoutManager(self)
        self.applied = []

    def apply_template(self, template_id):
        self.applied.append(template_id)


def _live_dialogs(parent, cls):
    return [d for d in parent.findChildren(cls) if isValid(d)]


def test_template_manager_reopen_does_not_accumulate(qapp, tmp_path, monkeypatch):
    mw = FakeMainWindow(TemplateManager(storage_path=tmp_path / "templates"))
    monkeypatch.setattr(
        TemplateManagerDialog, "exec", lambda self: QDialog.DialogCode.Accepted
    )

    for _ in range(3):
        MainWindow.open_template_manager(mw)
        flush_deferred_deletes(qapp)

    assert _live_dialogs(mw, TemplateManagerDialog) == []


def test_template_list_changed_refreshes_once_after_reopen(
    qapp, tmp_path, monkeypatch
):
    """关掉的管理器不得再当听众：一次变更只能触发一次刷新"""
    manager = TemplateManager(storage_path=tmp_path / "templates")
    mw = FakeMainWindow(manager)
    refreshes = []
    monkeypatch.setattr(
        TemplateManagerDialog, "_refresh_template_list", lambda self: refreshes.append(1)
    )
    monkeypatch.setattr(
        TemplateManagerDialog, "exec", lambda self: QDialog.DialogCode.Accepted
    )

    for _ in range(3):
        MainWindow.open_template_manager(mw)
        flush_deferred_deletes(qapp)
        refreshes.clear()

    manager.template_list_changed.emit()

    assert refreshes == []


def test_help_dialog_does_not_accumulate(qapp, monkeypatch):
    mw = FakeMainWindow(None)
    monkeypatch.setattr(HelpDialog, "exec", lambda self: QDialog.DialogCode.Accepted)

    for _ in range(3):
        LayoutManager(mw).show_help()
        flush_deferred_deletes(qapp)

    assert _live_dialogs(mw, HelpDialog) == []
