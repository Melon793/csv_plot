"""Log window dialog - singleton with QSettings geometry persistence"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QVBoxLayout

from src.core.logger import LogManager
from src.ui.widgets.log_viewer import LogViewer


class LogWindow(QDialog):
    _instance: LogWindow | None = None

    @classmethod
    def _live_instance(cls) -> LogWindow | None:
        """返回仍然存活的单例；C++ 对象已销毁时顺手清空引用。

        单例的 parent 是主窗口，主窗口销毁会连带销毁本对话框的 C++ 对象，
        但类属性 ``_instance`` 仍持有 Python 包装器；不回收引用时下一次
        ``get_instance(...).show()`` 直接抛 RuntimeError。
        （同 ``VariableInfoDialog._live_instance``，那边已有测试固定。）
        """
        dlg = cls._instance
        if dlg is None:
            return None
        try:
            dlg.isVisible()
        except RuntimeError:
            cls._instance = None
            return None
        return dlg

    @classmethod
    def get_instance(cls, parent=None) -> LogWindow:
        dlg = cls._live_instance()
        if dlg is None:
            dlg = cls._instance = cls(parent)
        return dlg

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("日志窗口")
        self.setMinimumSize(600, 300)

        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.CustomizeWindowHint
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowCloseButtonHint
            | Qt.WindowType.WindowMinMaxButtonsHint
        )

        self._log_viewer = LogViewer(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._log_viewer)

        log_manager = LogManager.get_instance()
        log_manager.ui_handler.new_log.connect(self._log_viewer.add_log_entry)

        self._restore_geometry()

    def _restore_geometry(self):
        from src.core.settings import AppSettings
        settings = AppSettings()
        geometry = settings.get_log_window_geometry()
        if geometry is not None:
            self.restoreGeometry(geometry)
        else:
            self.resize(800, 300)

    def _save_geometry(self):
        from src.core.settings import AppSettings
        settings = AppSettings()
        settings.set_log_window_geometry(self.saveGeometry())

    def showEvent(self, event):
        super().showEvent(event)
        self._restore_geometry()

    def closeEvent(self, event):
        self._save_geometry()
        super().closeEvent(event)

    def hideEvent(self, event):
        self._save_geometry()
        super().hideEvent(event)
