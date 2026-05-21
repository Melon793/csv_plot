"""Log window dialog - singleton with QSettings geometry persistence"""

from __future__ import annotations

from PySide6.QtCore import Qt, QSettings
from PySide6.QtWidgets import QDialog, QVBoxLayout

from src.core.logger import LogManager
from src.ui.widgets.log_viewer import LogViewer


class LogWindow(QDialog):
    _instance: LogWindow | None = None

    @classmethod
    def get_instance(cls, parent=None) -> LogWindow:
        if cls._instance is None:
            cls._instance = cls(parent)
        return cls._instance

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
        log_manager._ui_handler.new_log.connect(self._log_viewer.add_log_entry)

        self._restore_geometry()

    def _restore_geometry(self):
        settings = QSettings("CSVPlot", "LogWindow")
        geometry = settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        else:
            self.resize(800, 500)

    def _save_geometry(self):
        settings = QSettings("CSVPlot", "LogWindow")
        settings.setValue("geometry", self.saveGeometry())

    def showEvent(self, event):
        super().showEvent(event)
        self._restore_geometry()

    def closeEvent(self, event):
        self._save_geometry()
        super().closeEvent(event)

    def hideEvent(self, event):
        self._save_geometry()
        super().hideEvent(event)
