"""LogViewer - QPlainTextEdit with QSyntaxHighlighter, ring buffer and batch refresh"""

from __future__ import annotations
import logging
import os
from dataclasses import dataclass

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QSyntaxHighlighter,
    QTextCharFormat,
    QTextDocument,
    QKeySequence,
    QShortcut,
)
from src.core.font_cache import get_monospace_font_cached
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPlainTextEdit,
    QPushButton,
    QComboBox,
    QLineEdit,
    QCheckBox,
    QLabel,
    QFileDialog,
)


@dataclass
class LogEntry:
    levelno: int
    levelname: str
    message: str
    name: str
    asctime: str


LEVEL_COLORS: dict[int, QColor] = {
    logging.DEBUG: QColor(128, 128, 128),
    logging.INFO: QColor(200, 200, 200),
    logging.WARNING: QColor(255, 200, 50),
    logging.ERROR: QColor(255, 80, 80),
    logging.CRITICAL: QColor(255, 40, 40),
}

LEVEL_NAMES: dict[int, str] = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO",
    logging.WARNING: "WARNING",
    logging.ERROR: "ERROR",
    logging.CRITICAL: "CRITICAL",
}


class LogSyntaxHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._formats: dict[int, QTextCharFormat] = {}
        for level, color in LEVEL_COLORS.items():
            fmt = QTextCharFormat()
            fmt.setForeground(color)
            self._formats[level] = fmt

    def highlightBlock(self, text: str):
        if not text:
            return
        for level in (logging.CRITICAL, logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG):
            level_name = LEVEL_NAMES.get(level, "")
            if level_name and level_name in text:
                self.setFormat(0, len(text), self._formats[level])
                return


class LogViewer(QWidget):
    MAX_BUFFER_LINES = 10000
    APPEND_BATCH_INTERVAL_MS = 50
    FLUSH_THRESHOLD = 20

    filter_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self._all_entries: list[LogEntry] = []
        self._buffer: list[LogEntry] = []
        self._scroll_locked = True
        self._level_filter: set[int] = set()

        self._build_ui()

        self._batch_timer = QTimer(self)
        self._batch_timer.setInterval(self.APPEND_BATCH_INTERVAL_MS)
        self._batch_timer.timeout.connect(self._flush_buffer)

        shortcut = QShortcut(QKeySequence.StandardKey.Find, self)
        shortcut.activated.connect(self._focus_search)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(4, 4, 4, 2)
        toolbar.setSpacing(4)

        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("搜索日志...")
        self._search_input.setClearButtonEnabled(True)
        self._search_input.textChanged.connect(self._on_search_text_changed)
        toolbar.addWidget(self._search_input)

        self._search_prev_btn = QPushButton("◀")
        self._search_prev_btn.setFixedWidth(28)
        self._search_prev_btn.setToolTip("上一个匹配")
        self._search_prev_btn.clicked.connect(self._search_prev)
        toolbar.addWidget(self._search_prev_btn)

        self._search_next_btn = QPushButton("▶")
        self._search_next_btn.setFixedWidth(28)
        self._search_next_btn.setToolTip("下一个匹配")
        self._search_next_btn.clicked.connect(self._search_next)
        toolbar.addWidget(self._search_next_btn)

        self._level_combo = QComboBox()
        self._level_combo.addItem("全部", "all")
        self._level_combo.addItem("DEBUG+", "debug")
        self._level_combo.addItem("INFO+", "info")
        self._level_combo.addItem("WARNING+", "warning")
        self._level_combo.addItem("ERROR+", "error")
        self._level_combo.addItem("CRITICAL", "critical")
        self._level_combo.setCurrentText("INFO+")
        self._level_combo.currentIndexChanged.connect(self._on_level_filter_changed)
        toolbar.addWidget(self._level_combo)

        self._clear_btn = QPushButton("清空")
        self._clear_btn.clicked.connect(self.clear_logs)
        toolbar.addWidget(self._clear_btn)

        self._save_btn = QPushButton("保存")
        self._save_btn.clicked.connect(self._save_logs)
        toolbar.addWidget(self._save_btn)

        self._pin_check = QCheckBox("置顶")
        self._pin_check.toggled.connect(self._on_pin_toggled)
        toolbar.addWidget(self._pin_check)

        layout.addLayout(toolbar)

        self._text_edit = QPlainTextEdit()
        self._text_edit.setReadOnly(True)
        mono_font = get_monospace_font_cached()
        if mono_font:
            font = QFont(mono_font, 11)
        else:
            font = QFont("monospace", 11)
            font.setStyleHint(QFont.StyleHint.Monospace)
        self._text_edit.setFont(font)
        self._text_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._text_edit.setMaximumBlockCount(0)
        self._text_edit.verticalScrollBar().valueChanged.connect(self._on_scroll_changed)
        self._highlighter = LogSyntaxHighlighter(self._text_edit.document())
        layout.addWidget(self._text_edit, 1)

        status_bar = QHBoxLayout()
        status_bar.setContentsMargins(4, 2, 4, 4)
        status_bar.setSpacing(8)
        self._count_label = QLabel("共 0 条")
        self._filtered_label = QLabel("")
        status_bar.addWidget(self._count_label)
        status_bar.addWidget(self._filtered_label)
        status_bar.addStretch()
        layout.addLayout(status_bar)

    def append_log(self, entry: LogEntry):
        self._buffer.append(entry)
        self._all_entries.append(entry)

        if len(self._all_entries) > self.MAX_BUFFER_LINES:
            self._all_entries = self._all_entries[-self.MAX_BUFFER_LINES:]

        if len(self._buffer) >= self.FLUSH_THRESHOLD:
            self._flush_buffer()
        elif not self._batch_timer.isActive():
            self._batch_timer.start()

    def _flush_buffer(self):
        self._batch_timer.stop()
        if not self._buffer:
            return

        self._text_edit.verticalScrollBar().blockSignals(True)

        show_all = len(self._level_filter) == 0
        lines = []
        for entry in self._buffer:
            if show_all or entry.levelno in self._level_filter:
                lines.append(entry.message)

        if lines:
            text = "\n".join(lines) + "\n"
            scrollbar = self._text_edit.verticalScrollBar()
            was_at_bottom = scrollbar.value() >= scrollbar.maximum() - 4

            self._text_edit.insertPlainText(text)

            if was_at_bottom and self._scroll_locked:
                self._text_edit.moveCursor(self._text_edit.textCursor().MoveOperation.End)

        self._text_edit.verticalScrollBar().blockSignals(False)
        self._buffer.clear()
        self._update_count_label()

    def _rebuild_display(self):
        self._text_edit.clear()
        show_all = len(self._level_filter) == 0
        filtered = []
        for entry in self._all_entries:
            if show_all or entry.levelno in self._level_filter:
                filtered.append(entry)

        if filtered:
            text = "\n".join(e.message for e in filtered) + "\n"
            self._text_edit.setPlainText(text)

        if self._scroll_locked:
            self._text_edit.moveCursor(self._text_edit.textCursor().MoveOperation.End)

        self._update_count_label(visible_count=len(filtered))

    def clear_logs(self):
        self._batch_timer.stop()
        self._buffer.clear()
        self._all_entries.clear()
        self._text_edit.clear()
        self._update_count_label(visible_count=0)

    def _save_logs(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存日志",
            os.path.expanduser("~"),
            "日志文件 (*.log);;文本文件 (*.txt);;所有文件 (*)",
        )
        if not file_path:
            return

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                for entry in self._all_entries:
                    f.write(f"{entry.message}\n")
        except OSError as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "保存失败", f"无法保存日志文件: {e}")

    def _on_level_filter_changed(self):
        level_key = self._level_combo.currentData()
        level_map = {
            "all": set(),
            "debug": {logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL},
            "info": {logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL},
            "warning": {logging.WARNING, logging.ERROR, logging.CRITICAL},
            "error": {logging.ERROR, logging.CRITICAL},
            "critical": {logging.CRITICAL},
        }
        self._level_filter = level_map.get(level_key, set())
        self._rebuild_display()

    def _on_search_text_changed(self, text: str):
        if not text:
            self._text_edit.moveCursor(self._text_edit.textCursor().MoveOperation.Start)
            return
        self._text_edit.moveCursor(self._text_edit.textCursor().MoveOperation.Start)
        self._search_next()

    def _search_next(self):
        text = self._search_input.text()
        if not text:
            return
        found = self._text_edit.find(text)
        if not found:
            self._text_edit.moveCursor(self._text_edit.textCursor().MoveOperation.Start)
            self._text_edit.find(text)

    def _search_prev(self):
        text = self._search_input.text()
        if not text:
            return
        found = self._text_edit.find(text, QTextDocument.FindFlag.FindBackward)
        if not found:
            self._text_edit.moveCursor(self._text_edit.textCursor().MoveOperation.End)
            self._text_edit.find(text, QTextDocument.FindFlag.FindBackward)

    def _focus_search(self):
        self._search_input.setFocus()
        self._search_input.selectAll()

    def _on_scroll_changed(self, value: int):
        scrollbar = self._text_edit.verticalScrollBar()
        self._scroll_locked = value >= scrollbar.maximum() - 4

    def _on_pin_toggled(self, checked: bool):
        window = self.window()
        if window:
            if checked:
                window.setWindowFlags(window.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
            else:
                window.setWindowFlags(window.windowFlags() & ~Qt.WindowType.WindowStaysOnTopHint)
            window.show()

    def _update_count_label(self, visible_count: int | None = None):
        total = len(self._all_entries)
        self._count_label.setText(f"共 {total} 条")
        if visible_count is not None and len(self._level_filter) > 0:
            self._filtered_label.setText(f"(显示 {visible_count} 条)")
        else:
            self._filtered_label.setText("")

    def add_log_entry(self, log_dict: dict):
        entry = LogEntry(
            levelno=log_dict.get("levelno", logging.INFO),
            levelname=log_dict.get("levelname", "INFO"),
            message=log_dict.get("message", ""),
            name=log_dict.get("name", ""),
            asctime=log_dict.get("asctime", ""),
        )
        self.append_log(entry)
