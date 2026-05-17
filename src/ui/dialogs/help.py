"""帮助对话框"""

from __future__ import annotations
from pathlib import Path
import os, sys
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QPushButton, QLabel, QTextEdit, QDialog
from src.core.config import debug_log


def _resource_path(relative_path: str) -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(os.path.join(sys._MEIPASS, relative_path))
    elif getattr(sys, "frozen", False):
        return Path(os.path.dirname(sys.executable)) / relative_path
    return Path(relative_path)


class HelpDialog(QDialog):
    """
    帮助对话框类
    用于显示应用程序的帮助文档，包括README.md文件内容
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("帮助文档")
        self.resize(800, 600)
        layout = QVBoxLayout(self)

        # 把窗口移动到屏幕中心
        screen = QApplication.primaryScreen().availableGeometry()
        size = self.geometry()
        x = (screen.width() - size.width()) // 2
        y = (screen.height() - size.height()) // 4
        self.move(x, y)

        # 文本区域
        text_edit = QTextEdit(self)
        text_edit.setReadOnly(True)
        
        # 加载 README.md
        readme_path = _resource_path("README.md")
        if readme_path.exists():
            with open(readme_path, "r", encoding="utf-8") as f:
                text_edit.setMarkdown(f.read())
        else:
            text_edit.setPlainText("README.md 文件未找到。")

        layout.addWidget(text_edit)

        # 关闭按钮
        close_btn = QPushButton("关闭", self)
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)



    

