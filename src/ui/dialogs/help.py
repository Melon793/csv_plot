"""帮助对话框"""

from __future__ import annotations
from PySide6.QtWidgets import QApplication, QVBoxLayout, QPushButton, QTextEdit, QDialog
from src.utils.paths import resource_path
from src._version import get_version, get_build_time


class HelpDialog(QDialog):
    """
    帮助对话框类
    用于显示应用程序的帮助文档内容
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

        # 加载 docs/help.md，动态注入版本号与编译时间
        help_path = resource_path("docs/help.md")
        if help_path.exists():
            with open(help_path, "r", encoding="utf-8") as f:
                md_content = f.read()
            version = get_version()
            build_time = get_build_time()
            if build_time:
                header = f"# CSV Plot v{version}\n\n编译时间：{build_time}\n\n***\n\n"
            else:
                header = f"# CSV Plot v{version}\n\n***\n\n"
            text_edit.setMarkdown(header + md_content)
        else:
            text_edit.setPlainText("帮助文档未找到。")

        layout.addWidget(text_edit)

        # 关闭按钮
        close_btn = QPushButton("关闭", self)
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
