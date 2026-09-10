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

        # 加载 docs/help.md，在原文标题下方插入编译时间与版本号
        help_path = resource_path("docs/help.md")
        if help_path.exists():
            with open(help_path, "r", encoding="utf-8") as f:
                md_content = f.read()
            version = get_version()
            build_time = get_build_time()
            # 仅构建产物有 build_time，开发环境不插入
            if build_time:
                lines = md_content.splitlines()
                # 定位第一个 Markdown 标题行, 在其下方插入独立段落
                idx = next(
                    (i for i, ln in enumerate(lines) if ln.lstrip().startswith("#")),
                    None,
                )
                insert_at = (idx + 1) if idx is not None else 0
                # 前后各补一个空行，保证与相邻段落分隔（多余空行会被 Markdown 折叠）
                lines[insert_at:insert_at] = ["", f"编译时间：{build_time} (v{version})", ""]
                md_content = "\n".join(lines)
            text_edit.setMarkdown(md_content)
        else:
            text_edit.setPlainText("帮助文档未找到。")

        layout.addWidget(text_edit)

        # 关闭按钮
        close_btn = QPushButton("关闭", self)
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
