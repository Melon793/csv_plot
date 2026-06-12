"""SheetSelectorDialog - Excel Sheet 选择对话框"""

from __future__ import annotations
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QDialogButtonBox, QMessageBox,
)


class SheetSelectorDialog(QDialog):
    """Excel Sheet 选择对话框

    使用 openpyxl 获取 Sheet 元数据（名称、行数、列数），
    不实际解析数据内容。
    """

    def __init__(self, file_path: str, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.selected_sheet: str | None = None

        # 异常处理：文件损坏或密码保护时给出友好提示
        try:
            from src.data.excel_loader import ExcelDataLoader
            self.sheet_info = ExcelDataLoader.get_sheet_info(file_path)
        except Exception as e:
            self.sheet_info = []
            self._load_error = str(e)
        else:
            self._load_error = None

        self._build_ui()

        if self._load_error:
            QMessageBox.warning(
                self, "无法读取 Excel 文件",
                f"文件可能已损坏或受密码保护。\n\n错误详情: {self._load_error}"
            )

        # 单 Sheet 快捷路径：自动选择，跳过对话框
        if not self._load_error and len(self.sheet_info) == 1:
            self.selected_sheet = self.sheet_info[0]['name']
            self.accept()

    def _build_ui(self):
        self.setWindowTitle("选择要导入的 Sheet")
        self.setMinimumSize(450, 300)
        self.resize(500, 350)

        layout = QVBoxLayout(self)

        # 标题标签
        title_label = QLabel("请选择要导入的工作表：")
        title_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(title_label)

        # Sheet 列表
        self.list_widget = QListWidget()
        self.list_widget.setAlternatingRowColors(True)

        for info in self.sheet_info:
            text = f"{info['name']}    —    {info['rows']} 行 × {info['cols']} 列"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, info['name'])
            self.list_widget.addItem(item)

        if self.sheet_info:
            self.list_widget.setCurrentRow(0)

        layout.addWidget(self.list_widget)

        # 按钮
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        # 异常时禁用确定按钮
        if self._load_error or not self.sheet_info:
            button_box.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)

    def _on_accept(self):
        current_item = self.list_widget.currentItem()
        if current_item:
            self.selected_sheet = current_item.data(Qt.ItemDataRole.UserRole)
            self.accept()

    def get_selected_sheet(self) -> str | None:
        return self.selected_sheet
