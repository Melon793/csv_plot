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

        self.sheet_info, self._load_error = self._probe(file_path)

        self._build_ui()

        if self._load_error:
            QMessageBox.warning(
                self, "无法读取 Excel 文件",
                f"文件可能已损坏或受密码保护。\n\n错误详情: {self._load_error}"
            )

    @staticmethod
    def _probe(file_path: str) -> tuple[list[dict], str | None]:
        """读 Sheet 元数据，返回 (sheet_info, 错误信息)。

        文件损坏或受密码保护时第一项为空列表、第二项为异常文本。
        """
        try:
            from src.data.excel_loader import ExcelDataLoader
            return ExcelDataLoader.get_sheet_info(file_path), None
        except Exception as e:
            return [], str(e)

    @classmethod
    def pick_sheet(cls, file_path: str, parent=None) -> str | None:
        """取要导入的 sheet 名；用户取消或读取失败返回 None。

        单 Sheet 直接返回、不弹框。构造函数里调 `accept()` 拦不住调用方随后的
        `exec()`（QDialog::exec 会重置 result 并重新 show），所以短路必须发生在
        弹框之前。
        """
        sheet_info, error = cls._probe(file_path)
        if not error and len(sheet_info) == 1:
            return sheet_info[0]['name']

        dialog = cls(file_path, parent)
        try:
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return None
            # get_selected_sheet() 读的是弹窗内控件，必须在 deleteLater 之前取
            return dialog.get_selected_sheet()
        finally:
            dialog.deleteLater()

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
