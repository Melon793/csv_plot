"""标注列表对话框 —— 当前子图标注的定位 / 显隐 / 编辑 / 删除

列表里只放"每条标注一行 + 一行操作按钮"，不做树形分组：P1 的标注不绑定曲线，
平铺一层最直观，也避开了"曲线改名后分组归属算谁的"这类语义问题。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)


class AnnotationListDialog(QDialog):
    """标注列表面板"""

    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self._manager = manager
        self.setWindowTitle("标注列表")
        self.resize(460, 380)

        self._list = QListWidget()
        self._list.itemChanged.connect(self._on_item_changed)
        self._list.itemDoubleClicked.connect(lambda _item: self._edit_selected())

        self._count_label = QLabel()
        self._count_label.setStyleSheet("color: gray;")

        select_btn = QPushButton("定位并选中")
        select_btn.clicked.connect(self._focus_selected)
        edit_btn = QPushButton("编辑属性")
        edit_btn.clicked.connect(self._edit_selected)
        delete_btn = QPushButton("删除")
        delete_btn.clicked.connect(self._delete_selected)
        clear_btn = QPushButton("清除全部")
        clear_btn.clicked.connect(self._clear_all)
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.accept)

        top_row = QHBoxLayout()
        top_row.addWidget(select_btn)
        top_row.addWidget(edit_btn)
        top_row.addWidget(delete_btn)
        top_row.addStretch(1)

        bottom_row = QHBoxLayout()
        bottom_row.addWidget(clear_btn)
        bottom_row.addStretch(1)
        bottom_row.addWidget(close_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(self._count_label)
        layout.addWidget(self._list, 1)
        layout.addLayout(top_row)
        layout.addLayout(bottom_row)

        self._reload()

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _reload(self) -> None:
        """重建列表

        重建期间必须挡住 itemChanged：逐条 addItem 会把每个条目都算作"用户改了
        复选状态"，进而把显隐反复回写一遍。
        """
        self._list.blockSignals(True)
        self._list.clear()
        models = self._manager.items()
        for model in models:
            item = QListWidgetItem(model.summary())
            item.setData(Qt.ItemDataRole.UserRole, model.id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if model.visible else Qt.CheckState.Unchecked
            )
            if model.id == self._manager.selected_id:
                item.setText("● " + item.text())
            self._list.addItem(item)
        self._list.blockSignals(False)

        self._count_label.setText(f"共 {len(models)} 条标注（去勾选可临时隐藏）")

    def _selected_id(self) -> str | None:
        row = self._list.currentRow()
        if row < 0:
            return None
        item = self._list.item(row)
        return item.data(Qt.ItemDataRole.UserRole) if item is not None else None

    # ------------------------------------------------------------------
    # 槽
    # ------------------------------------------------------------------
    def _on_item_changed(self, item: QListWidgetItem) -> None:
        annotation_id = item.data(Qt.ItemDataRole.UserRole)
        if not annotation_id:
            return
        self._manager.set_visible(
            annotation_id, item.checkState() == Qt.CheckState.Checked
        )

    def _focus_selected(self) -> None:
        annotation_id = self._selected_id()
        if annotation_id is None:
            QMessageBox.information(self, "提示", "请先选中一条标注")
            return
        self._manager.select(annotation_id)
        self._reload()

    def _edit_selected(self) -> None:
        annotation_id = self._selected_id()
        if annotation_id is None:
            QMessageBox.information(self, "提示", "请先选中一条标注")
            return
        if self._manager.open_property_dialog(annotation_id):
            self._reload()

    def _delete_selected(self) -> None:
        annotation_id = self._selected_id()
        if annotation_id is None:
            QMessageBox.information(self, "提示", "请先选中一条标注")
            return
        self._manager.remove(annotation_id)
        self._reload()

    def _clear_all(self) -> None:
        if self._manager.count() == 0:
            return
        confirm = QMessageBox.question(
            self,
            "确认清除",
            f"确定清除当前子图的 {self._manager.count()} 条标注吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self._manager.clear()
            self._reload()
