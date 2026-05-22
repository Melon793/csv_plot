"""模板管理器对话框"""

from __future__ import annotations
import yaml
from pathlib import Path
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QMessageBox,
    QFileDialog,
    QSplitter,
)
from src.core.template_manager import TemplateManager
from src.core.template_models import PlotTemplate
from src.core.plot_config import (
    PlotSessionConfig,
    TemplateNotFoundError,
    TemplateNameConflictError,
)
from src.core.logger import get_logger
from src.ui.dialogs.template_editor_dialog import TemplateEditorDialog


logger = get_logger(__name__)


class _ProportionalTable(QTableWidget):
    """按比例自适应列宽的表格"""

    def __init__(self, ratios: list[int], parent=None):
        super().__init__(parent)
        self._ratios = ratios

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_proportions()

    def _apply_proportions(self):
        total_w = self.viewport().width()
        if total_w <= 0:
            return
        total_ratio = sum(self._ratios)
        for i, r in enumerate(self._ratios):
            width = int(total_w * r / total_ratio)
            if width < 30:
                width = 30
            self.setColumnWidth(i, width)


class TemplateManagerDialog(QDialog):
    """模板管理器对话框 - 加载、管理、导入导出模板"""

    template_applied = Signal(str)  # template_id

    def __init__(
        self,
        template_manager: TemplateManager,
        parent=None,
    ):
        super().__init__(parent)
        self._template_manager = template_manager
        self._selected_template_id: str | None = None
        self.setWindowTitle("📋 模板管理器")
        self.resize(800, 600)
        self._setup_ui()
        self._connect_signals()
        self._refresh_template_list()

    def _setup_ui(self):
        """设置 UI"""
        layout = QVBoxLayout(self)

        # 搜索栏
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("🔍 搜索:"))
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("输入模板名称或描述...")
        search_layout.addWidget(self._search_edit)
        layout.addLayout(search_layout)

        # 分割器
        splitter = QSplitter(Qt.Orientation.Vertical)

        # 模板列表（3:1:1:3 比例自适应列宽）
        self._table = _ProportionalTable(ratios=[3, 1, 1, 3])
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(
            ["名称", "变量数", "Plot 数", "更新时间"]
        )
        self._table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self._table.setSelectionMode(
            QTableWidget.SelectionMode.SingleSelection
        )
        self._table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        splitter.addWidget(self._table)

        # 详情面板
        self._details_label = QLabel("选中: -")
        self._details_label.setWordWrap(True)
        splitter.addWidget(self._details_label)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

        # 按钮栏
        button_layout = QHBoxLayout()
        self._new_btn = QPushButton("➕ 新建")
        self._import_btn = QPushButton("📥 导入")
        self._export_btn = QPushButton("📤 导出")
        self._edit_btn = QPushButton("✏️ 编辑")
        self._duplicate_btn = QPushButton("📋 复制")
        self._delete_btn = QPushButton("🗑️ 删除")
        button_layout.addWidget(self._new_btn)
        button_layout.addWidget(self._import_btn)
        button_layout.addWidget(self._export_btn)
        button_layout.addWidget(self._edit_btn)
        button_layout.addWidget(self._duplicate_btn)
        button_layout.addWidget(self._delete_btn)
        button_layout.addStretch()
        self._load_btn = QPushButton("📂 加载选中")
        self._load_btn.setDefault(True)
        self._close_btn = QPushButton("关闭")
        button_layout.addWidget(self._load_btn)
        button_layout.addWidget(self._close_btn)
        layout.addLayout(button_layout)

    def _connect_signals(self):
        """连接信号"""
        self._search_edit.textChanged.connect(self._refresh_template_list)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        self._table.itemDoubleClicked.connect(self._on_double_clicked)

        self._import_btn.clicked.connect(self._on_import_clicked)
        self._export_btn.clicked.connect(self._on_export_clicked)
        self._edit_btn.clicked.connect(self._on_edit_clicked)
        self._new_btn.clicked.connect(self._on_new_clicked)
        self._duplicate_btn.clicked.connect(self._on_duplicate_clicked)
        self._delete_btn.clicked.connect(self._on_delete_clicked)
        self._load_btn.clicked.connect(self._on_load_clicked)
        self._close_btn.clicked.connect(self.accept)

        self._template_manager.template_list_changed.connect(
            self._refresh_template_list
        )

    def _refresh_template_list(self):
        """刷新模板列表"""
        keyword = self._search_edit.text()
        templates = self._template_manager.search(keyword=keyword)

        self._table.setRowCount(len(templates))
        for row, template in enumerate(templates):
            name_item = QTableWidgetItem(template.metadata.name)
            name_item.setData(Qt.ItemDataRole.UserRole, template.metadata.id)
            self._table.setItem(row, 0, name_item)

            var_count = self._count_variables(template.config)
            self._table.setItem(row, 1, QTableWidgetItem(str(var_count)))

            plot_count = len(template.config.get("plots", []))
            self._table.setItem(row, 2, QTableWidgetItem(str(plot_count)))

            updated_at = template.metadata.updated_at[:16]  # 只显示日期和时间
            self._table.setItem(row, 3, QTableWidgetItem(updated_at))

    def _on_selection_changed(self):
        """选择变化时更新详情"""
        selected = self._table.selectedItems()
        if selected:
            item = selected[0]
            template_id = item.data(Qt.ItemDataRole.UserRole)
            self._selected_template_id = template_id
            template = self._template_manager.get_template(template_id)
            if template:
                self._details_label.setText(
                    f"选中: {template.metadata.name}\n\n"
                    f"描述: {template.metadata.description or '无'}\n\n"
                    f"创建时间: {template.metadata.created_at[:16]}\n"
                    f"更新时间: {template.metadata.updated_at[:16]}"
                )
                return

        self._selected_template_id = None
        self._details_label.setText("选中: -")

    def _on_double_clicked(self, item):
        """双击加载模板"""
        self._on_load_clicked()

    def _on_new_clicked(self):
        """新建空白模板"""
        dialog = TemplateEditorDialog(
            self._template_manager,
            parent=self,
        )
        dialog.template_saved.connect(self._on_template_edited)
        dialog.exec()

    def _on_import_clicked(self):
        """导入模板"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "导入模板",
            "",
            "YAML Files (*.yaml *.yml);;All Files (*)",
        )
        if file_path:
            try:
                template = self._template_manager.import_template(Path(file_path))
                if template:
                    QMessageBox.information(
                        self, "成功", f"已导入模板: {template.metadata.name}"
                    )
            except Exception as e:
                QMessageBox.critical(
                    self, "错误", f"导入模板失败: {str(e)}"
                )

    def _on_export_clicked(self):
        """导出模板"""
        if not self._selected_template_id:
            QMessageBox.warning(self, "警告", "请先选择一个模板")
            return

        template = self._template_manager.get_template(self._selected_template_id)
        if not template:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出模板",
            f"{template.metadata.name}.yaml",
            "YAML Files (*.yaml *.yml);;All Files (*)",
        )
        if file_path:
            try:
                success = self._template_manager.export_template(
                    self._selected_template_id, Path(file_path)
                )
                if success:
                    QMessageBox.information(self, "成功", "模板已导出")
                else:
                    QMessageBox.critical(self, "错误", "导出模板失败")
            except Exception as e:
                QMessageBox.critical(
                    self, "错误", f"导出模板失败: {str(e)}"
                )

    def _on_edit_clicked(self):
        """编辑模板"""
        if not self._selected_template_id:
            QMessageBox.warning(self, "警告", "请先选择一个模板")
            return

        dialog = TemplateEditorDialog(
            self._template_manager,
            edit_template_id=self._selected_template_id,
            parent=self,
        )
        dialog.template_saved.connect(self._on_template_edited)
        dialog.exec()

    def _on_duplicate_clicked(self):
        """复制模板"""
        if not self._selected_template_id:
            QMessageBox.warning(self, "警告", "请先选择一个模板")
            return

        template = self._template_manager.get_template(self._selected_template_id)
        if not template:
            return

        new_name = f"{template.metadata.name} (副本)"
        reply = QMessageBox.question(
            self,
            "复制模板",
            f"将模板复制为: {new_name}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self._template_manager.duplicate_template(
                    self._selected_template_id, new_name
                )
                QMessageBox.information(self, "成功", "模板已复制")
            except Exception as e:
                QMessageBox.critical(
                    self, "错误", f"复制模板失败: {str(e)}"
                )

    def _on_delete_clicked(self):
        """删除模板"""
        if not self._selected_template_id:
            QMessageBox.warning(self, "警告", "请先选择一个模板")
            return

        template = self._template_manager.get_template(self._selected_template_id)
        if not template:
            return

        reply = QMessageBox.question(
            self,
            "删除模板",
            f"确定要删除模板 '{template.metadata.name}' 吗?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self._template_manager.delete_template(self._selected_template_id)
                self._selected_template_id = None
            except TemplateNotFoundError:
                QMessageBox.warning(self, "警告", "模板不存在")
            except Exception as e:
                QMessageBox.critical(
                    self, "错误", f"删除模板失败: {str(e)}"
                )

    def _on_template_edited(self, template_id):
        """模板编辑完成"""
        self._selected_template_id = template_id
        self._refresh_template_list()

    def _on_load_clicked(self):
        """加载选中的模板"""
        if not self._selected_template_id:
            QMessageBox.warning(self, "警告", "请先选择一个模板")
            return
        self.template_applied.emit(self._selected_template_id)
        self.accept()

    @staticmethod
    def _count_variables(config: dict) -> int:
        """计算配置中的变量数量"""
        var_set = set()
        plots = config.get("plots", [])
        for plot in plots:
            curves = plot.get("curves", [])
            var_set.update(curves)
        return len(var_set)
