"""模板编辑器对话框 - YAML 编辑和网格预览"""

from __future__ import annotations
import yaml
from pathlib import Path
from datetime import datetime
from typing import Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QTextEdit,
    QMessageBox,
    QSplitter,
    QGroupBox,
    QGridLayout,
    QFrame,
    QScrollArea,
    QWidget,
)
from src.core.template_manager import TemplateManager
from src.core.template_models import PlotTemplate, TemplateMetadata, count_template_variables
from src.core.plot_config import (
    PlotSessionConfig,
    PlotConfig,
    TemplateNameConflictError,
    TemplateValidationError,
)
from src.core.logger import get_logger


logger = get_logger(__name__)


class TemplateEditorDialog(QDialog):
    """模板编辑器对话框 - YAML 编辑和网格预览"""

    template_saved = Signal(str)  # template_id

    # 预览网格上限（行列乘积）：YAML 编辑区逐键触发预览渲染，超限只提示不渲染
    PREVIEW_MAX_CELLS = 144

    DEFAULT_TEMPLATE_CONFIG = {
        "layout_rows": 2,
        "layout_cols": 2,
        "time_factor": 1.0,
        "time_offset": 0.0,
        "plots": [
            {"curves": ["rpm", "speed"]},
            {"curves": ["throttle"]},
            {"curves": ["voltage"]},
            {"curves": ["current"]},
        ],
    }

    def __init__(
        self,
        template_manager: TemplateManager,
        current_config: Optional[PlotSessionConfig] = None,
        edit_template_id: Optional[str] = None,
        initial_name: str = "",
        initial_desc: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self._template_manager = template_manager
        self._current_config = current_config
        self._edit_template_id = edit_template_id
        self._template = None
        self._initial_name = initial_name
        self._initial_desc = initial_desc

        if edit_template_id:
            self._template = self._template_manager.get_template(edit_template_id)
            self.setWindowTitle("✏️ 编辑模板")
        elif current_config:
            self.setWindowTitle("💾 新建模板")
        else:
            self.setWindowTitle("✏️ 新建空白模板")

        self.resize(1200, 700)
        self._setup_ui()
        self._load_initial_content()
        self._connect_signals()

    def _setup_ui(self):
        """设置 UI"""
        layout = QVBoxLayout(self)

        # 名称和描述输入
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("名称:"))
        self._name_edit = QLineEdit()
        input_layout.addWidget(self._name_edit)

        input_layout.addWidget(QLabel("描述:"))
        self._desc_edit = QLineEdit()
        input_layout.addWidget(self._desc_edit)
        layout.addLayout(input_layout)

        # 分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左侧：YAML 编辑器
        yaml_group = QGroupBox("YAML 编辑器")
        yaml_layout = QVBoxLayout(yaml_group)
        self._yaml_edit = QTextEdit()
        self._yaml_edit.setPlaceholderText("# 在这里编辑 YAML 配置...")
        yaml_layout.addWidget(self._yaml_edit)
        splitter.addWidget(yaml_group)

        # 右侧：布局预览
        preview_group = QGroupBox("布局预览")
        preview_layout = QVBoxLayout(preview_group)

        # 预览区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._preview_widget = QWidget()
        self._preview_layout = QGridLayout(self._preview_widget)
        self._preview_layout.setSpacing(5)
        self._preview_layout.setContentsMargins(10, 10, 10, 10)
        scroll.setWidget(self._preview_widget)
        preview_layout.addWidget(scroll)

        # 统计信息
        self._stats_label = QLabel("cells: 0 × 0 | vars: 0 | plots: 0")
        preview_layout.addWidget(self._stats_label)
        splitter.addWidget(preview_group)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter)

        # 按钮栏
        button_layout = QHBoxLayout()
        self._save_btn = QPushButton("保存")
        self._saveas_btn = QPushButton("另存为...")
        self._cancel_btn = QPushButton("取消")
        button_layout.addStretch()
        button_layout.addWidget(self._save_btn)
        button_layout.addWidget(self._saveas_btn)
        button_layout.addWidget(self._cancel_btn)
        layout.addLayout(button_layout)

    @staticmethod
    def _generate_yaml_with_comments(config: dict) -> str:
        """生成带中文注释的 YAML 字符串"""
        lines = []
        
        # created_at
        if "created_at" in config:
            lines.append("# 模板创建时间")
            lines.append(f'created_at: "{config["created_at"]}"')
            lines.append("")
        
        # layout
        lines.append("# 绘图区域布局行数/列数")
        lines.append(f'layout_rows: {config.get("layout_rows", 1)}')
        lines.append(f'layout_cols: {config.get("layout_cols", 1)}')
        lines.append("")
        
        # time
        lines.append("# 时间轴缩放因子、偏移量（用于调整时间轴显示比例）")
        lines.append(f'time_factor: {config.get("time_factor", 1.0)}')
        lines.append(f'time_offset: {config.get("time_offset", 0.0)}')
        lines.append("")
        
        # plots
        lines.append("# 各子图配置列表（按从左到右、从上到下的顺序）")
        plots = config.get("plots", []) or []
        if plots:
            lines.append("plots:")
            for i, plot in enumerate(plots):
                lines.append(f"  - # 第 {i+1} 个子图的曲线变量名列表")
                lines.append("    curves:")
                curves = plot.get("curves", []) if plot else []
                if curves:
                    for curve in curves:
                        lines.append(f"      - {curve}")
                else:
                    lines.append("      []")
        else:
            lines.append("plots: []")
        
        return "\n".join(lines) + "\n"

    def _load_initial_content(self):
        """加载初始内容"""
        if self._template:
            # 编辑现有模板
            self._name_edit.setText(self._template.metadata.name)
            self._desc_edit.setText(self._template.metadata.description)
            yaml_str = self._generate_yaml_with_comments(self._template.config)
            self._yaml_edit.setPlainText(yaml_str)
        elif self._current_config:
            # 从当前配置创建新模板
            self._name_edit.setText(self._initial_name)
            self._desc_edit.setText(self._initial_desc)
            yaml_str = self._generate_yaml_with_comments(self._current_config.to_dict())
            self._yaml_edit.setPlainText(yaml_str)
        else:
            # 新建空白模板
            self._name_edit.setText(self._initial_name)
            self._desc_edit.setText(self._initial_desc)
            yaml_str = self._generate_yaml_with_comments(self.DEFAULT_TEMPLATE_CONFIG)
            self._yaml_edit.setPlainText(yaml_str)

        self._update_preview()

    def _connect_signals(self):
        """连接信号"""
        self._yaml_edit.textChanged.connect(self._update_preview)
        # 显式包一层：clicked(bool) 的 checked 不能落到 save_as 形参上
        self._save_btn.clicked.connect(lambda checked=False: self._on_save_clicked())
        self._saveas_btn.clicked.connect(
            lambda checked=False: self._on_save_clicked(save_as=True)
        )
        self._cancel_btn.clicked.connect(self.reject)

    def _update_preview(self):
        """更新预览"""
        try:
            # 解析 YAML
            yaml_str = self._yaml_edit.toPlainText()
            if not yaml_str.strip():
                self._clear_preview()
                return

            config = yaml.safe_load(yaml_str)
            if not config:
                self._clear_preview()
                return

            # 渲染网格
            rows = config.get("layout_rows", 1)
            cols = config.get("layout_cols", 1)
            plots = config.get("plots", []) or []

            # P0-7: 预览挂在 textChanged 上逐键触发，必须先钳制网格规模 ——
            # 用户把 layout_rows 多打一个 0 就会同步创建 rows×cols 个
            # QFrame+QVBoxLayout+QLabel（1000×1000 = 100 万控件）冻结 UI
            # 线程甚至 OOM，只能强杀进程。保存路径有 PlotSessionConfig
            # 校验，但预览发生在保存之前，此处不兜住就无人兜。
            if (
                not isinstance(rows, int)
                or not isinstance(cols, int)
                or rows <= 0
                or cols <= 0
                or rows * cols > self.PREVIEW_MAX_CELLS
            ):
                self._clear_preview()
                self._stats_label.setText(
                    f"⚠️ 布局过大或非法（{rows} × {cols}），"
                    f"仅保存时校验，不渲染预览"
                )
                return

            # 清除现有预览
            self._clear_preview()

            # 渲染新预览
            for row in range(rows):
                for col in range(cols):
                    index = row * cols + col
                    frame = QFrame()
                    frame.setFrameStyle(QFrame.Shape.Box)
                    frame.setStyleSheet("background-color: #f0f0f0;")

                    frame_layout = QVBoxLayout(frame)
                    frame_layout.setContentsMargins(5, 5, 5, 5)

                    if index < len(plots):
                        plot_config = plots[index]
                        curves = plot_config.get("curves", [])
                        if curves:
                            label_text = "\n".join(curves)
                            label = QLabel(label_text)
                            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                            label.setWordWrap(True)
                            if len(curves) > 1:
                                frame.setStyleSheet(
                                    "background-color: #e8f4fc; border: 1px solid #99c2ff;"
                                )
                            else:
                                frame.setStyleSheet(
                                    "background-color: #e8fce8; border: 1px solid #99ff99;"
                                )
                            frame_layout.addWidget(label)
                        else:
                            label = QLabel("(空)")
                            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                            label.setStyleSheet("color: #999;")
                            frame_layout.addWidget(label)
                    else:
                        label = QLabel("(空)")
                        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                        label.setStyleSheet("color: #999;")
                        frame_layout.addWidget(label)

                    self._preview_layout.addWidget(frame, row, col)

            # 更新统计
            var_count = count_template_variables(config)
            plot_count = len(plots)
            self._stats_label.setText(
                f"cells: {rows} × {cols} | vars: {var_count} | plots: {plot_count}"
            )

        except yaml.YAMLError as e:
            self._stats_label.setText(f"⚠️ YAML 格式错误: {str(e)}")
        except Exception as e:
            self._stats_label.setText(f"⚠️ 错误: {str(e)}")

    def _clear_preview(self):
        """清除预览"""
        while self._preview_layout.count():
            item = self._preview_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._stats_label.setText("cells: 0 × 0 | vars: 0 | plots: 0")

    def _on_save_clicked(self, save_as: bool = False):
        """保存按钮点击

        `save_as` 只决定**本次**写入哪个模板：编辑目标走局部 `edit_id`，
        名称为空 / YAML 校验失败 / 冲突框选「取消」而 return 时，实例上的
        `_edit_template_id` 保持原样，之后的「保存」仍更新原模板。
        """
        edit_id = None if save_as else self._edit_template_id
        try:
            name = self._name_edit.text().strip()
            if not name:
                QMessageBox.warning(self, "警告", "请输入模板名称")
                return

            # 解析 YAML
            yaml_str = self._yaml_edit.toPlainText()
            config = yaml.safe_load(yaml_str)
            if not config:
                QMessageBox.warning(self, "警告", "YAML 内容不能为空")
                return

            # 验证配置结构
            if "layout_rows" not in config or "layout_cols" not in config:
                QMessageBox.warning(self, "警告", "配置缺少布局信息")
                return

            session_config = PlotSessionConfig.from_dict(config)

            # 同名冲突检查：由用户选择覆盖 / 另存新名 / 取消
            conflict = self._find_conflict_template(name, edit_id)
            success_message = None
            if conflict is not None:
                action = self._ask_conflict_resolution(name, edit_id)
                if action == "cancel":
                    return
                if action == "overwrite":
                    edit_id = conflict.metadata.id
                    success_message = f"已覆盖模板 [{name}]"
                else:
                    # action == "rename"：自动改用不冲突的新名称继续保存
                    name = self._suggest_unique_name(name)
                    self._name_edit.setText(name)

            template = self._template_manager.save_template(
                session_config,
                name,
                self._desc_edit.text().strip(),
                template_id=edit_id,
            )
            if save_as:
                # 写盘成功后才把编辑目标切到新模板（并丢弃载入表单用的旧快照）
                self._edit_template_id = template.metadata.id
                self._template = None
            QMessageBox.information(
                self,
                "成功",
                success_message or ("模板已更新" if edit_id else "模板已保存"),
            )
            try:
                self.template_saved.emit(template.metadata.id)
            finally:
                self.accept()

        except TemplateNameConflictError:
            QMessageBox.warning(self, "警告", "模板名称已存在")
        except yaml.YAMLError as e:
            QMessageBox.warning(self, "警告", f"YAML 格式错误: {str(e)}")
        except Exception as e:
            logger.error(f"Save error: {e}")
            QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")

    def _find_conflict_template(self, name: str, edit_id: str | None = None):
        """查找同名模板（排除本次写入的模板自身），无冲突返回 None"""
        for t in self._template_manager.get_all_templates():
            if t.metadata.name == name and t.metadata.id != edit_id:
                return t
        return None

    def _suggest_unique_name(self, base: str) -> str:
        """生成不冲突的新名称：名称 (2)、名称 (3)..."""
        i = 2
        new_name = f"{base} ({i})"
        while self._template_manager.exists(new_name):
            i += 1
            new_name = f"{base} ({i})"
        return new_name

    def _ask_conflict_resolution(self, name: str, edit_id: str | None = None) -> str:
        """同名冲突时的选择：overwrite / rename / cancel

        编辑模式下不提供覆盖选项，避免误覆盖另一个模板。
        """
        box = QMessageBox(self)
        box.setWindowTitle("模板名称冲突")
        box.setIcon(QMessageBox.Icon.Warning)
        overwrite_btn = None
        if edit_id:
            box.setText(f"已存在同名模板 [{name}]，请另存为新名称或取消后修改名称")
        else:
            box.setText(f"已存在同名模板 [{name}]，如何处理？")
            overwrite_btn = box.addButton(
                "覆盖原模板", QMessageBox.ButtonRole.DestructiveRole
            )
        rename_btn = box.addButton("另存新名", QMessageBox.ButtonRole.AcceptRole)
        cancel_btn = box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(cancel_btn)
        box.exec()
        clicked = box.clickedButton()
        if overwrite_btn is not None and clicked is overwrite_btn:
            return "overwrite"
        if clicked is rename_btn:
            return "rename"
        return "cancel"
