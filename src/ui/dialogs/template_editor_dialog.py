"""模板编辑器对话框 - YAML 编辑和网格预览"""

from __future__ import annotations
import json
import yaml
from typing import Optional
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
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
from src.core.template_models import (
    count_template_plots,
    count_template_variables,
    iter_template_pages,
)
from src.core.plot_config import (
    PlotSessionConfig,
    TemplateNameConflictError,
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
        #: 保存成功时给主窗口播报用的动词，见 `_on_save_clicked`
        self.saved_summary = ""
        #: 最近一次解析出的预览页（原始 dict，保留手写 YAML 的非法值供校验提示）
        self._preview_config: dict = {}
        self._preview_pages: list[dict] = []
        #: 首次载入按 active_page 选页；之后逐键刷新保留当前页，避免跳回第 1 页
        self._preview_initialized = False
        self._suppress_page_combo = False

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

        # 页选择：放在 _preview_layout **之外**，_clear_preview 只会清网格，
        # 逐键刷新不会把下拉框一起 deleteLater。
        page_row = QHBoxLayout()
        page_row.addWidget(QLabel("页面:"))
        self._page_combo = QComboBox()
        self._page_combo.setMinimumWidth(260)
        self._page_combo.setToolTip("按页预览；切换页不会修改左侧 YAML")
        page_row.addWidget(self._page_combo, 1)
        preview_layout.addLayout(page_row)

        # 预览区域（只渲染当前选中页）
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._preview_widget = QWidget()
        self._preview_layout = QGridLayout(self._preview_widget)
        self._preview_layout.setSpacing(5)
        self._preview_layout.setContentsMargins(10, 10, 10, 10)
        scroll.setWidget(self._preview_widget)
        preview_layout.addWidget(scroll)

        # 统计信息：cells 是当前页，vars/plots 是全模板跨页
        self._stats_label = QLabel(
            "pages: 0 | cells: 0 × 0 | vars 全模板: 0 | plots 全模板: 0"
        )
        self._stats_label.setWordWrap(True)
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
    def _yaml_scalar(value) -> str:
        """输出合法的 YAML 标量/流式集合。

        ``yaml.safe_dump(scalar).strip()`` 对普通字符串会留下 ``...`` 文档结束符，
        插入手拼 YAML 后会触发 ParserError。JSON 标量是 YAML 的合法子集
        （带引号字符串、true/false/null、数字、数组），这里统一走 json。
        """
        try:
            return json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            return json.dumps(str(value), ensure_ascii=False)

    @staticmethod
    def _yaml_comment_text(value) -> str:
        """注释文本不能带换行，否则会把后面的 YAML 注释掉一半。"""
        return str(value).replace("\r", " ").replace("\n", " ")

    @staticmethod
    def _safe_int(value, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _generate_yaml_with_comments(cls, config: dict) -> str:
        """生成带中文注释的 v2 YAML。

        使用 ``iter_template_pages`` 的原始 dict，而不是
        ``PlotSessionConfig.from_dict``：模板可能被手工编辑过，非法
        ``layout_rows: abc`` 不应让编辑器在打开时就抛 ValueError。
        """
        config = config if isinstance(config, dict) else {}
        pages = list(iter_template_pages(config))
        lines: list[str] = []

        if config.get("created_at"):
            lines.append("# 模板创建时间")
            lines.append(f"created_at: {cls._yaml_scalar(config['created_at'])}")
            lines.append("")

        lines.append("# 时间轴缩放因子、偏移量（用于调整时间轴显示比例）")
        lines.append(
            f"time_factor: {cls._yaml_scalar(config.get('time_factor', 1.0))}"
        )
        lines.append(
            f"time_offset: {cls._yaml_scalar(config.get('time_offset', 0.0))}"
        )
        lines.append("")

        active_page = min(
            max(cls._safe_int(config.get("active_page", 0)), 0),
            len(pages) - 1,
        )
        lines.append("# 应用模板后激活的页序号（0 起）")
        lines.append(f"active_page: {active_page}")

        cursor_mode = config.get("cursor_mode")
        mark_range = config.get("saved_mark_range")
        mark_scalar = None
        if isinstance(mark_range, (list, tuple)) and len(mark_range) >= 2:
            try:
                mark_scalar = cls._yaml_scalar(
                    [float(mark_range[0]), float(mark_range[1])]
                )
            except (TypeError, ValueError):
                mark_scalar = None
        if cursor_mode is not None or mark_scalar is not None:
            lines.append("")
            lines.append("# 可选视图态：缺省表示应用模板时保留当前值")
            if cursor_mode is not None:
                lines.append(f"cursor_mode: {cls._yaml_scalar(cursor_mode)}")
            if mark_scalar is not None:
                lines.append(f"saved_mark_range: {mark_scalar}")
        lines.append("")

        lines.append("# 多绘图页（按标签顺序；每页有自己的布局与子图）")
        lines.append("pages:")
        for page_index, page in enumerate(pages, start=1):
            raw_title = page.get("title", "")
            display_title = str(raw_title) or f"page {page_index}"
            lines.append(
                f"  - # 第 {page_index} 页："
                f"{cls._yaml_comment_text(display_title)}"
            )
            lines.append(f"    title: {cls._yaml_scalar(raw_title)}")
            lines.append(
                f"    layout_rows: "
                f"{cls._yaml_scalar(page.get('layout_rows', 1))}"
            )
            lines.append(
                f"    layout_cols: "
                f"{cls._yaml_scalar(page.get('layout_cols', 1))}"
            )

            factors = page.get("row_height_factors") or {}
            if isinstance(factors, dict) and factors:
                lines.append("    row_height_factors:  # 行高百分比，缺省 100")
                for row, percentage in factors.items():
                    row_text = (
                        str(row)
                        if isinstance(row, int)
                        else cls._yaml_scalar(row)
                    )
                    lines.append(
                        f"      {row_text}: {cls._yaml_scalar(percentage)}"
                    )

            plots = page.get("plots") or []
            if not isinstance(plots, list) or not plots:
                lines.append("    plots: []")
                continue
            lines.append("    plots:")
            for plot_index, plot in enumerate(plots, start=1):
                plot = plot if isinstance(plot, dict) else {}
                lines.append(
                    f"      - # 第 {plot_index} 个子图"
                    "（按从左到右、从上到下）"
                )
                curves = plot.get("curves") or []
                if isinstance(curves, str):
                    curves = [curves]
                if not isinstance(curves, list) or not curves:
                    lines.append("        curves: []")
                    continue
                lines.append("        curves:")
                for curve in curves:
                    lines.append(f"          - {cls._yaml_scalar(curve)}")

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
        self._page_combo.currentIndexChanged.connect(self._on_preview_page_changed)
        # 显式包一层：clicked(bool) 的 checked 不能落到 save_as 形参上
        self._save_btn.clicked.connect(lambda checked=False: self._on_save_clicked())
        self._saveas_btn.clicked.connect(
            lambda checked=False: self._on_save_clicked(save_as=True)
        )
        self._cancel_btn.clicked.connect(self.reject)

    @staticmethod
    def _page_plot_count(page: dict) -> int:
        return sum(
            1
            for plot in (page.get("plots") or [])
            if isinstance(plot, dict) and plot.get("curves")
        )

    def _update_preview(self):
        """解析 YAML，按 active_page/当前选择刷新页下拉与预览。"""
        try:
            yaml_str = self._yaml_edit.toPlainText()
            if not yaml_str.strip():
                self._clear_preview()
                return

            config = yaml.safe_load(yaml_str)
            if not isinstance(config, dict) or not config:
                self._clear_preview()
                return

            pages = list(iter_template_pages(config))
            if not pages:
                self._clear_preview()
                return

            previous = self._page_combo.currentIndex()
            if self._preview_initialized:
                target = previous if previous >= 0 else 0
            else:
                target = self._safe_int(config.get("active_page", 0))
            target = min(max(target, 0), len(pages) - 1)

            self._preview_config = config
            self._preview_pages = pages
            self._rebuild_page_combo(pages, target)
            self._preview_initialized = True
            self._render_preview_page(target)

        except yaml.YAMLError as e:
            self._clear_preview()
            self._stats_label.setText(f"⚠️ YAML 格式错误: {str(e)}")
        except Exception as e:
            self._clear_preview()
            self._stats_label.setText(f"⚠️ 错误: {str(e)}")

    def _rebuild_page_combo(self, pages: list[dict], target: int) -> None:
        """重建页下拉；保留 target，且屏蔽中间 clear/add 的伪信号。"""
        self._suppress_page_combo = True
        try:
            self._page_combo.clear()
            for index, page in enumerate(pages, start=1):
                title = self._yaml_comment_text(
                    page.get("title") or f"page {index}"
                )
                rows = page.get("layout_rows", 1)
                cols = page.get("layout_cols", 1)
                label = f"{index}. {title} · {rows}×{cols}"
                plot_count = self._page_plot_count(page)
                if plot_count:
                    label += f" · {plot_count} 个子图"
                self._page_combo.addItem(label)
            self._page_combo.setCurrentIndex(target)
            self._page_combo.setEnabled(len(pages) > 1)
        finally:
            self._suppress_page_combo = False

    def _on_preview_page_changed(self, index: int) -> None:
        if self._suppress_page_combo or index < 0:
            return
        self._render_preview_page(index)

    def _render_preview_page(self, page_index: int) -> None:
        """渲染选中页；超出 PREVIEW_MAX_CELLS 时只提示、不建控件。"""
        if not (0 <= page_index < len(self._preview_pages)):
            self._clear_preview_grid()
            return
        page = self._preview_pages[page_index]
        rows = page.get("layout_rows", 1)
        cols = page.get("layout_cols", 1)

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
            self._clear_preview_grid()
            self._stats_label.setText(
                f"⚠️ 布局过大或非法（第 {page_index + 1} 页 "
                f"{rows} × {cols}），仅保存时校验，不渲染预览"
            )
            return

        self._clear_preview_grid()
        plots = page.get("plots") or []
        if not isinstance(plots, list):
            plots = []

        for row in range(rows):
            for col in range(cols):
                cell_index = row * cols + col
                frame = QFrame()
                frame.setFrameStyle(QFrame.Shape.Box)
                frame.setStyleSheet("background-color: #f0f0f0;")

                frame_layout = QVBoxLayout(frame)
                frame_layout.setContentsMargins(5, 5, 5, 5)

                plot_config = (
                    plots[cell_index]
                    if cell_index < len(plots)
                    and isinstance(plots[cell_index], dict)
                    else {}
                )
                curves = plot_config.get("curves") or []
                if isinstance(curves, str):
                    curves = [curves]
                if not isinstance(curves, list):
                    curves = []

                if curves:
                    label_text = "\n".join(str(curve) for curve in curves)
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

                self._preview_layout.addWidget(frame, row, col)

        title = self._yaml_comment_text(
            page.get("title") or f"page {page_index + 1}"
        )
        var_count = count_template_variables(self._preview_config)
        plot_count = count_template_plots(self._preview_config)
        self._stats_label.setText(
            f"pages: {len(self._preview_pages)} | 当前 {title} · "
            f"cells: {rows} × {cols} | vars 全模板: {var_count} | "
            f"plots 全模板: {plot_count}"
        )

    def _clear_preview_grid(self) -> None:
        """只清网格，不动页下拉与统计（逐键刷新/切页都会调用）。"""
        while self._preview_layout.count():
            item = self._preview_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _clear_preview(self):
        """清空网格、页下拉与统计。"""
        self._clear_preview_grid()
        self._preview_config = {}
        self._preview_pages = []
        self._suppress_page_combo = True
        try:
            self._page_combo.clear()
            self._page_combo.setEnabled(False)
        finally:
            self._suppress_page_combo = False
        self._stats_label.setText(
            "pages: 0 | cells: 0 × 0 | vars 全模板: 0 | plots 全模板: 0"
        )

    @staticmethod
    def _validate_config_for_save(config: dict) -> str | None:
        """保存前的结构校验；返回错误文案，None 表示通过。

        预览必须容错（逐键触发），保存则要在 ``PlotSessionConfig.from_dict``
        之前把页级非法值拦成明确提示，而不是一句 "invalid literal for int()"。
        """
        if not isinstance(config, dict):
            return "YAML 顶层必须是映射"
        if not config:
            return "YAML 内容不能为空"

        raw_pages = config.get("pages")
        page_dicts: list[dict] = []
        if isinstance(raw_pages, list):
            for index, page in enumerate(raw_pages, start=1):
                if not isinstance(page, dict):
                    return f"第 {index} 页配置不是字典"
                page_dicts.append(page)

        has_top_layout = (
            "layout_rows" in config and "layout_cols" in config
        )
        if not has_top_layout and not page_dicts:
            return "配置缺少布局信息（layout_rows/cols 或非空 pages）"

        for index, page in enumerate(page_dicts, start=1):
            for key in ("layout_rows", "layout_cols"):
                if key not in page:
                    continue
                value = page[key]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value <= 0
                ):
                    return f"第 {index} 页 {key} 必须是正整数"
            factors = page.get("row_height_factors")
            if factors is None:
                continue
            if not isinstance(factors, dict):
                return f"第 {index} 页 row_height_factors 必须是字典"
            for row, percentage in factors.items():
                try:
                    int(row)
                    int(percentage)
                except (TypeError, ValueError):
                    return f"第 {index} 页 row_height_factors 键/值必须是整数"

        if not page_dicts and has_top_layout:
            for key in ("layout_rows", "layout_cols"):
                value = config.get(key)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value <= 0
                ):
                    return f"{key} 必须是正整数"

        if "active_page" in config:
            try:
                int(config["active_page"] or 0)
            except (TypeError, ValueError):
                return "active_page 必须是整数"
        return None

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
            error = self._validate_config_for_save(config)
            if error:
                QMessageBox.warning(self, "警告", error)
                return

            session_config = PlotSessionConfig.from_dict(config)
            page_count = len(session_config.normalized_pages())
            session_config.active_page = min(
                max(session_config.active_page, 0), page_count - 1
            )

            # 同名冲突检查：由用户选择覆盖 / 另存新名 / 取消
            conflict = self._find_conflict_template(name, edit_id)
            success_message = None
            if conflict is not None:
                action = self._ask_conflict_resolution(name, edit_id)
                if action == "cancel":
                    return
                if action == "overwrite":
                    edit_id = conflict.metadata.id
                    success_message = "已覆盖模板"
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
            # 成功不再弹框（点 OK 只是为了关掉一个已成立的事实），改由主窗口
            # 播到状态栏右区。这里只给动词、不带模板名：名字由那一条拼，
            # 拼成"已覆盖模板：X · 3 个变量 / 2 个子图"
            self.saved_summary = success_message or (
                "模板已更新" if edit_id else "模板已保存"
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
