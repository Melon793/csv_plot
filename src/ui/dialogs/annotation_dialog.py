"""标注属性对话框 —— 文字内容 / 颜色 / 线宽 / 字号 / 填充与透明度

取值一律从模型读、按模型写：色值格式与数值范围由 core.annotation_models 收拢，
这里只负责收集用户意图（result_changes 返回的字典直接喂给
AnnotationManager.apply_style，再走一遍模型层校验）。
"""

from __future__ import annotations

from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from src.core.annotation_models import (
    FILL_ALPHA_RANGE,
    FONT_SIZE_RANGE,
    MAX_TEXT_LENGTH,
    STROKE_WIDTH_RANGE,
)

FILL_COLOR_FALLBACK = "#FFEE88"  # "无填充"取消勾选时的起始色


class AnnotationDialog(QDialog):
    """单条标注的属性编辑对话框"""

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self._model = model
        self.setWindowTitle("标注属性")

        self._text_edit = QPlainTextEdit(model.text)
        self._text_edit.setPlaceholderText("输入标注文字（可多行）")
        self._text_edit.setTabChangesFocus(True)

        self._stroke = QColor(model.stroke)
        self._stroke_btn = QPushButton()
        self._stroke_btn.clicked.connect(self._pick_stroke)
        self._paint_color_button(self._stroke_btn, self._stroke)

        self._width_spin = QSpinBox()
        if model.kind == "text":
            # 文字允许 0 = 无外边框；其它类型图元层会钳回 1，不让用户设置
            # 一个"看起来生效了其实没有"的值
            self._width_spin.setRange(*STROKE_WIDTH_RANGE)
            self._width_spin.setToolTip("0 = 无文字外边框")
        else:
            self._width_spin.setRange(1, STROKE_WIDTH_RANGE[1])
        self._width_spin.setValue(int(model.stroke_width))

        self._font_spin = QSpinBox()
        self._font_spin.setRange(*FONT_SIZE_RANGE)
        self._font_spin.setValue(int(model.font_size))
        self._font_spin.setSuffix(" px")

        self._fill_enabled = QCheckBox("启用填充")
        self._fill_enabled.setChecked(bool(model.fill))
        self._fill_enabled.toggled.connect(self._sync_fill_enabled)
        self._fill = QColor(model.fill or FILL_COLOR_FALLBACK)
        self._fill_btn = QPushButton()
        self._fill_btn.clicked.connect(self._pick_fill)
        self._paint_color_button(self._fill_btn, self._fill)

        fill_row = QHBoxLayout()
        fill_row.addWidget(self._fill_btn)
        fill_row.addWidget(self._fill_enabled)
        fill_row.addStretch(1)

        self._alpha_spin = QSpinBox()
        self._alpha_spin.setRange(*FILL_ALPHA_RANGE)
        self._alpha_spin.setValue(int(model.fill_alpha))
        self._alpha_spin.setToolTip("0 = 全透明，255 = 不透明")

        form = QFormLayout()
        if model.kind == "text":
            form.addRow("文字内容:", self._text_edit)
        form.addRow("颜色:", self._stroke_btn)
        form.addRow("线宽:", self._width_spin)
        if model.kind == "text":
            form.addRow("字号:", self._font_spin)
        form.addRow("填充:", fill_row)
        form.addRow("填充透明度:", self._alpha_spin)

        hint = QLabel(f"类型：{model.kind}　（几何请在图上直接拖拽调整）")
        hint.setStyleSheet("color: gray;")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(hint)
        layout.addWidget(buttons)

        self._sync_fill_enabled(self._fill_enabled.isChecked())

    # ------------------------------------------------------------------
    # 交互
    # ------------------------------------------------------------------
    def _paint_color_button(self, button: QPushButton, color: QColor) -> None:
        button.setFixedWidth(72)
        button.setText(color.name())
        button.setStyleSheet(
            f"background-color: {color.name()};"
            f"color: {'#000000' if color.lightness() > 128 else '#FFFFFF'};"
        )

    def _pick_stroke(self) -> None:
        chosen = QColorDialog.getColor(self._stroke, self, "选择标注颜色")
        if chosen.isValid():
            self._stroke = chosen
            self._paint_color_button(self._stroke_btn, chosen)

    def _pick_fill(self) -> None:
        chosen = QColorDialog.getColor(self._fill, self, "选择填充颜色")
        if chosen.isValid():
            self._fill = chosen
            self._paint_color_button(self._fill_btn, chosen)
            self._fill_enabled.setChecked(True)

    def _sync_fill_enabled(self, enabled: bool) -> None:
        self._fill_btn.setEnabled(bool(enabled))
        self._alpha_spin.setEnabled(bool(enabled))

    # ------------------------------------------------------------------
    # 结果
    # ------------------------------------------------------------------
    def result_changes(self) -> dict:
        """返回可直接交给 AnnotationManager.apply_style 的变更字典"""
        changes: dict = {
            "stroke": self._stroke.name(),
            "stroke_width": self._width_spin.value(),
            "fill": self._fill.name() if self._fill_enabled.isChecked() else None,
            "fill_alpha": self._alpha_spin.value(),
            "font_size": self._font_spin.value(),
        }
        if self._model.kind == "text":
            changes["text"] = self._text_edit.toPlainText()[:MAX_TEXT_LENGTH]
        return changes
