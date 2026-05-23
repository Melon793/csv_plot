"""
PlotContainerWidget —— 包装单个 Plot，负责显示拖拽提示

此组件从 csv_plot_pyqt6.py 中迁移而来，保持原有功能不变，
仅做位置调整以符合模块化结构。
"""

from __future__ import annotations
from typing import Any
from PySide6.QtWidgets import QWidget, QSizePolicy, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, QMargins


class PlotContainerWidget(QWidget):
    """包装单个 Plot, 负责显示拖拽提示"""

    def __init__(self, plot_widget: Any, parent=None):
        super().__init__(parent)
        self.plot_widget = plot_widget
        # 设置容器的大小策略，允许拉伸
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(QMargins(0, 0, 5, 5))
        layout.setSpacing(0)
        layout.addWidget(plot_widget, 1)  # 拉伸因子1，让plot占用所有空间
        self._init_indicator()

    def _init_indicator(self):
        self._indicator = QWidget(self)
        self._indicator.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        self._indicator.hide()
        self._indicator.setStyleSheet(
            "background-color: rgba(0, 120, 215, 40);"
            "border: 2px dashed #0078d7;"
            "border-radius: 12px;"
        )
        layout = QVBoxLayout(self._indicator)
        layout.setContentsMargins(16, 16, 16, 16)
        self._indicator_label = QLabel("", self._indicator)
        self._indicator_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._indicator_label.setWordWrap(True)
        self._indicator_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        self._indicator_label.setStyleSheet(
            "color: #0b365a; font-size: 16px; font-weight: bold; background: transparent; border: none;"
        )
        layout.addWidget(self._indicator_label, alignment=Qt.AlignmentFlag.AlignCenter)

    def _build_indicator_text(self, var_names: list[str]) -> str:
        has_curve = bool(getattr(self.plot_widget, "curve", None))
        has_multi_curves = bool(getattr(self.plot_widget, "curves", None))
        multi_mode = bool(
            getattr(self.plot_widget, "is_multi_curve_mode", False)
            or len(var_names) > 1
            or has_multi_curves
        )

        if multi_mode:
            return "释放以添加"

        if has_curve:
            return "释放以替换"

        return "释放以添加"

    def show_drag_indicator(
        self, var_names: list[str] | None = None, text_override: str | None = None
    ):
        text = text_override or self._build_indicator_text(var_names or [])
        self._indicator_label.setText(text)
        self._indicator.setGeometry(self.rect())
        self._indicator.raise_()
        self._indicator.show()

    def hide_drag_indicator(self):
        self._indicator.hide()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._indicator.isVisible():
            self._indicator.setGeometry(self.rect())
