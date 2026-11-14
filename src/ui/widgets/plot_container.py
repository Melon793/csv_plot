"""
PlotContainerWidget —— 包装单个 Plot，负责显示拖拽提示

此组件从 csv_plot_pyqt6.py 中迁移而来，保持原有功能不变，
仅做位置调整以符合模块化结构。
"""

from __future__ import annotations
from typing import Any
from PySide6.QtWidgets import QWidget, QSizePolicy, QVBoxLayout
from PySide6.QtCore import QMargins

from src.ui.drag_visuals import DragIndicatorOverlay


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
        self._indicator = DragIndicatorOverlay(self)
        self._indicator.hide()

    def _build_indicator_text(self, _var_names: list[str]) -> str:
        """构建拖拽指示文本。非 Shift 路径统一返回固定文本。"""
        return "释放以添加"

    def show_drag_indicator(
        self, var_names: list[str] | None = None, text_override: str | None = None
    ):
        text = text_override or self._build_indicator_text(var_names or [])
        self._indicator.set_content(text)
        self._indicator.setGeometry(self.rect())
        self._indicator.raise_()
        self._indicator.show()

    def hide_drag_indicator(self):
        self._indicator.hide()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._indicator.isVisible():
            self._indicator.setGeometry(self.rect())
