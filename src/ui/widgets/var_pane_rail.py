"""变量区折叠态细边栏（D1）。

左栏折叠后剩下的 28px 竖条：» 按钮＋竖排「变量」二字常驻，**整条**都是
展开热区（点击任意位置都发 ``expand_requested``）。恢复入口是界面的一部分
而不是纪律问题 —— 这正是细边栏方案对「完全隐藏」的本质优势，折叠态才敢
持久化（D12）。

本体只发信号，折叠/展开的状态迁移统一走
``LayoutManager.set_var_pane_collapsed``，播报也收在那里。
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QToolButton, QVBoxLayout, QWidget

from src.core.config import VAR_PANE_RAIL_WIDTH


class VarPaneRail(QWidget):
    """折叠态细边栏：固定 28px 宽，整条可点。"""

    expand_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(VAR_PANE_RAIL_WIDTH)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("展开变量区 (Ctrl+B)")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(6)

        self.expand_btn = QToolButton(self)
        self.expand_btn.setText("»")
        self.expand_btn.setToolTip("展开变量区 (Ctrl+B)")
        self.expand_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.expand_btn.setFixedWidth(VAR_PANE_RAIL_WIDTH - 4)
        self.expand_btn.clicked.connect(self.expand_requested)
        layout.addWidget(self.expand_btn, 0, Qt.AlignmentFlag.AlignHCenter)

        # 竖排「变量」：一字一行。QLabel 默认不吃鼠标点击，事件会冒到
        # 本体的 mousePressEvent —— 文字区也就是热区的一部分
        self._title_label = QLabel("变\n量", self)
        self._title_label.setAlignment(
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop
        )
        layout.addWidget(self._title_label, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addStretch(1)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.expand_requested.emit()
            event.accept()
            return
        super().mousePressEvent(event)