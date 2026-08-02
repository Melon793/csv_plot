"""LayoutGridSelector - 网格布局选择对话框

类似 Word 插入表格的交互方式，通过鼠标悬停预览、点击确认布局。
替代旧版基于 QSpinBox 的 LayoutInputDialog。
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
    QGridLayout,
)

# === 视觉规范常量（见设计文档 §4.2 / §10） ===
_STYLE_CELL_DEFAULT = "background-color: #E8E8E8; border: 1px solid #CCCCCC;"
_STYLE_CELL_HIGHLIGHT = "background-color: #4A90D9; border: 1px solid #FFFFFF;"
_STYLE_HINT = (
    "background-color: #F5F5F5; border: 1px solid #DDDDDD; "
    "padding: 6px 10px; color: #333333;"
)
_STYLE_STATIC_HINT = "color: #888888; font-size: 12px;"

# === 单元格尺寸（见设计文档 §4.3） ===
_CELL_SIZE = 40
_CELL_SPACING = 4


class CellButton(QPushButton):
    """单个网格单元格按钮

    仅负责悬停/点击事件上报；网格离开检测由父级 GridContainerWidget 统一处理。
    """

    cell_clicked = Signal(int, int)  # (row, col) 点击信号
    cell_hovered = Signal(int, int)  # (row, col) 悬停信号

    def __init__(self, row: int, col: int, parent=None):
        # 必须先调用父类构造，否则 setFixedSize/setStyleSheet 会抛异常
        super().__init__(parent)
        self.row = row
        self.col = col
        self.setFixedSize(_CELL_SIZE, _CELL_SIZE)
        self.setStyleSheet(_STYLE_CELL_DEFAULT)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        # 焦点由 grid_widget 统一持有，单元格不参与 Tab 链与方向键焦点
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def enterEvent(self, event):
        self.cell_hovered.emit(self.row, self.col)
        super().enterEvent(event)

    def mousePressEvent(self, event):
        # 仅左键触发确认，符合 Word 范式与 Qt 按钮惯例
        if event.button() == Qt.MouseButton.LeftButton:
            self.cell_clicked.emit(self.row, self.col)
        # 需调用父类方法以保证事件传播
        super().mousePressEvent(event)


class GridContainerWidget(QWidget):
    """网格容器：承载所有 CellButton，负责网格级别的鼠标离开与焦点离开检测

    利用 Qt 事件传播特性——鼠标在子控件(CellButton)间移动不会触发本容器的 leaveEvent，
    只有真正离开整个容器几何范围时才触发，因此无需手动判断坐标。
    焦点离开网格时同样发射 grid_left，使高亮回退到初始预高亮状态（见设计文档 §4.3）。
    """

    grid_left = Signal()  # 鼠标离开整个网格区域 或 焦点离开网格 信号

    def leaveEvent(self, event):
        self.grid_left.emit()
        super().leaveEvent(event)

    def focusOutEvent(self, event):
        # 焦点离开网格（如 Tab 切到取消按钮）时，回退高亮到初始预高亮状态
        self.grid_left.emit()
        super().focusOutEvent(event)


class LayoutGridSelector(QDialog):
    """网格布局选择对话框

    类似 Word 插入表格的交互方式，通过鼠标悬停预览、点击确认布局。

    初始化行为:
        打开对话框时立即预高亮 (1,1)~(cur_rows, cur_cols) 单元格，
        提示框显示 "当前布局: {cur_rows}行 x {cur_cols}列"，
        使其与当前实际布局状态保持一致，便于用户参照调整。
        鼠标离开网格时回退到此初始预高亮状态，而非全部恢复灰色。

    焦点策略:
        setFocusPolicy(Qt.StrongFocus) 以接收 keyPressEvent；
        打开对话框后 setFocus() 到 grid_widget，使方向键立即可用。
    """

    def __init__(
        self,
        max_rows: int = 4,
        max_cols: int = 3,
        cur_rows: int = 3,
        cur_cols: int = 1,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("设置图表布局")

        # 钳制参数到合法范围，避免越界
        self.max_rows = max(1, max_rows)
        self.max_cols = max(1, max_cols)
        self._cur_rows = max(1, min(cur_rows, self.max_rows))
        self._cur_cols = max(1, min(cur_cols, self.max_cols))

        # 当前悬停位置（0-based），-1 表示未悬停/已离开网格
        self._hover_row = -1
        self._hover_col = -1

        # 选择结果，默认为当前布局（供 Enter 直接确认与 cancel 后兜底）
        self._result_row = self._cur_rows
        self._result_col = self._cur_cols

        # 单元格二维数组
        self.cells: list[list[CellButton]] = []

        self._build_ui()
        self._apply_initial_highlight()

        # 焦点策略：grid_widget 接收方向键
        self.grid_widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.grid_widget.setFocus()

    # === UI 构建 ===

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        # 静态提示（网格上方）—— 固定文案，告知用户操作方式
        self.static_hint = QLabel("点击单元格确认布局")
        self.static_hint.setStyleSheet(_STYLE_STATIC_HINT)
        self.static_hint.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.static_hint)

        # 网格容器
        self.grid_widget = GridContainerWidget(self)
        grid_layout = QGridLayout(self.grid_widget)
        grid_layout.setSpacing(_CELL_SPACING)
        grid_layout.setContentsMargins(0, 0, 0, 0)

        for r in range(self.max_rows):
            row_cells: list[CellButton] = []
            for c in range(self.max_cols):
                cell = CellButton(r, c, self.grid_widget)
                cell.cell_hovered.connect(self._on_cell_hovered)
                cell.cell_clicked.connect(self._on_cell_clicked)
                grid_layout.addWidget(cell, r, c)
                row_cells.append(cell)
            self.cells.append(row_cells)

        layout.addWidget(self.grid_widget)

        # 动态提示框（网格下方）—— 随鼠标状态变化
        self.hint_label = QLabel()
        self.hint_label.setStyleSheet(_STYLE_HINT)
        self.hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.hint_label)

        # 取消按钮行（仅保留取消按钮，无确定按钮）
        btn_layout = QHBoxLayout()
        btn_layout.addStretch(1)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)

        # 鼠标离开网格 -> 回退到初始预高亮
        self.grid_widget.grid_left.connect(self._on_grid_left)

        # 固定整体尺寸，避免拉伸导致网格变形（见 §4.3）
        self.setFixedSize(self.sizeHint())

    # === 初始状态 ===

    def _apply_initial_highlight(self):
        """打开对话框时预高亮 (1,1)~(cur_rows,cur_cols)，设置提示框初始文案"""
        self._hover_row = -1
        self._hover_col = -1
        self._update_highlight(-1, -1)
        self._update_hint(-1, -1)

    # === 事件处理 ===

    def _on_cell_hovered(self, row: int, col: int):
        self._hover_row = row
        self._hover_col = col
        self._update_highlight(row, col)
        self._update_hint(row, col)

    def _on_grid_left(self):
        """鼠标离开网格，回退到初始预高亮状态（非全灰）"""
        self._hover_row = -1
        self._hover_col = -1
        self._update_highlight(-1, -1)
        self._update_hint(-1, -1)

    def _on_cell_clicked(self, row: int, col: int):
        """点击确认，设置结果并关闭对话框"""
        self._result_row = row + 1
        self._result_col = col + 1
        self.accept()

    def _on_key_move(self, dr: int, dc: int):
        """方向键移动：根据 (dr,dc) 更新 _hover_row/_hover_col（钳制到合法范围）

        从当前 _hover_row/_hover_col 或初始预高亮位置出发；仅预览，不确认。
        """
        if self._hover_row < 0 or self._hover_col < 0:
            # 从初始预高亮位置出发
            new_row = self._cur_rows - 1
            new_col = self._cur_cols - 1
        else:
            new_row = self._hover_row
            new_col = self._hover_col
        new_row = max(0, min(new_row + dr, self.max_rows - 1))
        new_col = max(0, min(new_col + dc, self.max_cols - 1))
        self._hover_row = new_row
        self._hover_col = new_col
        self._update_highlight(new_row, new_col)
        self._update_hint(new_row, new_col)

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            # 确认当前高亮位置（若未悬停则确认当前布局）
            if self._hover_row < 0 or self._hover_col < 0:
                self._result_row = self._cur_rows
                self._result_col = self._cur_cols
            else:
                self._result_row = self._hover_row + 1
                self._result_col = self._hover_col + 1
            self.accept()
            return
        if key == Qt.Key.Key_Escape:
            self.reject()
            return
        if key == Qt.Key.Key_Up:
            self._on_key_move(-1, 0)
            return
        if key == Qt.Key.Key_Down:
            self._on_key_move(1, 0)
            return
        if key == Qt.Key.Key_Left:
            self._on_key_move(0, -1)
            return
        if key == Qt.Key.Key_Right:
            self._on_key_move(0, 1)
            return
        # 其余键（含 Tab）交给父类处理，由 Qt 完成网格↔取消按钮的焦点切换
        super().keyPressEvent(event)

    # === 视觉更新 ===

    def _update_highlight(self, hover_row: int, hover_col: int):
        """更新高亮：(1,1)~(row+1,col+1) 高亮，其余默认

        Args:
            hover_row/hover_col: 0-based 索引。
                - 当任一为 -1 时，回退到初始预高亮 (1,1)~(cur_rows,cur_cols)。
                - 若 cur_rows/cur_cols 也为 -1（不应发生），则全部恢复默认灰色。
        """
        if hover_row < 0 or hover_col < 0:
            eff_row = self._cur_rows - 1
            eff_col = self._cur_cols - 1
        else:
            eff_row = hover_row
            eff_col = hover_col

        for r in range(self.max_rows):
            for c in range(self.max_cols):
                cell = self.cells[r][c]
                if r <= eff_row and c <= eff_col:
                    cell.setStyleSheet(_STYLE_CELL_HIGHLIGHT)
                else:
                    cell.setStyleSheet(_STYLE_CELL_DEFAULT)

    def _update_hint(self, row: int, col: int):
        """更新提示文字。row/col 为 -1 时显示当前布局文案"""
        if row < 0 or col < 0:
            self.hint_label.setText(
                f"当前布局: {self._cur_rows}行 x {self._cur_cols}列"
            )
        else:
            self.hint_label.setText(f"{row + 1}行 x {col + 1}列")

    # === 对外接口 ===

    def values(self):
        """返回 (row, col) 元组（兼容旧 LayoutInputDialog 接口）"""
        return self._result_row, self._result_col
