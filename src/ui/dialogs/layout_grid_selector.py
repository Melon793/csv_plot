"""LayoutGridSelector - 网格布局选择对话框

类似 Word 插入表格的交互方式，通过鼠标悬停预览、点击确认布局。
替代旧版基于 QSpinBox 的 LayoutInputDialog。
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
    QGridLayout,
    QFrame,
    QLayout,
)

# === 设计令牌：色板 ===
# 唯一一份值在 src/ui/theme.py（与状态栏抽屉共用），这里只按本文件的旧名引入：
# 改名会波及 tests/component/test_layout_grid_selector_visual.py 与下面的样式串
from src.ui.theme import (
    BG as _BG,
    CHIP_CUR_BD as _CELL_CUR_BD,
    CHIP_CUR_BG as _CELL_CUR_BG,
    CHIP_OFF_BD as _CELL_OFF_BD,
    CHIP_OFF_BG as _CELL_OFF_BG,
    CHIP_SEL_BD as _CELL_SEL_BD,
    CHIP_SEL_BG_BOT as _CELL_SEL_BG_BOT,
    CHIP_SEL_BG_TOP as _CELL_SEL_BG_TOP,
    SEP as _SEP,
    TEXT_MUTED as _TEXT_MUTED,
    TEXT_PRIMARY as _TEXT_PRIMARY,
)

# === 设计令牌：度量 ===
_CELL_SIZE = 38
_CELL_SPACING = 5
_RULER_THICK = 16  # 标尺行/列的最小厚度（cocoa 上 mac 样式仍会给网格行额外加高，见下方说）
_R_CELL = 6  # 单元格圆角半径
_SWATCH_SIZE = 10  # 图例色块边长
_F_HINT_MAIN_PX = 22  # 大号结果文案的像素字号

# === 单元格三态 ===
_S_OFF = "off"
_S_CURRENT = "current"
_S_SEL = "sel"
# state -> (渐变上/纯色, 渐变下/纯色, 描边)；上下同值表示不用渐变
_STATE_PALETTE: dict[str, tuple[str, str, str]] = {
    _S_OFF: (_CELL_OFF_BG, _CELL_OFF_BG, _CELL_OFF_BD),
    _S_CURRENT: (_CELL_CUR_BG, _CELL_CUR_BG, _CELL_CUR_BD),
    _S_SEL: (_CELL_SEL_BG_TOP, _CELL_SEL_BG_BOT, _CELL_SEL_BD),
}

# === 文本与控件样式 ===
_STYLE_MUTED = f"color: {_TEXT_MUTED}; font-size: 12px;"
_STYLE_RULER = f"color: {_TEXT_MUTED}; font-size: 11px;"
_STYLE_HINT_MAIN = (
    f"color: {_TEXT_PRIMARY}; font-size: {_F_HINT_MAIN_PX}px; font-weight: 600;"
)
# 「取消」按钮**刻意不给样式表**（作者定）：全应用的普通按钮都不带 QSS，一给
# 就被 QStyleSheetStyle 接管、退出平台绘制 —— 实测同一枚原生按钮在 macOS /
# Windows / Fusion 下是 78x33 / 100x30 / 80x27 且跟着 palette 变深浅，而这枚
# 带 QSS 的「取消」三档一律 106x31、深浅两态像素不变，于是和顶栏按钮成了两套。
# 上面那 12 枚网格方块不在此列：三态色是这扇对话框唯一的信息通道（浅灰=未选 /
# 淡蓝=当前 / 实心蓝=将要切换，且"当前"与"将要"要同框），原生布尔态表达不了。


class CellButton(QPushButton):
    """单个网格单元格按钮

    仅负责悬停/点击事件上报；网格离开检测由父级 GridContainerWidget 统一处理。
    视觉由 `set_visual` 驱动（三态配色，每格恒为独立圆角片），内部用签名缓存
    跳过无变化的重设。
    """

    cell_clicked = Signal(int, int)  # (row, col) 点击信号
    cell_hovered = Signal(int, int)  # (row, col) 悬停信号

    def __init__(self, row: int, col: int, parent=None):
        # 必须先调用父类构造，否则下方的 setFixedSize 会抛异常
        super().__init__(parent)
        self.row = row
        self.col = col
        # 渲染签名（最近一次应用的 state）；None 表示尚未应用过样式
        self._sig: str | None = None
        self.setFixedSize(_CELL_SIZE, _CELL_SIZE)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        # 焦点由 grid_widget 统一持有，单元格不参与 Tab 链与方向键焦点
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAccessibleName(f"{row + 1}行{col + 1}列")

    @property
    def visual(self):
        """当前视觉签名（最近一次应用的 state），供测试与调试读取"""
        return self._sig

    def set_visual(self, state: str):
        """更新单元格视觉：三态配色 + 固定圆角

        Args:
            state: _S_OFF / _S_CURRENT / _S_SEL 之一

        签名与上次相同时直接返回：鼠标每划过一格都会触发整网格刷新，
        而绝大多数单元格的视觉并未改变，跳过可省掉大量样式串解析。
        """
        if state == self._sig:
            return
        self._sig = state
        bg_top, bg_bot, bd = _STATE_PALETTE[state]
        if bg_top == bg_bot:
            bg = bg_top
        else:
            bg = (
                f"qlineargradient(x1:0, y1:0, x2:0, y2:1, "
                f"stop:0 {bg_top}, stop:1 {bg_bot})"
            )
        # 属性先于样式串设置：setStyleSheet 会触发样式重解析，
        # 此时 cellState 属性已就位，属性选择器才能正确命中
        self.setProperty("cellState", state)
        self.setStyleSheet(
            f'QPushButton[cellState="{state}"] {{'
            f"background-color: {bg};"
            f"border: 1px solid {bd};"
            f"border-radius: {_R_CELL}px;"
            f"}}"
        )

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

    注：行列标尺也在本容器内，鼠标从单元格移到标尺数字上时不会触发回退，
    预览停在上一格——这是有意行为，避免鼠标掠过标尺时预览闪烁。
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

    三态视觉（消除"当前布局"与"悬停预览"混淆的关键）:
        淡蓝 = 当前布局；实心蓝 = 将要切换成的新布局预览；浅灰 = 未选。
        悬停矩形之外的当前布局单元格仍显示为淡蓝，新旧范围同框可比。

    初始化行为:
        打开对话框时立即把 (1,1)~(cur_rows, cur_cols) 预高亮为"当前布局"态，
        提示区显示 "{cur_rows} × {cur_cols}" 与 "当前布局" 副文案，
        使其与当前实际布局状态保持一致，便于用户参照调整。
        鼠标离开网格时回退到此初始状态，而非全部恢复灰色。

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
        self.setObjectName("layoutGridSelector")
        self.setStyleSheet(
            f"QDialog#layoutGridSelector {{ background-color: {_BG}; }}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        # 网格容器（第 0 行/列为行列标尺）
        self.grid_widget = GridContainerWidget(self)
        grid_layout = QGridLayout(self.grid_widget)
        grid_layout.setSpacing(_CELL_SPACING)
        grid_layout.setContentsMargins(0, 0, 0, 0)

        for c in range(self.max_cols):
            grid_layout.addWidget(self._make_ruler(str(c + 1)), 0, c + 1)

        for r in range(self.max_rows):
            grid_layout.addWidget(self._make_ruler(str(r + 1)), r + 1, 0)
            row_cells: list[CellButton] = []
            for c in range(self.max_cols):
                cell = CellButton(r, c, self.grid_widget)
                cell.cell_hovered.connect(self._on_cell_hovered)
                cell.cell_clicked.connect(self._on_cell_clicked)
                grid_layout.addWidget(cell, r + 1, c + 1)
                row_cells.append(cell)
            self.cells.append(row_cells)

        # 显式锁定网格行列尺寸：标尺带取固定厚度，单元格行/列恒为 _CELL_SIZE。
        # 不锁定时 cocoa 上未 polish 的 QGridLayout 会漏算末行（137x185 vs
        # 145x188），使整体尺寸算小、固定尺寸的方块纵向重叠
        grid_layout.setColumnMinimumWidth(0, _RULER_THICK)
        grid_layout.setRowMinimumHeight(0, _RULER_THICK)
        for i in range(self.max_cols):
            grid_layout.setColumnMinimumWidth(i + 1, _CELL_SIZE)
        for i in range(self.max_rows):
            grid_layout.setRowMinimumHeight(i + 1, _CELL_SIZE)

        layout.addWidget(self.grid_widget, 0, Qt.AlignmentFlag.AlignHCenter)

        # 提示区：左侧大号结果 + 副文案，右侧状态图例
        hint_row = QHBoxLayout()
        hint_row.setSpacing(16)

        hint_texts = QVBoxLayout()
        hint_texts.setSpacing(2)
        self.hint_label = QLabel()
        # 只设 styleHint/weight：字号由样式串接管（QStyleSheetStyle 的 sizeHint
        # 按样式解析后的字体计算，不会裁字），而 QSS 未指定的 family/styleHint
        # 沿用 widget font，因此等宽提示仍生效
        main_font = QFont()
        main_font.setStyleHint(QFont.StyleHint.TypeWriter)
        main_font.setWeight(QFont.Weight.DemiBold)
        self.hint_label.setFont(main_font)
        self.hint_label.setStyleSheet(_STYLE_HINT_MAIN)
        hint_texts.addWidget(self.hint_label)
        self.hint_sub_label = QLabel()
        self.hint_sub_label.setStyleSheet(_STYLE_MUTED)
        hint_texts.addWidget(self.hint_sub_label)
        hint_row.addLayout(hint_texts)

        hint_row.addStretch(1)
        legend_col = QVBoxLayout()
        legend_col.setSpacing(4)
        legend_col.addWidget(
            self._make_legend_swatch(_CELL_CUR_BG, _CELL_CUR_BD, "当前布局")
        )
        legend_col.addWidget(
            self._make_legend_swatch(_CELL_SEL_BG_BOT, _CELL_SEL_BD, "新布局")
        )
        hint_row.addLayout(legend_col)

        layout.addLayout(hint_row)

        # 页脚：分隔线 + 操作说明 + 取消按钮
        separator = QFrame(self)
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setLineWidth(0)
        separator.setFixedHeight(1)
        separator.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        separator.setStyleSheet(f"background-color: {_SEP}; border: none;")
        layout.addWidget(separator)

        footer = QHBoxLayout()
        footer.setSpacing(12)
        self.static_hint = QLabel(
            "点击方块即可应用 · 方向键微调 · Enter 确认 · Esc 取消"
        )
        self.static_hint.setStyleSheet(_STYLE_MUTED)
        footer.addWidget(self.static_hint)
        footer.addStretch(1)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        footer.addWidget(self.cancel_btn)
        layout.addLayout(footer)

        # 鼠标离开网格 -> 回退到初始预高亮
        self.grid_widget.grid_left.connect(self._on_grid_left)

        # 由布局接管整体尺寸：避免拉伸导致网格变形（见 §4.3）。
        # 不用 setFixedSize(self.sizeHint())——构造期的 hint 在 polish 后会变
        # （实测 offscreen 少 2px、cocoa 多 8px，方向还不一致），锁小会把固定
        # 尺寸的方块压叠；SetFixedSize 在布局生效时取值，平台无关
        layout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)

    def _make_ruler(self, text: str) -> QLabel:
        """行/列标尺文字（1..N），帮用户数格子、给网格版式骨架"""
        label = QLabel(text)
        label.setStyleSheet(_STYLE_RULER)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return label

    def _make_legend_swatch(self, bg: str, bd: str, text: str) -> QWidget:
        """状态图例项：10x10 色块 + 说明文字"""
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)
        chip = QFrame(row)
        chip.setFixedSize(_SWATCH_SIZE, _SWATCH_SIZE)
        chip.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        chip.setStyleSheet(
            f"background-color: {bg}; border: 1px solid {bd}; "
            f"border-radius: 3px;"
        )
        row_layout.addWidget(chip)
        label = QLabel(text, row)
        label.setStyleSheet(_STYLE_MUTED)
        row_layout.addWidget(label)
        return row

    # === 初始状态 ===

    def _apply_initial_highlight(self):
        """打开对话框时把当前布局区域预高亮，并设置提示区初始文案"""
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
        """更新三态高亮：新布局预览 / 当前布局 / 未选

        判定规则（消除新旧布局混淆的核心）：
            sel_rect     = (0,0)~(hover_row, hover_col)   # 未悬停时为空
            current_rect = (0,0)~(cur_rows-1, cur_cols-1)
            在 sel_rect          -> sel      实心蓝
            否则在 current_rect  -> current  淡蓝
            否则                 -> off      浅灰

        Args:
            hover_row/hover_col: 0-based 索引。
                - 任一为 -1 时 sel_rect 为空，仅保留"当前布局"淡蓝区域。
                - 若 cur_rows/cur_cols 也为 -1（不应发生），则全部恢复未选灰色。
        """
        hovering = hover_row >= 0 and hover_col >= 0
        states: list[list[str]] = []
        for r in range(self.max_rows):
            row_states: list[str] = []
            for c in range(self.max_cols):
                if hovering and r <= hover_row and c <= hover_col:
                    row_states.append(_S_SEL)
                elif r < self._cur_rows and c < self._cur_cols:
                    row_states.append(_S_CURRENT)
                else:
                    row_states.append(_S_OFF)
            states.append(row_states)

        for r in range(self.max_rows):
            for c in range(self.max_cols):
                self.cells[r][c].set_visual(states[r][c])

    def _update_hint(self, row: int, col: int):
        """更新提示区文案。row/col 为 -1 时显示当前布局，否则显示预览布局"""
        cur = f"{self._cur_rows} × {self._cur_cols}"
        if row < 0 or col < 0:
            self.hint_label.setText(cur)
            self.hint_sub_label.setText("当前布局 · 点击方块即可切换")
        else:
            self.hint_label.setText(f"{row + 1} × {col + 1}")
            self.hint_sub_label.setText(f"行 × 列 · 当前 {cur}")

    # === 对外接口 ===

    def values(self):
        """返回 (row, col) 元组（兼容旧 LayoutInputDialog 接口）"""
        return self._result_row, self._result_col
