"""曲线快捷面板共享组件（编辑器强化轮 C3）。

色板 + 绘制方式 / 数据点 / 样式预设分段控件 +「更多颜色…」，原属
LegendTextBrowser（形态三），抽共享后 legend 与变量编辑器共用同一
面板与 apply 链路（入口 parity）。

红线（与原实现同口径）：进入先问存活；exec 嵌套事件循环内不改 scene，
选择只写 picked，exec 返回后统一 apply；菜单 deleteLater 级联释放；
exec 后 / 更多颜色对话框后复查曲线归属。
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPalette, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QColorDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)
from PySide6.QtGui import QAction

from src.core.config import DEFAULT_CURVE_COLORS, widget_alive
from src.core.data_types import (
    DRAW_MODE_AUTO,
    DRAW_MODE_STEP_POST,
    DRAW_MODE_STEP_PRE,
    LINE_PRESET_AUTO,
    LINE_PRESET_BOLD,
    LINE_PRESET_DASH,
    LINE_PRESET_DOT,
    SYMBOL_POLICY_AUTO,
    SYMBOL_POLICY_OFF,
    SYMBOL_POLICY_ON,
)
from src.core.logger import get_logger
from src.ui import theme

logger = get_logger("widget.curve_style_panel")


# 图例菜单/状态栏用的中文短标签（键为 data_types 常量）
DRAW_MODE_LABELS = {
    DRAW_MODE_AUTO: "直线",
    DRAW_MODE_STEP_POST: "阶梯(后)",
    DRAW_MODE_STEP_PRE: "阶梯(前)",
}
SYMBOL_POLICY_LABELS = {
    SYMBOL_POLICY_AUTO: "自动",
    SYMBOL_POLICY_ON: "始终显示",
    SYMBOL_POLICY_OFF: "始终隐藏",
}
LINE_PRESET_LABELS = {
    LINE_PRESET_AUTO: "默认",
    LINE_PRESET_BOLD: "粗实线",
    LINE_PRESET_DASH: "虚线",
    LINE_PRESET_DOT: "点线",
}


# 调色板网格（方案 A）：5 列 × 2 行色块，中文色名只用于 tooltip / 无障碍名
PALETTE_COLUMNS = 6
_SWATCH_SIZE = 26      # 色块边长（逻辑像素，Qt 按 DPR 缩放）
_SWATCH_RADIUS = 5     # 圆角半径
_SWATCH_HOVER_RING = QColor("#bdbdbd")
_SWATCH_SELECT_RING = QColor(theme.ACCENT)
_COLOR_LABELS = {
    "#1658DB": "钴蓝",
    "#425CE0": "靛蓝",  # 历史候选色：本地模板/配置可能存该 hex，保留供 tooltip 命名
    "#0072B2": "蓝色",  # 旧首色：历史模板/配置仍存该 hex，保留供 tooltip 命名
    "#D62728": "红色",
    "#009E73": "青绿",
    "#CC79A7": "粉紫",
    "#E69F00": "橙色",
    "#56B4E9": "天蓝",
    "#6A3D9A": "深紫",
    "#8C564B": "棕色",
    "#999933": "橄榄色",
    "#882255": "酒红",
    "#006D77": "深青",
    "#8B1A1A": "深红",
}


class _ColorSwatch(QWidget):
    """调色板网格中的单个色块。

    只负责视觉与命中：填充色、hover 环、选中双环（当前色）、中文
    tooltip。左键在 press 记录、release 时指针仍在色块内才回调——若在
    press 内就关闭菜单，同一次物理 release 会漏到下层图例/绘图区，
    可能误触发显隐或框选（评审实测项）。
    """

    def __init__(
        self,
        color_name: str,
        label: str,
        selected: bool,
        on_pick,
        parent=None,
    ):
        super().__init__(parent)
        self.color_name = color_name
        self._selected = bool(selected)
        self._on_pick = on_pick
        self._pressed = False
        self._hovered = False
        self.setFixedSize(_SWATCH_SIZE, _SWATCH_SIZE)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setToolTip(f"{label}（当前）" if selected else label)
        self.setAccessibleName(label)

    def is_selected(self) -> bool:
        return self._selected

    def enterEvent(self, event):
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self._pressed = False
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._pressed = True
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            hit = self._pressed and self.rect().contains(
                event.position().toPoint()
            )
            self._pressed = False
            if hit:
                event.accept()
                self._on_pick(self.color_name)
            else:
                # 按下后拖出色块再松开：不消费事件，交回 QMenu 处理
                # （点菜单外即关闭，保持原生菜单手感）
                event.ignore()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (
            Qt.Key.Key_Return,
            Qt.Key.Key_Enter,
            Qt.Key.Key_Space,
        ):
            event.accept()
            self._on_pick(self.color_name)
            return
        super().keyPressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(self.color_name))
        body = self.rect().adjusted(3, 3, -3, -3)
        painter.drawRoundedRect(body, _SWATCH_RADIUS, _SWATCH_RADIUS)

        active = self._selected or self.hasFocus()
        if active or self._hovered:
            painter.setPen(
                QPen(
                    _SWATCH_SELECT_RING if active else _SWATCH_HOVER_RING,
                    2 if active else 1,
                )
            )
            painter.setBrush(Qt.BrushStyle.NoBrush)
            ring = self.rect().adjusted(1, 1, -1, -1)
            painter.drawRoundedRect(ring, _SWATCH_RADIUS + 2, _SWATCH_RADIUS + 2)
        painter.end()


# ---------------------------------------------------------------------------
# 曲线快捷面板：分段控件（形态三）与自绘图例
# ---------------------------------------------------------------------------

# 分段段标签：按钮上用短词（省横向空间），tooltip 给全称（与右键菜单
# 旧文案一致，DRAW_MODE_LABELS / SYMBOL_POLICY_LABELS）
_SEG_DRAW_ITEMS = (
    (DRAW_MODE_AUTO, "直线", "直线"),
    (DRAW_MODE_STEP_POST, "阶梯后", "阶梯(后)"),
    (DRAW_MODE_STEP_PRE, "阶梯前", "阶梯(前)"),
)
_SEG_SYMBOL_ITEMS = (
    (SYMBOL_POLICY_AUTO, "自动", "自动"),
    (SYMBOL_POLICY_ON, "显示", "始终显示"),
    (SYMBOL_POLICY_OFF, "隐藏", "始终隐藏"),
)
_SEG_LINE_PRESET_ITEMS = (
    (LINE_PRESET_AUTO, "默认", "默认（自动粗细，实线）"),
    (LINE_PRESET_BOLD, "粗实", "粗实线（持久突出）"),
    (LINE_PRESET_DASH, "虚线", "虚线（参考/次要曲线）"),
    (LINE_PRESET_DOT, "点线", "点线（第三区分档）"),
)

# 面板标题行变量名最大宽度（超出 elide 截断，全文进 tooltip）
_PANEL_VAR_NAME_MAX_WIDTH = 200

# 分段控件按钮 QSS：中性描边 + 选中品牌蓝（与色板选中环同色系）
_SEG_QSS = f"""
    QPushButton {{
        background: transparent;
        border: 1px solid palette(mid);
        border-radius: 4px;
        padding: 2px 8px;
    }}
    QPushButton:hover {{
        background: palette(midlight);
    }}
    QPushButton:checked {{
        background: {theme.TINT_SOFT};
        border-color: {theme.ACCENT};
    }}
"""


def _icon_painter(pixmap: QPixmap) -> tuple[QPainter, QColor]:
    """在透明 pixmap 上初始化画笔（主题文字色、圆角端点），返回画笔与颜色。"""
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    color = QPalette().color(QPalette.ColorRole.WindowText)
    pen = QPen(color, 1.6)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    return painter, color


def _draw_mode_icon(mode: str) -> QPixmap:
    """绘制方式 18×18 线型示意图标：直线 / 阶梯后 / 阶梯前。

    阶梯图标以实心点标注采样点位置（后 = 跳变发生在采样点，点在拐角；
    前 = 跳变提前一个采样间隔，点在平台上），与 data_types 的
    step_mode_for 语义对应。
    """
    dpr = 1.0
    screen = QApplication.primaryScreen()
    if screen is not None:
        dpr = screen.devicePixelRatio()
    pixmap = QPixmap(round(18 * dpr), round(18 * dpr))
    pixmap.setDevicePixelRatio(dpr)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter, color = _icon_painter(pixmap)
    if mode == DRAW_MODE_STEP_POST:
        painter.drawPolyline(
            [QPointF(2.5, 14.0), QPointF(9.0, 14.0),
             QPointF(9.0, 4.0), QPointF(15.5, 4.0)]
        )
        dot = QPointF(9.0, 14.0)
    elif mode == DRAW_MODE_STEP_PRE:
        painter.drawPolyline(
            [QPointF(2.5, 14.0), QPointF(5.0, 14.0),
             QPointF(5.0, 4.0), QPointF(15.5, 4.0)]
        )
        dot = QPointF(9.0, 4.0)
    else:  # DRAW_MODE_AUTO：直线
        painter.drawPolyline([QPointF(2.5, 14.0), QPointF(15.5, 4.0)])
        dot = QPointF(2.5, 14.0)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(color)
    painter.drawEllipse(dot, 1.8, 1.8)
    painter.end()
    return pixmap


def _symbol_policy_icon(policy: str) -> QPixmap:
    """数据点 18×18 图标：自动（半填充）/ 始终显示（实心）/ 始终隐藏（虚线空心）。"""
    dpr = 1.0
    screen = QApplication.primaryScreen()
    if screen is not None:
        dpr = screen.devicePixelRatio()
    pixmap = QPixmap(round(18 * dpr), round(18 * dpr))
    pixmap.setDevicePixelRatio(dpr)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter, color = _icon_painter(pixmap)
    rect = QRectF(3.0, 3.0, 12.0, 12.0)
    if policy == SYMBOL_POLICY_ON:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(rect)
    elif policy == SYMBOL_POLICY_OFF:
        pen = QPen(color, 1.4, Qt.PenStyle.DashLine)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(rect)
    else:  # SYMBOL_POLICY_AUTO：半填充（左实右空）
        painter.drawEllipse(rect)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawPie(rect, 90 * 16, 180 * 16)
    painter.end()
    return pixmap


def _line_preset_icon(preset: str) -> QPixmap:
    """样式预设 18×18 线段示意图标：用真实 QPen 宽度/线型画，所见即所得。"""
    dpr = 1.0
    screen = QApplication.primaryScreen()
    if screen is not None:
        dpr = screen.devicePixelRatio()
    pixmap = QPixmap(round(18 * dpr), round(18 * dpr))
    pixmap.setDevicePixelRatio(dpr)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter, color = _icon_painter(pixmap)
    if preset == LINE_PRESET_BOLD:
        pen = QPen(color, 4.0)
    elif preset == LINE_PRESET_DASH:
        pen = QPen(color, 2.0, Qt.PenStyle.DashLine)
    elif preset == LINE_PRESET_DOT:
        pen = QPen(color, 2.0, Qt.PenStyle.DotLine)
    else:  # LINE_PRESET_AUTO：细实线（symbol 态口径）
        pen = QPen(color, 1.6)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    painter.drawLine(QPointF(2.5, 9.0), QPointF(15.5, 9.0))
    painter.end()
    return pixmap


class _SegmentedControl(QWidget):
    """快捷面板内的横排分段单选控件（图标 + 短标签）。

    每组独立实例成容器：QPushButton 的 autoExclusive 互斥范围限定在
    同一 parent 的按钮间，两组（绘制方式 / 数据点）共存不串扰。段
    点击只回调 ``on_pick(value)``（由调用方决定记录 picked 与关闭
    菜单），不在此处改曲线状态——所有应用集中在菜单关闭后统一执行
    （嵌套事件循环内不改 scene 红线）。
    """

    def __init__(self, items, current, on_pick, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        for value, label, full_label, icon in items:
            btn = QPushButton(self)
            btn.setIcon(QIcon(icon))
            btn.setText(label)
            btn.setToolTip(full_label)
            btn.setCheckable(True)
            btn.setAutoExclusive(True)
            btn.setChecked(value == current)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _checked=False, v=value: on_pick(v))
            layout.addWidget(btn)
        self.setStyleSheet(_SEG_QSS)


def build_style_panel_widget(
    pw,
    var_name: str,
    picked: dict[str, str],
    holder: list,
    draw_group_suffix: str = "",
    title_text: str | None = None,
) -> QWidget:
    """构建曲线快捷面板主体（左键色块菜单 / 右键「样式」子菜单共用）。

    结构（形态三）：变量名标题行（当前色条 + elide 截断）→ 预设
    色板网格 → 分隔线 → 绘制方式 / 数据点两组图标分段控件。所有
    选择只写入 ``picked`` 并同步请求关闭 ``holder[0]``（顶层菜单），
    不在嵌套事件循环内改 scene——应用集中在菜单关闭后统一执行
    （``_apply_curve_panel_picks``）。面板随宿主菜单 deleteLater
    级联释放。

    Args:
        var_name: 目标曲线（须已在 pw.curves 中）
        picked: 选择收集 dict（键 color / draw_mode / symbol_policy /
            line_preset，单选语义：任一选择即关菜单，最多一个键）
        holder: 长度 1 的 list，元素为需要关闭的顶层菜单
        draw_group_suffix: 绘制方式分组标签后缀（枚举通道自动阶梯
            时传「（枚举通道自动）」以保留可解释性标注）
    """
    ci = pw.curves[var_name]
    current = ci.color
    draw_mode = getattr(ci, "draw_mode", DRAW_MODE_AUTO)
    symbol_policy = getattr(ci, "symbol_policy", SYMBOL_POLICY_AUTO)
    line_preset = getattr(ci, "line_preset", LINE_PRESET_AUTO)

    def _pick(key: str, value: str) -> None:
        picked[key] = value
        if holder:
            # 选中即关菜单（release/click 内同步 close，exec 随后
            # 返回 None）；不用 singleShot，避免定时器仍持有即将
            # deleteLater 的菜单
            holder[0].close()

    host = QWidget()
    vbox = QVBoxLayout(host)
    vbox.setContentsMargins(8, 8, 8, 6)
    vbox.setSpacing(6)

    # 标题行：当前色小条 + 变量名（超长 elide，全文进 tooltip）
    name_row = QWidget(host)
    name_lay = QHBoxLayout(name_row)
    name_lay.setContentsMargins(0, 0, 0, 0)
    name_lay.setSpacing(6)
    color_bar = QLabel(name_row)
    color_bar.setFixedSize(8, 8)
    color_bar.setStyleSheet(
        f"background-color: {current}; border-radius: 4px;"
        " border: 1px solid rgba(0, 0, 0, 0.25);"
    )
    fm = name_row.fontMetrics()
    display_name = title_text if title_text is not None else var_name
    name_label = QLabel(
        fm.elidedText(
            display_name,
            Qt.TextElideMode.ElideRight,
            _PANEL_VAR_NAME_MAX_WIDTH,
        ),
        name_row,
    )
    name_label.setToolTip(display_name)
    name_lay.addWidget(color_bar)
    name_lay.addWidget(name_label)
    name_lay.addStretch(1)
    vbox.addWidget(name_row)

    # 预设色板（顺序沿用 pw.curve_colors，行优先；当前色双环标注）
    grid = QGridLayout()
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setSpacing(6)
    for index, color_name in enumerate(pw.curve_colors):
        swatch = _ColorSwatch(
            color_name,
            _COLOR_LABELS.get(color_name, color_name),
            color_name == current,
            lambda c: _pick("color", c),
        )
        grid.addWidget(swatch, index // PALETTE_COLUMNS, index % PALETTE_COLUMNS)
    vbox.addLayout(grid)

    sep = QFrame(host)
    sep.setFrameShape(QFrame.Shape.HLine)
    vbox.addWidget(sep)

    # 两组分段控件（图标 + 短标签；每段 tooltip 给全称）
    draw_items = [
        (value, label, full, _draw_mode_icon(value))
        for value, label, full in _SEG_DRAW_ITEMS
    ]
    symbol_items = [
        (value, label, full, _symbol_policy_icon(value))
        for value, label, full in _SEG_SYMBOL_ITEMS
    ]
    preset_items = [
        (value, label, full, _line_preset_icon(value))
        for value, label, full in _SEG_LINE_PRESET_ITEMS
    ]
    for title, items, cur, key in (
        (f"绘制方式{draw_group_suffix}", draw_items, draw_mode, "draw_mode"),
        ("数据点", symbol_items, symbol_policy, "symbol_policy"),
        ("样式", preset_items, line_preset, "line_preset"),
    ):
        if key == "line_preset":
            # 性能已知限制行内注（2026-11-21 拍板 C）：非实线 pen 走
            # drawPath 描边，大数据量滚轮显著慢于实线；标题行右侧空白
            # 容纳小注，零新增高度。出处 docs/help.md 已知限制。
            cap_row = QWidget(host)
            cap_lay = QHBoxLayout(cap_row)
            cap_lay.setContentsMargins(0, 0, 0, 0)
            cap_lay.setSpacing(6)
            cap = QLabel(title, cap_row)
            cap.setStyleSheet("color: palette(mid);")
            note = QLabel("(大数据量建议实线)", cap_row)
            if note.font().pointSizeF() > 0:
                note_font = note.font()
                note_font.setPointSizeF(note_font.pointSizeF() - 1)
                note.setFont(note_font)
            note.setStyleSheet("color: palette(mid);")
            cap_lay.addWidget(cap)
            cap_lay.addWidget(note)
            cap_lay.addStretch(1)
            vbox.addWidget(cap_row)
        else:
            cap = QLabel(title, host)
            cap.setStyleSheet("color: palette(mid);")
            vbox.addWidget(cap)
        seg = _SegmentedControl(
            items, cur, lambda v, k=key: _pick(k, v), parent=host
        )
        vbox.addWidget(seg)
    return host


def build_curve_panel_menu(
    pw, var_name: str, picked: dict[str, str], holder: list
) -> tuple[QMenu, QAction]:
    """构建曲线快捷面板菜单（面板主体 +「更多颜色…」）。

    Returns:
        (menu, act_more)：``act_more`` 供 ``exec`` 返回值分发。
    """
    menu = QMenu(pw)
    panel = build_style_panel_widget(pw, var_name, picked, holder)
    widget_act = QWidgetAction(menu)
    widget_act.setDefaultWidget(panel)
    menu.addAction(widget_act)
    menu.addSeparator()
    act_more = menu.addAction("更多颜色…")
    # 压掉 QMenu 默认项内边距，避免面板四周多出一圈空白；
    # 不设 border，保留 macOS 原生菜单边框/圆角/阴影
    menu.setStyleSheet("QMenu { padding: 2px; }")
    return menu, act_more


def apply_curve_panel_picks(pw, var_name: str, picked: dict[str, str]) -> None:
    """快捷面板关闭后统一执行选择（色 / 绘制方式 / 数据点 / 样式预设）。

    单选语义（任一选择即关菜单）：dict 最多一个键，用 if/elif 表达。
    复查归属（打开期间曲线可能被 reload/编辑器删除）+ 同值 no-op；
    应用走 MultiCurveManager 三个共享入口，成功后状态栏播报。
    """
    if var_name not in pw.curves:
        return
    ci = pw.curves[var_name]
    main_window = pw.window()
    show = getattr(main_window, "_show_status_message", None)
    mcm = pw._multi_curve_manager

    if "color" in picked:
        value = picked["color"]
        if ci.color != value and mcm.apply_curve_color(var_name, value):
            if callable(show):
                show(f"已修改 {var_name} 颜色")
    elif "draw_mode" in picked:
        value = picked["draw_mode"]
        if (
            getattr(ci, "draw_mode", DRAW_MODE_AUTO) != value
            and mcm.apply_curve_draw_mode(var_name, value)
        ):
            if callable(show):
                show(f"已切换 {var_name} 绘制方式：{DRAW_MODE_LABELS[value]}")
    elif "symbol_policy" in picked:
        value = picked["symbol_policy"]
        if (
            getattr(ci, "symbol_policy", SYMBOL_POLICY_AUTO) != value
            and mcm.apply_curve_symbol_policy(var_name, value)
        ):
            if callable(show):
                show(f"已切换 {var_name} 数据点显示：{SYMBOL_POLICY_LABELS[value]}")
    elif "line_preset" in picked:
        value = picked["line_preset"]
        if (
            getattr(ci, "line_preset", LINE_PRESET_AUTO) != value
            and mcm.apply_curve_line_preset(var_name, value)
        ):
            if callable(show):
                show(f"已切换 {var_name} 样式：{LINE_PRESET_LABELS[value]}")


def exec_curve_panel(pw, var_name: str, global_pos: QPoint) -> None:
    """色块左键：弹出曲线快捷面板（色板 + 绘制方式 + 数据点）+「更多颜色…」。

    图例快捷样式入口（形态三）：色 / 绘制方式 / 数据点同面板直达，
    应用统一走 MultiCurveManager.apply_curve_color /
    apply_curve_draw_mode / apply_curve_symbol_policy。选择即关
    菜单并记录 picked，exec 返回后统一执行（嵌套事件循环内不改
    scene 红线）。调用方（anchorClicked 分发）经 singleShot(0)
    进入，先问存活；菜单 exec 返回后立即复查一次（已删则不再弹
    对话框），对话框返回后再复查一次（打开期间曲线可能被
    reload/编辑器删除，§6 R3 口径），同值选择为 no-op。菜单用
    deleteLater 释放，避免 QMenu/内嵌面板挂在 pw 下累积。
    """
    if not widget_alive(pw):
        logger.debug("legend 快捷面板跳过：控件已销毁")
        return
    if var_name not in pw.curves:
        return

    picked: dict[str, str] = {}
    holder: list[QMenu] = []
    menu, act_more = build_curve_panel_menu(pw, var_name, picked, holder)
    holder.append(menu)
    chosen = menu.exec(global_pos)
    menu.deleteLater()

    if chosen is not act_more:
        if picked:
            apply_curve_panel_picks(pw, var_name, picked)
        return  # Esc / 点菜单外：取消

    # 「更多颜色…」：菜单关闭后弹 Qt 自绘对话框（不在嵌套事件循环
    # 内再嵌套）；打开前复查归属，已删则不再弹
    if var_name not in pw.curves:
        return
    current = pw.curves[var_name].color
    # macOS 原生面板不支持 Qt 自定义色，统一用 Qt 自绘对话框，
    # 并把默认色板预填到“自定义颜色”，方便直接点选。
    for index, hex_color in enumerate(DEFAULT_CURVE_COLORS):
        QColorDialog.setCustomColor(index, QColor(hex_color))
    color = QColorDialog.getColor(
        QColor(current),
        pw,
        "选择颜色",
        QColorDialog.ColorDialogOption.DontUseNativeDialog,
    )
    if not color.isValid():
        return
    color_name = color.name()

    # 对话框关闭后二次复查 + 同色 no-op
    if var_name not in pw.curves or pw.curves[var_name].color == color_name:
        return
    if pw._multi_curve_manager.apply_curve_color(var_name, color_name):
        main_window = pw.window()
        show = getattr(main_window, "_show_status_message", None)
        if callable(show):
            show(f"已修改 {var_name} 颜色")
