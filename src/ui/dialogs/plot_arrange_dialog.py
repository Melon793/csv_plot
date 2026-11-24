"""整理布局对话框 —— 像整理手机桌面一样重排子图（设计稿 v4）。

交互口径（D8，已拍板）：
- 拖拽落位 = 交换/安放，只动源/目标格；多选整块 = 块交换；
- 拖入等候区暂存（跨页中转站），踢出/回填按可见区优先序前向补洞；
- 等候区非空 ⇒「确定」禁用；
- 隐藏区（右列+底行 L 形）虚线可收纳；
- 点「确定」才把排列置换应用到真实页面，取消 = 天然撤销。

实现要点：icon 不走 QGridLayout，按格位坐标手动定位，移动用
QPropertyAnimation(pos)（FLIP 等价），拖动预览期从当前动画位置重启。
"""

from __future__ import annotations

import html
import math

from PySide6.QtCore import (
    QEasingCurve,
    QPoint,
    QPropertyAnimation,
    QRect,
    Qt,
)
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTabBar,
    QVBoxLayout,
    QWidget,
)

from src.core.logger import get_logger
from src.core.plot_arrange_model import (
    PageState,
    ArrangeState,
    backfill,
    can_commit,
    clone_state,
    insert_from_tray,
    kick_to_tray,
    move_within,
    ordered_ids,
    seat_visible,
)
from src.ui.widgets.plot_arrange_commit import (
    ArrangeSnapshot,
    build_state,
    snapshot_pages,
    validate_state,
)

logger = get_logger("ui.plot_arrange_dialog")

GRID_MARGIN = 12
GRID_GAP = 10
ICON_MIN_H = 84
ANIM_MS = 180
CHIP_W = 150
CHIP_H = 42
CHIP_GAP = 7
TRAY_PAD = 10
# 松手收拢：按 topLeft 距离定时；3px 内视为已落位（避免 2→3px 时 0→70ms 跳变）
SETTLE_SPEED_PX_PER_MS = 2.0
SETTLE_MIN_MS = 70
SETTLE_MAX_MS = 260
SETTLE_SNAP_PX = 3.0


def _settle_ms(dist: float) -> int:
    if dist <= SETTLE_SNAP_PX:
        return 0
    return int(
        min(SETTLE_MAX_MS, max(SETTLE_MIN_MS, dist / SETTLE_SPEED_PX_PER_MS))
    )

_ACCENT = "#1658DB"
_ACCENT_SOFT = "#E8EFFC"
_LINE = "#d9dfe8"
_MUTED = "#6b7686"
_HOLE = "#aab3c0"

_ICON_QSS = """
QFrame#PlotIcon {
    background: #ffffff;
    border: 1.5px solid #d9dfe8;
    border-radius: 7px;
}
QFrame#PlotIcon[selected="true"] {
    border: 1.5px solid #1658DB;
    background: #E8EFFC;
}
QLabel#IconBadge {
    color: #6b7686; background: #eef1f6; border-radius: 4px;
    padding: 0px 5px; font-size: 10px;
}
QLabel#IconName { font-size: 12px; color: #1c2430; }
QLabel#IconMore { font-size: 11px; color: #6b7686; }
"""

_CHIP_QSS = """
QFrame#TrayChip {
    background: #ffffff;
    border: 1.5px solid #d9dfe8;
    border-radius: 7px;
}
QFrame#TrayChip[selected="true"] {
    border: 1.5px solid #1658DB;
    background: #E8EFFC;
}
QLabel#ChipName { font-size: 12px; color: #1c2430; }
QLabel#ChipMore { font-size: 11px; color: #6b7686; }
"""


def _repolish(widget: QWidget) -> None:
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def _plot_label(names: list[str]) -> str:
    return names[0] if names else "空图"


class _PlotIcon(QFrame):
    """格里的一枚图 icon：只显示曲线名，拖动交给对话框统一处理。"""

    def __init__(self, token: str, names: list[str], seat: int, dialog: QWidget):
        super().__init__(dialog._board)
        self.token = token
        self.seat = seat
        self._dialog = dialog
        self._press: QPoint | None = None
        self._dragging = False
        self.setObjectName("PlotIcon")
        self.setStyleSheet(_ICON_QSS)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 5, 8, 5)
        layout.setSpacing(1)
        badge_row = QHBoxLayout()
        badge_row.setContentsMargins(0, 0, 0, 0)
        badge_row.addStretch(1)
        badge = QLabel(f"图{seat + 1}")
        badge.setObjectName("IconBadge")
        badge_row.addWidget(badge)
        self._badge = badge
        layout.addLayout(badge_row)
        for name in names[:3]:
            label = QLabel(f"● {name}")
            label.setObjectName("IconName")
            label.setToolTip(name)
            label.setTextFormat(Qt.TextFormat.PlainText)
            layout.addWidget(label)
        self._name_labels = names[:3]
        if len(names) > 3:
            more = QLabel(f"等 {len(names)} 条")
            more.setObjectName("IconMore")
            layout.addWidget(more)
        elif not names:
            empty = QLabel("空图")
            empty.setObjectName("IconMore")
            layout.addWidget(empty)
        layout.addStretch(1)

    def set_selected(self, flag: bool) -> None:
        if bool(self.property("selected")) == flag:
            return
        self.setProperty("selected", flag)
        _repolish(self)

    def set_seat(self, seat: int) -> None:
        """格位号变化时同步角标「图M」（重排后编号必须跟着格位走）。"""
        if self.seat == seat:
            return
        self.seat = seat
        self._badge.setText(f"图{seat + 1}")

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        self._press = event.globalPosition().toPoint()
        self._dragging = False
        event.accept()

    def mouseMoveEvent(self, event):
        if self._press is None:
            return
        pos = event.globalPosition().toPoint()
        if not self._dragging:
            if (pos - self._press).manhattanLength() < 6:
                return
            self._dragging = True
            self._dialog.begin_drag(self.token, pos, self, from_tray=False)
        self._dialog.drag_move(pos)
        event.accept()

    def mouseReleaseEvent(self, event):
        if self._dragging:
            self._dialog.end_drag(event.globalPosition().toPoint())
        else:
            self._dialog.click_token(self.token, event.modifiers())
        self._press = None
        self._dragging = False
        event.accept()


class _TrayChip(QFrame):
    """等候区里的一枚图暂存 chip。"""

    def __init__(self, token: str, names: list[str], dialog: QWidget):
        super().__init__(dialog._tray_panel)
        self.token = token
        self._dialog = dialog
        self._press: QPoint | None = None
        self._dragging = False
        self.setObjectName("TrayChip")
        self.setStyleSheet(_CHIP_QSS)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)
        name = QLabel(html.escape(_plot_label(names)))
        name.setObjectName("ChipName")
        layout.addWidget(name)
        if len(names) > 1:
            more = QLabel(f"+{len(names) - 1}")
            more.setObjectName("ChipMore")
            layout.addWidget(more)

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        self._press = event.globalPosition().toPoint()
        self._dragging = False
        event.accept()

    def mouseMoveEvent(self, event):
        if self._press is None:
            return
        pos = event.globalPosition().toPoint()
        if not self._dragging:
            if (pos - self._press).manhattanLength() < 6:
                return
            self._dragging = True
            self._dialog.begin_drag(self.token, pos, self, from_tray=True)
        self._dialog.drag_move(pos)
        event.accept()

    def mouseReleaseEvent(self, event):
        if self._dragging:
            self._dialog.end_drag(event.globalPosition().toPoint())
        self._press = None
        self._dragging = False
        event.accept()


class _IconBoard(QWidget):
    """一格一 icon 的棋盘：自绘格位（含虚线隐藏区），icon 手动定位+动画。"""

    def __init__(self, dialog: "PlotArrangeDialog"):
        super().__init__(dialog)
        self._dialog = dialog
        self.icons: dict[str, _PlotIcon] = {}
        self._anims: dict[str, QPropertyAnimation] = {}
        self._highlight: int | None = None
        self._highlight_reject = False
        self._rubber_start: QPoint | None = None
        self._rubber_rect: QRect | None = None
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(ICON_MIN_H * 2)

    # ---- 几何 ----
    def _display_bounds(self) -> tuple[int, int]:
        """当前展示行列：整版 or 仅可见区（勾选开关时）。"""
        page = self._page()
        if not self._dialog.show_hidden():
            return (
                min(page.vis_rows, page.rows),
                min(page.vis_cols, page.cols),
            )
        return page.rows, page.cols

    def _display_seats(self) -> list[tuple[int, int, int]]:
        """展示格位列表：(row, col, 满网格 seat 下标)。"""
        page = self._page()
        rows, cols = self._display_bounds()
        return [
            (row, col, row * page.cols + col)
            for row in range(rows)
            for col in range(cols)
        ]

    def _axis_metrics(self, sizes: list[float], gap: int,
                      available: int) -> tuple[list[int], list[int]]:
        """把真实相对尺寸缩放进可用区域；整数分配、末位吃余数。"""
        weights = [max(1.0, float(size)) for size in sizes] or [1.0]
        weight_sum = sum(weights)
        offsets: list[int] = []
        lengths: list[int] = []
        cursor = GRID_MARGIN
        used = 0
        for index, weight in enumerate(weights):
            if index == len(weights) - 1:
                length = max(1, available - used)
            else:
                length = max(1, int(available * weight / weight_sum))
            offsets.append(cursor)
            lengths.append(length)
            cursor += length + gap
            used += length
        return offsets, lengths

    def _grid_metrics(self) -> tuple[list[int], list[int], list[int], list[int]]:
        """格位尺寸：列宽统一等分（利于跨区拖拽），行高镜像页面真实比例。

        行高是页级属性（row_height_factors + 子图最小高度），照抄才有
        "位子对应"；隐藏行取可见行中位数，与可见格同尺寸。
        """
        page = self._page()
        rows, cols = self._display_bounds()
        heights = list(self._dialog.row_heights_for_current())
        vis_rows = max(1, min(page.vis_rows, page.rows))
        known = [h for h in heights[:vis_rows] if h > 0]
        median = sorted(known)[len(known) // 2] if known else 100.0
        heights = [
            heights[i] if i < len(heights) and heights[i] > 0 else median
            for i in range(rows)
        ]
        widths = [1.0] * max(1, cols)
        avail_w = max(
            1, self.width() - 2 * GRID_MARGIN - max(0, cols - 1) * GRID_GAP
        )
        avail_h = max(
            1, self.height() - 2 * GRID_MARGIN - max(0, rows - 1) * GRID_GAP
        )
        xs, ws = self._axis_metrics(widths[:cols], GRID_GAP, avail_w)
        ys, hs = self._axis_metrics(heights[:rows], GRID_GAP, avail_h)
        return xs, ws, ys, hs

    def seat_rect(self, index: int) -> QRect:
        """满网格 seat 下标 → 屏幕矩形；隐藏格在"仅可见区"模式下返回空。"""
        page = self._page()
        row, col = divmod(index, page.cols)
        rows, cols = self._display_bounds()
        if row >= rows or col >= cols:
            return QRect()
        xs, ws, ys, hs = self._grid_metrics()
        x = xs[col] if col < len(xs) else GRID_MARGIN
        y = ys[row] if row < len(ys) else GRID_MARGIN
        w = ws[col] if col < len(ws) else 1
        h = hs[row] if row < len(hs) else 1
        return QRect(x, y, w, h)

    def seat_at(self, pos: QPoint) -> int | None:
        """就近吸附：格间缝对半分给相邻格，只有超出吸附范围才返回 None。

        吸附范围 = 展示格包围盒外扩 GRID_GAP/2 + 1px；仅可见区模式下
        候选集只含可见格，不会吸到隐藏格。
        """
        xs, ws, ys, hs = self._grid_metrics()
        limit = GRID_GAP / 2 + 1
        best: int | None = None
        best_dist = float("inf")
        for row, col, seat in self._display_seats():
            dx = max(xs[col] - pos.x(), 0, pos.x() - (xs[col] + ws[col]))
            dy = max(ys[row] - pos.y(), 0, pos.y() - (ys[row] + hs[row]))
            if dx <= limit and dy <= limit:
                dist = dx * dx + dy * dy
                if dist < best_dist:
                    best_dist = dist
                    best = seat
        return best

    # ---- 渲染 ----
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        page = self._page()
        seat_font = QFont(self.font())
        seat_font.setPointSizeF(max(8.0, seat_font.pointSizeF() - 1))
        painter.setFont(seat_font)
        occupied_seats = {icon.seat for icon in self.icons.values()}
        for _row, _col, index in self._display_seats():
            rect = self.seat_rect(index)
            hidden = not seat_visible(page, index)
            if hidden:
                painter.setPen(QPen(QColor(_HOLE), 1.2, Qt.PenStyle.DashLine))
                painter.setBrush(QColor("#f7f8fa"))
            else:
                painter.setPen(QPen(QColor("#dde4ee"), 1.2))
                painter.setBrush(QColor("#fbfcfe"))
            painter.drawRoundedRect(rect.adjusted(0, 0, -1, -1), 7, 7)
            if page.cells[index] is None and index not in occupied_seats:
                painter.setPen(QColor("#b3bcc9"))
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "空位")
            if hidden:
                painter.setPen(QColor(_HOLE))
                painter.drawText(
                    rect.adjusted(6, 0, -6, -4),
                    Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom,
                    "隐",
                )
        if self._highlight is not None:
            rect = self.seat_rect(self._highlight)
            if not rect.isNull():
                rect = rect.adjusted(-1, -1, 0, 0)
                color = "#c0392b" if self._highlight_reject else _ACCENT
                painter.setPen(QPen(QColor(color), 2.2))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRoundedRect(rect, 8, 8)
        if self._rubber_rect is not None:
            painter.setPen(QPen(QColor(_ACCENT), 1))
            painter.setBrush(QColor(22, 88, 219, 26))
            painter.drawRect(self._rubber_rect)

    # ---- icon 生命周期 ----
    def _icon_rect(self, seat: int) -> QRect:
        """icon 的真实落位矩形（格位内缩 3px），落位/收拢同源。"""
        rect = self.seat_rect(seat)
        if rect.isNull():
            return QRect()
        return rect.adjusted(3, 3, -3, -3)

    def _icon_target(self, seat: int) -> QPoint:
        return self._icon_rect(seat).topLeft()

    def sync(self, page: PageState, selection: set[str], exclude: set[str] = frozenset(),
             animate: bool = True, spawn_snap: bool = False) -> None:
        self._page_override = page
        displayed = {seat for _row, _col, seat in self._display_seats()}
        needed = {
            token
            for seat, token in enumerate(page.cells)
            if token and seat in displayed
        }
        for token in list(self.icons):
            if token not in needed and token not in exclude:
                icon = self.icons.pop(token)
                self._drop_anim(token)
                icon.deleteLater()
        for seat in sorted(displayed):
            token = page.cells[seat] if seat < len(page.cells) else None
            if token is None or token in exclude:
                continue
            icon = self.icons.get(token)
            is_new = icon is None
            if is_new:
                names = self._dialog.token_names(token)
                icon = _PlotIcon(token, names, seat, self._dialog)
                self.icons[token] = icon
                icon.show()
            icon.set_seat(seat)
            icon.set_selected(token in selection)
            if spawn_snap and is_new:
                # 本轮新建的 widget 直接出生在目标位，不从 (0,0) 起飞；
                # 既有 widget 一律走正常动画，不能被顺手瞬移
                rect = self._icon_rect(seat)
                self._drop_anim(token)
                if rect.isNull():
                    icon.hide()
                else:
                    icon.resize(rect.size())
                    icon.move(rect.topLeft())
                continue
            self._place(icon, token, animate)
        self.update()

    def _page(self):  # type: ignore[override]
        override = getattr(self, "_page_override", None)
        return override if override is not None else self._dialog.current_page_state()

    def _place(self, icon: _PlotIcon, token: str, animate: bool) -> None:
        """落位：终点未变不动画；跨目标从当前 pos retarget；旧动画显式销毁。

        旧实现每次 move 都新建 QPropertyAnimation 且只 stop 不 deleteLater，
        一次拖拽会累积上百个 QObject（评审确认的泄漏），这里统一清掉。
        """
        rect = self._icon_rect(icon.seat)
        if rect.isNull():
            icon.hide()
            self._drop_anim(token)
            return
        icon.resize(rect.size())
        target = self._icon_target(icon.seat)
        anim = self._anims.get(token)
        if anim is not None and anim.endValue() == target:
            return
        if not animate or icon.pos() == target:
            self._drop_anim(token)
            icon.move(target)
            return
        self._drop_anim(token)
        self._run_anim(token, icon, icon.pos(), target, ANIM_MS)

    def settle_token(self, token: str, start_center: QPoint) -> int:
        """松手收拢：从松手点（容器坐标、已夹取）按距离收拢到最终格。

        起点与终点统一在 topLeft 空间比较，避免中心→topLeft 混算偏半格。
        返回动画时长（0 = 距离≈0 直接落位）。
        """
        icon = self.icons.get(token)
        if icon is None:
            return 0
        rect = self._icon_rect(icon.seat)
        if rect.isNull():
            return 0
        icon.resize(rect.size())
        target = self._icon_target(icon.seat)
        start_tl = QPoint(
            start_center.x() - rect.width() // 2,
            start_center.y() - rect.height() // 2,
        )
        duration = _settle_ms(math.hypot(target.x() - start_tl.x(),
                                         target.y() - start_tl.y()))
        self._drop_anim(token)
        if duration == 0:
            icon.move(target)
            return 0
        self._run_anim(token, icon, start_tl, target, duration)
        return duration

    def _run_anim(self, token: str, widget: QWidget, start: QPoint,
                  end: QPoint, duration: int) -> None:
        anim = QPropertyAnimation(widget, b"pos", self)
        anim.setDuration(duration)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.setStartValue(start)
        anim.setEndValue(end)
        anim.finished.connect(lambda t=token, a=anim: self._on_anim_finished(t, a))
        anim.start()
        self._anims[token] = anim

    def _on_anim_finished(self, token: str, anim: QPropertyAnimation) -> None:
        if self._anims.get(token) is not anim:
            return
        self._anims.pop(token, None)
        anim.deleteLater()

    def _drop_anim(self, token: str) -> None:
        anim = self._anims.pop(token, None)
        if anim is not None:
            anim.stop()
            anim.deleteLater()

    def set_highlight(self, seat: int | None, reject: bool = False) -> None:
        if self._highlight == seat and self._highlight_reject == reject:
            return
        self._highlight = seat
        self._highlight_reject = reject
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._dialog._last_preview_sig = None
        page = self._page()
        for token, icon in self.icons.items():
            # 几何一变即掉动画：继续播会把 icon 停在旧格位上
            self._drop_anim(token)
            seat = icon.seat
            if 0 <= seat < page.rows * page.cols:
                rect = self._icon_rect(seat)
                if rect.isNull():
                    icon.hide()
                    continue
                icon.show()
                icon.setGeometry(rect)
        self._dialog.replay_drag_preview()

    # ---- 框选 ----
    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        self._rubber_start = event.position().toPoint()
        self._rubber_rect = QRect(self._rubber_start, self._rubber_start)
        self.update()
        event.accept()

    def mouseMoveEvent(self, event):
        if self._rubber_start is None:
            return super().mouseMoveEvent(event)
        pos = event.position().toPoint()
        self._rubber_rect = QRect(self._rubber_start, pos).normalized()
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event):
        if self._rubber_start is None:
            return super().mouseReleaseEvent(event)
        rect = self._rubber_rect or QRect()
        self._rubber_start = None
        self._rubber_rect = None
        picked: set[str] = set()
        if rect.width() > 6 and rect.height() > 6:
            for token, icon in self.icons.items():
                if rect.contains(icon.geometry().center()):
                    picked.add(token)
        additive = bool(event.modifiers() & (Qt.KeyboardModifier.ControlModifier
                                             | Qt.KeyboardModifier.MetaModifier))
        self._dialog.rubber_select(picked, additive)
        self.update()
        event.accept()


class _TrayPanel(QWidget):
    """等候区：手动换行摆放 chip，支持整体布局动画。"""

    def __init__(self, dialog: "PlotArrangeDialog"):
        super().__init__(dialog)
        self._dialog = dialog
        self.chips: dict[str, _TrayChip] = {}
        self._anims: dict[str, QPropertyAnimation] = {}
        self._frozen = False
        self.setMinimumHeight(CHIP_H + 2 * TRAY_PAD + 18)

    def set_frozen(self, frozen: bool) -> None:
        """拖拽期冻结高度（预览改变 chip 数时不得反过来改写棋盘几何）。"""
        self._frozen = bool(frozen)
        if not self._frozen:
            self._arrange(True)

    def _drop_anim(self, token: str) -> None:
        anim = self._anims.pop(token, None)
        if anim is not None:
            anim.stop()
            anim.deleteLater()

    def _on_anim_finished(self, token: str, anim: QPropertyAnimation) -> None:
        if self._anims.get(token) is not anim:
            return
        self._anims.pop(token, None)
        anim.deleteLater()

    def _run_anim(self, token: str, widget: QWidget, start: QPoint,
                  end: QPoint, duration: int) -> None:
        anim = QPropertyAnimation(widget, b"pos", self)
        anim.setDuration(duration)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.setStartValue(start)
        anim.setEndValue(end)
        anim.finished.connect(lambda t=token, a=anim: self._on_anim_finished(t, a))
        anim.start()
        self._anims[token] = anim

    def _chip_target(self, index: int) -> QPoint:
        """第 index 个 chip 的槽位 topLeft（按当前宽度换行）。"""
        inner = max(1, self.width() - 2 * TRAY_PAD)
        cols = max(1, (inner + CHIP_GAP) // (CHIP_W + CHIP_GAP))
        row, col = divmod(index, cols)
        return QPoint(
            TRAY_PAD + col * (CHIP_W + CHIP_GAP),
            TRAY_PAD + row * (CHIP_H + CHIP_GAP),
        )

    def settle_token(self, token: str, start_center: QPoint) -> int:
        """松手收拢：chip 从松手点（容器坐标、已夹取）按距离滑入槽位。"""
        chip = self.chips.get(token)
        if chip is None:
            return 0
        target = self._chip_target(list(self.chips).index(token))
        chip.resize(CHIP_W, CHIP_H)
        start_tl = QPoint(
            start_center.x() - CHIP_W // 2,
            start_center.y() - CHIP_H // 2,
        )
        duration = _settle_ms(math.hypot(target.x() - start_tl.x(),
                                         target.y() - start_tl.y()))
        self._drop_anim(token)
        if duration == 0:
            chip.move(target)
            return 0
        self._run_anim(token, chip, start_tl, target, duration)
        return duration

    def sync(self, state: ArrangeState, exclude: set[str] = frozenset(),
             animate: bool = True, spawn_snap: bool = False) -> None:
        new_tokens: set[str] = set()
        for token in list(self.chips):
            if token not in state.tray and token not in exclude:
                chip = self.chips.pop(token)
                self._drop_anim(token)
                chip.deleteLater()
        for token in state.tray:
            if token in exclude:
                continue
            chip = self.chips.get(token)
            if chip is None:
                chip = _TrayChip(token, self._dialog.token_names(token), self._dialog)
                self.chips[token] = chip
                chip.show()
                new_tokens.add(token)
        self._arrange(
            animate, exclude, snap=new_tokens if spawn_snap else frozenset()
        )

    def _arrange(self, animate: bool, exclude: set[str] = frozenset(),
                 snap: set[str] = frozenset()) -> None:
        for index, token in enumerate(self.chips):
            if token in exclude:  # 拖拽源保持原位（与棋盘侧一致）
                continue
            chip = self.chips[token]
            target = self._chip_target(index)
            chip.resize(CHIP_W, CHIP_H)
            if token in snap:
                # 本轮新建的 chip 直接出生在槽位，不从 (0,0) 起飞
                self._drop_anim(token)
                chip.move(target)
                continue
            anim = self._anims.get(token)
            if anim is not None and anim.endValue() == target:
                continue
            if not animate or chip.pos() == target:
                self._drop_anim(token)
                chip.move(target)
                continue
            self._drop_anim(token)
            self._run_anim(token, chip, chip.pos(), target, ANIM_MS)
        if self._frozen:
            return
        rows = (
            max(1, math.ceil(len(self.chips) / self._chip_cols()))
            if self.chips else 1
        )
        wanted = TRAY_PAD * 2 + rows * CHIP_H + (rows - 1) * CHIP_GAP + 16
        if self.minimumHeight() != wanted:
            self.setMinimumHeight(wanted)
            self.updateGeometry()

    def _chip_cols(self) -> int:
        inner = max(1, self.width() - 2 * TRAY_PAD)
        return max(1, (inner + CHIP_GAP) // (CHIP_W + CHIP_GAP))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(_HOLE), 1.2, Qt.PenStyle.DashLine))
        painter.setBrush(QColor("#f7f9fc"))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 9, 9)


class PlotArrangeDialog(QDialog):
    """整理布局对话框。点「确定」后由调用方读取 state 并应用置换。"""

    def __init__(self, pages: list, *, current_page=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("整理子图")
        self.resize(980, 700)

        self._snapshot: ArrangeSnapshot = snapshot_pages(pages)
        self._state: ArrangeState = build_state(self._snapshot)
        self._initial: ArrangeState = clone_state(self._state)
        self._selection: set[str] = set()
        self._anchor: str | None = None
        self._drag: dict | None = None
        #: 上次生效的预览签名（P2 同目标跳过；None 表示需重算）
        self._last_preview_sig = None
        #: replay_drag_preview 重入保护（几何变化回调可能再次触碰布局）
        self._replaying = False
        self._page_index = 0
        if current_page is not None and current_page in pages:
            self._page_index = pages.index(current_page)

        self._build_ui()
        self._refresh(animate=False)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(8)

        self._tabs = QTabBar(self)
        self._tabs.setExpanding(False)
        self._tabs.setDrawBase(False)
        for index, page in enumerate(self._state.pages):
            title = f"{page.title}  {len(ordered_ids(page))}/{page.rows * page.cols}"
            self._tabs.addTab(title)
        self._tabs.setCurrentIndex(self._page_index)
        tab_row = QHBoxLayout()
        tab_row.setContentsMargins(0, 0, 0, 0)
        tab_row.addWidget(self._tabs)
        tab_row.addStretch(1)
        self._adjust_layout_btn = QPushButton("子图行列…", self)
        self._adjust_layout_btn.setToolTip(
            "设置本页可见行列；立即生效，不随「取消」回退"
        )
        self._adjust_layout_btn.clicked.connect(self._on_adjust_visible_layout)
        tab_row.addWidget(self._adjust_layout_btn)
        self._show_hidden_box = QCheckBox("显示隐藏区", self)
        self._show_hidden_box.setToolTip(
            "勾选后在网格中展示隐藏行列，可拖入收纳；隐藏区里的图始终保留。"
            "「已建=可见」的页没有隐藏区，勾选对它无变化"
        )
        # 先按存档置位再连信号：置位发生在 board/tray 创建之前，不能触发刷新
        from src.core.settings import AppSettings

        self._show_hidden_box.setChecked(AppSettings().get_arrange_show_hidden())
        self._show_hidden_box.toggled.connect(self._on_show_hidden_toggled)
        tab_row.addWidget(self._show_hidden_box)
        root.addLayout(tab_row)
        # 连接必须放在 setCurrentIndex 之后：首个渲染时 board/tray 尚未创建
        self._tabs.currentChanged.connect(self._on_tab_changed)

        self._caption = QLabel(self)
        self._caption.setStyleSheet(f"color: {_MUTED}; font-size: 12px;")
        self._caption.setWordWrap(True)
        root.addWidget(self._caption)

        self._board = _IconBoard(self)
        root.addWidget(self._board, 1)

        self._tray_panel = _TrayPanel(self)
        root.addWidget(self._tray_panel)

        self._status = QLabel(self)
        self._status.setStyleSheet(f"color: {_MUTED}; font-size: 12px;")
        root.addWidget(self._status)

        buttons = QHBoxLayout()
        self._btn_kick = QPushButton("移入等候区 (Del)", self)
        self._btn_kick.clicked.connect(self._on_kick)
        self._btn_backfill = QPushButton("一键回填到本页", self)
        self._btn_backfill.clicked.connect(self._on_backfill)
        self._btn_reset = QPushButton("重置", self)
        self._btn_reset.clicked.connect(self._on_reset)
        buttons.addWidget(self._btn_kick)
        buttons.addWidget(self._btn_backfill)
        buttons.addWidget(self._btn_reset)
        buttons.addStretch(1)
        self._btn_cancel = QPushButton("取消", self)
        self._btn_cancel.clicked.connect(self._on_cancel_clicked)
        self._btn_ok = QPushButton("确定", self)
        self._btn_ok.setDefault(True)
        self._btn_ok.clicked.connect(self._on_accept)
        buttons.addWidget(self._btn_cancel)
        buttons.addWidget(self._btn_ok)
        root.addLayout(buttons)

    # ------------------------------------------------------------------
    # 状态访问
    # ------------------------------------------------------------------
    @property
    def snapshot(self) -> ArrangeSnapshot:
        return self._snapshot

    def result_state(self) -> ArrangeState:
        return self._state

    def current_page_state(self) -> PageState:
        return self._state.pages[self._page_index]

    def token_names(self, token: str) -> list[str]:
        return list(self._snapshot.names.get(token, []))

    def row_pcts_for_current(self) -> list[int]:
        """当前页各行行高百分比（镜像自页面，仅展示用，重排不迁移）。"""
        try:
            return list(self._snapshot.row_pcts[self._page_index])
        except IndexError:
            return []

    def row_heights_for_current(self) -> list[float]:
        """当前页各行高的真实相对尺寸（镜像页面布局）。"""
        try:
            return list(self._snapshot.row_heights[self._page_index])
        except IndexError:
            return []

    def _on_adjust_visible_layout(self) -> None:
        """复用「子图行列…」入口调整本页可见行列，并重映射对话框格位。

        扩可见区会补建新格（矩阵只增不减），行优先下标因此漂移；token id
        与格位解耦，这里按真实 (row, col) 重铺 cells，不丢托盘与选择。
        """
        if self._drag is not None:
            return
        snapshot_page = self._snapshot.pages[self._page_index]
        opener = getattr(
            getattr(self.parent(), "layout_manager", None),
            "open_layout_dialog",
            None,
        )
        if opener is None:
            self._status.setText("当前环境不支持调整子图行列")
            return
        before = (
            self._state.pages[self._page_index].rows,
            self._state.pages[self._page_index].cols,
            self._state.pages[self._page_index].vis_rows,
            self._state.pages[self._page_index].vis_cols,
        )
        opener(page=snapshot_page, parent=self)
        self._resync_page_after_visibility(self._page_index)
        after_page = self._state.pages[self._page_index]
        if (
            before
            == (after_page.rows, after_page.cols,
                after_page.vis_rows, after_page.vis_cols)
        ):
            return
        self._status.setText(
            f"可见布局已更新：可见 {after_page.vis_rows}×{after_page.vis_cols}"
            f" · 已建 {after_page.rows}×{after_page.cols}"
        )

    def _resync_page_after_visibility(self, page_index: int) -> None:
        """按真实矩阵重测该页：格位重映射、快照 dims/行高/可见区同步。"""
        from src.ui.widgets.plot_arrange_commit import page_layout_metrics

        real_page = self._snapshot.pages[page_index]
        metrics = page_layout_metrics(real_page)
        if metrics is None:
            return
        state_page = self._state.pages[page_index]
        old_cols = state_page.cols
        positions: dict[str, tuple[int, int]] = {}
        for seat, token in enumerate(state_page.cells):
            if token:
                positions[token] = divmod(seat, old_cols)
        rows, cols = metrics["rows"], metrics["cols"]
        cells: list[str | None] = [None] * (rows * cols)
        for token, (row, col) in positions.items():
            if row < rows and col < cols:
                cells[row * cols + col] = token
        state_page.rows = rows
        state_page.cols = cols
        state_page.cells = cells
        state_page.vis_rows = metrics["vis_rows"]
        state_page.vis_cols = metrics["vis_cols"]
        self._snapshot.dims[page_index] = (rows, cols)
        self._snapshot.built[page_index] = True
        self._snapshot.row_pcts[page_index] = metrics["pcts"]
        self._snapshot.row_heights[page_index] = metrics["row_heights"]
        for token, (row, col) in positions.items():
            self._snapshot.seat_of_token[token] = (row, col)
        self._board._page_override = None
        self._refresh(animate=False)

    def show_hidden(self) -> bool:
        box = getattr(self, "_show_hidden_box", None)
        return bool(box is not None and box.isChecked())

    @staticmethod
    def page_has_hidden(page) -> bool:
        """该页是否存在隐藏区（已建矩阵大于当前可见矩形）。"""
        return page.rows > page.vis_rows or page.cols > page.vis_cols

    def _on_show_hidden_toggled(self, checked: bool) -> None:
        if self._drag is not None:
            # 拖拽期不切展示区：回滚复选框，不写 setting、不刷新
            box = self._show_hidden_box
            box.blockSignals(True)
            box.setChecked(not checked)
            box.blockSignals(False)
            return
        try:
            from src.core.settings import AppSettings

            AppSettings().set_arrange_show_hidden(bool(checked))
        except Exception:
            logger.debug("[ARRANGE] 记忆「显示隐藏区」失败", exc_info=True)
        self._board._page_override = None
        self._board.set_highlight(None)
        self._last_preview_sig = None
        self._refresh(animate=False)

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------
    def display_bounds(self) -> tuple[int, int]:
        """当前展示区行列：显示隐藏区开=整网格，关=可见矩形。"""
        return self._board._display_bounds()

    def _refresh(self, animate: bool = True, spawn_snap: bool = False) -> None:
        self._last_preview_sig = None
        page = self.current_page_state()
        self._board.sync(
            page, self._selection, animate=animate, spawn_snap=spawn_snap
        )
        self._tray_panel.sync(
            self._state, animate=animate, spawn_snap=spawn_snap
        )
        self._refresh_buttons()
        self._refresh_tab_texts()
        self._refresh_caption()

    def _refresh_caption(self) -> None:
        page = self.current_page_state()
        pcts = self._snapshot.row_pcts[self._page_index]
        shown = "/".join(f"{pct}%" for pct in pcts)
        if len(pcts) > 8:
            shown = "/".join(f"{pct}%" for pct in pcts[:8]) + "…"
        if not self.page_has_hidden(page):
            mode = " · 本页无隐藏区（已建=可见）"
        elif self.show_hidden():
            mode = " · 显示隐藏区（可拖入收纳）"
        else:
            mode = " · 隐藏区已收起（勾选「显示隐藏区」可收纳）"
        self._caption.setText(
            f"「{page.title}」可见 {page.vis_rows}×{page.vis_cols}"
            f" · 已建 {page.rows}×{page.cols}"
            f" · 有图 {len(ordered_ids(page))} 个"
            f" · 行高 {shown}（跟行走，不随图迁移）{mode}"
        )

    def _refresh_tab_texts(self) -> None:
        for index, page in enumerate(self._state.pages):
            self._tabs.setTabText(
                index,
                f"{page.title}  {len(ordered_ids(page))}/{page.rows * page.cols}",
            )

    def _structure_ok(self) -> bool:
        """结构校验：cells ∪ tray 必须与快照 token 集一一对应（防算法 bug 静默丢图）。"""
        tokens: list[str] = []
        for page in self._state.pages:
            tokens.extend(token for token in page.cells if token)
        tokens.extend(self._state.tray)
        return sorted(tokens) == sorted(self._snapshot.tokens.keys())

    def _refresh_buttons(self) -> None:
        self._btn_kick.setEnabled(bool(self._selection))
        structure_ok = self._structure_ok()
        ok = can_commit(self._state) and structure_ok
        self._btn_ok.setEnabled(ok)
        if not structure_ok:
            self._btn_ok.setToolTip("内部状态异常")
            self._status.setText("内部状态异常（图数不一致），请先「重置」")
        elif ok:
            self._btn_ok.setToolTip("")
            hidden_hint = (
                "隐藏区可收纳"
                if self.show_hidden()
                else "勾选「显示隐藏区」可收纳"
            )
            self._status.setText(
                f"拖动=交换/安放 · 拖入等候区暂存 · {hidden_hint} · 点「确定」应用"
            )
        else:
            self._btn_ok.setToolTip("等候区还有图未安置")
            self._status.setText(
                f"等候区还有 {len(self._state.tray)} 个图未安置 —— 全部拖回格子后才能确定"
            )
        page = self.current_page_state()
        self._btn_backfill.setEnabled(
            bool(self._state.tray) and len(ordered_ids(page)) < page.rows * page.cols
        )

    def _on_tab_changed(self, index: int) -> None:
        if not (0 <= index < len(self._state.pages)):
            return
        # 必须在改 _page_index 之前取消：预览/收拢是按旧页格位算的
        self._cancel_drag()
        self._page_index = index
        self._selection.clear()
        self._anchor = None
        self._board._page_override = None
        self._board.set_highlight(None)
        self._refresh(animate=False)

    # ------------------------------------------------------------------
    # 选择
    # ------------------------------------------------------------------
    def click_token(self, token: str, modifiers) -> None:
        order = ordered_ids(self.current_page_state())
        if modifiers & Qt.KeyboardModifier.ShiftModifier and self._anchor in order:
            start = order.index(self._anchor)
            end = order.index(token)
            low, high = (start, end) if start < end else (end, start)
            if not (modifiers & (Qt.KeyboardModifier.ControlModifier
                                 | Qt.KeyboardModifier.MetaModifier)):
                self._selection.clear()
            self._selection.update(order[low : high + 1])
        elif modifiers & (Qt.KeyboardModifier.ControlModifier
                          | Qt.KeyboardModifier.MetaModifier):
            if token in self._selection:
                self._selection.discard(token)
            else:
                self._selection.add(token)
            self._anchor = token
        else:
            self._selection = {token}
            self._anchor = token
        self._refresh(animate=True)

    def rubber_select(self, tokens: set[str], additive: bool) -> None:
        if not additive:
            self._selection.clear()
        self._selection.update(tokens)
        self._refresh_buttons()
        self._board.sync(self.current_page_state(), self._selection, animate=False)

    def clear_selection(self) -> None:
        if not self._selection:
            return
        self._selection.clear()
        self._anchor = None
        self._refresh(animate=True)

    # ------------------------------------------------------------------
    # 拖拽
    # ------------------------------------------------------------------
    def begin_drag(self, token: str, global_pos: QPoint, widget: QWidget,
                   *, from_tray: bool) -> None:
        if from_tray:
            ids = [token]
        else:
            if token not in self._selection:
                self._selection = {token}
                self._anchor = token
                self._board.sync(
                    self.current_page_state(), self._selection, animate=False
                )
            ids = [t for t in ordered_ids(self.current_page_state())
                   if t in self._selection]
        top_left = widget.mapToGlobal(QPoint(0, 0))
        # 先掉该 token 的旧动画，避免暗化 icon 在浮影背后继续滑
        if from_tray:
            self._tray_panel._drop_anim(token)
        else:
            self._board._drop_anim(token)
        # 原位 icon 变暗留在格子里（拖拽源提示），跟手的是独立浮影；
        # 不搬本体可避免"原图卡在标题/边界上"的观感问题
        dim = QGraphicsOpacityEffect(widget)
        dim.setOpacity(0.35)
        widget.setGraphicsEffect(dim)
        ghost = self._make_drag_ghost(token, ids)
        ghost.show()
        ghost.raise_()
        self._last_preview_sig = None
        self._tray_panel.set_frozen(True)  # 拖拽期冻结托盘高度，避免几何反馈环
        self._drag = {
            "ids": ids,
            "anchor": token,
            "bounds": self.display_bounds(),
            "from": "tray" if from_tray else "grid",
            "widget": widget,
            "ghost": ghost,
            "offset": global_pos - top_left,
            "target": None,
        }
        self._move_drag_widget(global_pos)

    def _make_drag_ghost(self, token: str, ids: list[str]) -> QFrame:
        """拖拽浮影：紧凑卡片（名字+数量），阴影提示"浮起"。"""
        ghost = QFrame(self)
        ghost.setObjectName("TrayChip")
        ghost.setStyleSheet(_CHIP_QSS)
        ghost.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout = QHBoxLayout(ghost)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(5)
        names = self.token_names(token)
        name = QLabel(html.escape(_plot_label(names)))
        name.setObjectName("ChipName")
        layout.addWidget(name)
        if len(ids) > 1:
            extra = QLabel(f"整块 {len(ids)} 个")
            extra.setObjectName("ChipMore")
            layout.addWidget(extra)
        elif len(names) > 1:
            extra = QLabel(f"+{len(names) - 1}")
            extra.setObjectName("ChipMore")
            layout.addWidget(extra)
        ghost.adjustSize()
        effect = QGraphicsDropShadowEffect(ghost)
        effect.setBlurRadius(18.0)
        effect.setOffset(0.0, 4.0)
        effect.setColor(QColor(20, 30, 50, 110))
        ghost.setGraphicsEffect(effect)
        return ghost

    def _move_drag_widget(self, global_pos: QPoint) -> None:
        if self._drag is None:
            return
        ghost = self._drag.get("ghost")
        if ghost is not None:
            # 缩小后的浮影以光标为中心，避免宽卡"横在"目标格上
            center = QPoint(ghost.width() // 2, ghost.height() // 2)
            ghost.move(self.mapFromGlobal(global_pos - center))

    def _tray_gap(self) -> int:
        """棋盘与托盘之间的布局间距（px）；未布局时退回 GRID_GAP。"""
        try:
            board_bottom = self._board.mapTo(self, QPoint(0, self._board.height())).y()
            tray_top = self._tray_panel.mapTo(self, QPoint(0, 0)).y()
        except RuntimeError:
            return GRID_GAP
        return max(0, tray_top - board_bottom) + 1

    def _hit(self, global_pos: QPoint):
        board_pos = self._board.mapFromGlobal(global_pos)
        if self._board.rect().contains(board_pos):
            seat = self._board.seat_at(board_pos)
            if seat is not None:
                return ("seat", seat)
        tray_pos = self._tray_panel.mapFromGlobal(global_pos)
        # 两控件之间的间距两者都不包含：向上扩展给托盘，避免缝里判定抖动
        if self._tray_panel.rect().adjusted(
            0, -self._tray_gap() - GRID_MARGIN, 0, 0
        ).contains(tray_pos):
            return ("tray", None)
        return None

    def _preview(self, target) -> tuple[ArrangeState | None, bool, dict]:
        """按落点算假设状态（不动真实状态）。"""
        if self._drag is None:
            return None, False, {}
        ids = self._drag["ids"]
        if target is None:
            return None, False, {}
        state = clone_state(self._state)
        if target[0] == "seat":
            if self._drag["from"] == "grid":
                result = move_within(
                    state, self._page_index, ids, target[1],
                    anchor=self._drag.get("anchor"),
                    bounds=self._drag.get("bounds"),
                )
                return state, result["ok"], {
                    "swapped": result.get("swapped", []),
                    "anchor_seat": result.get("anchor_seat"),
                    "mode": result.get("mode"),
                    "clamped": bool(result.get("clamped")),
                    "moved": bool(result.get("moved")),
                }
            result = insert_from_tray(state, self._page_index, ids, target[1])
            return state, result["ok"], {"displaced": result.get("displaced", [])}
        if target[0] == "tray" and self._drag["from"] == "grid":
            kick_to_tray(state, self._page_index, ids)
            return state, True, {"kicked": len(ids)}
        return None, False, {}

    @staticmethod
    def _anchor_final_seat(page: PageState, anchor: str | None) -> int | None:
        if not anchor:
            return None
        try:
            return page.cells.index(anchor)
        except ValueError:
            return None

    def _preview_signature(self, target, preview, ok: bool):
        """预览视觉指纹：排列 + 托盘顺序 + 模式；高亮/状态栏不参与。"""
        if self._drag is None or target is None:
            return ("none",)
        if preview is None or not ok:
            return ("reject", target)
        page = preview.pages[self._page_index]
        return (
            self._page_index,
            self.show_hidden(),
            tuple(self._drag["ids"]),
            target,
            tuple(page.cells),
            tuple(preview.tray),
        )

    def drag_move(self, global_pos: QPoint) -> None:
        if self._drag is None:
            return
        self._move_drag_widget(global_pos)
        self._drag["last_pos"] = global_pos
        target = self._hit(global_pos)
        self._drag["target"] = target
        preview, ok, info = self._preview(target)
        exclude = set(self._drag["ids"])
        anchor_seat = None
        if preview is not None and ok and target and target[0] == "seat":
            # 高亮永远跟 anchor 的最终格；取不到时在 preview 里定位，仍取不到就不高亮
            anchor_seat = info.get("anchor_seat")
            if anchor_seat is None:
                anchor_seat = self._anchor_final_seat(
                    preview.pages[self._page_index], self._drag.get("anchor")
                )
            highlight_seat = anchor_seat
        elif target and target[0] == "seat":
            highlight_seat = target[1]
        else:
            highlight_seat = None
        reject = bool(target) and not ok
        signature = self._preview_signature(target, preview, ok)
        if signature == self._last_preview_sig:
            # 视觉不变：只同步高亮，不重排、不重启动画（防抽动核心）
            self._board.set_highlight(highlight_seat, reject=reject)
            return
        self._last_preview_sig = signature
        if preview is not None and ok:
            self._board.sync(
                preview.pages[self._page_index], self._selection,
                exclude=exclude, animate=True, spawn_snap=True,
            )
            self._tray_panel.sync(
                preview, exclude=exclude, animate=True, spawn_snap=True
            )
            if target[0] == "seat":
                self._board.set_highlight(anchor_seat, reject=False)
            else:
                self._board.set_highlight(None)
            text = self._preview_status_text(target, info, len(self._drag["ids"]))
            if text:
                self._status.setText(text)
        else:
            self._board.sync(
                self.current_page_state(), self._selection,
                exclude=exclude, animate=True,
            )
            self._tray_panel.sync(self._state, exclude=exclude, animate=True)
            self._board.set_highlight(highlight_seat, reject=reject)

    def _preview_status_text(self, target, info: dict, count: int) -> str | None:
        """拖动预览期状态文案（松手后会被通用提示覆盖）。"""
        if count > 1 and target and target[0] == "seat":
            if info.get("mode") == "rigid":
                text = "整块平移：保持相对位置"
                if info.get("clamped"):
                    text += (
                        "（已到边界，无法继续移动）"
                        if not info.get("moved") else "（已贴边）"
                    )
            else:
                text = "整块重排：按阅读序落入连续格位"
            swapped = info.get("swapped") or []
            if swapped:
                text += f"；目标格原图移去源格（{len(swapped)} 个）"
            return text
        if info.get("swapped"):
            return (
                f"交换：目标格原图将移去源格（{len(info['swapped'])} 个），其余图不动"
            )
        if info.get("displaced"):
            return f"顶替：目标格原图将移入等候区（{len(info['displaced'])} 个）"
        if target and target[0] == "seat":
            return (
                "安放：源格将变为空位，其余图不动"
                if self._drag["from"] == "grid" else "安放到空位"
            )
        if info.get("kicked"):
            return f"移入等候区（{info['kicked']} 个）"
        return None

    def replay_drag_preview(self) -> None:
        """棋盘几何变化后（拖拽中缩放窗口）按最后指针位置重放预览。"""
        if self._drag is None or self._replaying:
            return
        last = self._drag.get("last_pos")
        if last is None:
            return
        self._replaying = True
        try:
            self.drag_move(last)
        finally:
            self._replaying = False

    def _cancel_drag(self) -> None:
        """Esc / 外部取消：撤浮影、恢复原位 icon、解冻托盘、丢弃预览。"""
        drag = self._drag
        if drag is None:
            return
        ghost = drag.get("ghost")
        if ghost is not None:
            ghost.hide()
            ghost.deleteLater()
        widget = drag.get("widget")
        if widget is not None:
            widget.setGraphicsEffect(None)
        self._drag = None
        self._last_preview_sig = None
        self._board.set_highlight(None)
        self._board._page_override = None
        self._tray_panel.set_frozen(False)
        self._refresh(animate=True)

    def end_drag(self, global_pos: QPoint) -> None:
        if self._drag is None:
            return
        drag = self._drag
        # 硬不变量：落位目标 = 最后展示的预览目标（不重算，避免"预览要落位、
        # 松手却落空/换目标"）
        target = drag.get("target")
        anchor = drag.get("anchor")
        token = getattr(drag.get("widget"), "token", None)
        self._drag = None
        self._last_preview_sig = None
        preview, ok, _info = self._preview_with(drag, target)
        widget = drag["widget"]
        ghost = drag.get("ghost")
        if ghost is not None:
            ghost.hide()
            ghost.deleteLater()
        widget.setGraphicsEffect(None)
        # 删源 widget 前先掉动画，统一用 widget.token（多选时 ids[0] 未必是被按下的）
        if token is not None:
            if drag["from"] == "tray":
                self._tray_panel._drop_anim(token)
            else:
                self._board._drop_anim(token)
        if ok and preview is not None:
            self._state = preview
            self._selection = {
                t for t in self._selection if t in ordered_ids(self.current_page_state())
            }
        if drag["from"] == "grid":
            if ok and target and target[0] == "tray" and token is not None:
                widget.hide()
                self._board.icons.pop(token, None)
                widget.deleteLater()
        elif ok and token is not None:
            widget.hide()
            self._tray_panel.chips.pop(token, None)
            widget.deleteLater()
        # 其余情况原位 icon 保留，下一轮 sync 会滑到落点/滑回原位
        self._board.set_highlight(None)
        self._board._page_override = None
        self._tray_panel.set_frozen(False)
        self._refresh(animate=True, spawn_snap=True)
        # 两段式：先让托盘高度/棋盘尺寸落地，再按最终格位收拢
        self.layout().activate()
        self._tray_panel._arrange(True)
        self.layout().activate()
        # 只有「有预览且落位成功」才收拢；托盘→托盘、死区、出界都不收拢
        if ok and preview is not None and anchor:
            self._settle_anchor(anchor, global_pos)

    @staticmethod
    def _clamp_to(container: QWidget, global_pos: QPoint) -> QPoint:
        """把松手点映射到容器坐标并夹到容器矩形内。"""
        pos = container.mapFromGlobal(global_pos)
        rect = container.rect()
        return QPoint(
            min(max(pos.x(), rect.left()), max(rect.left(), rect.right())),
            min(max(pos.y(), rect.top()), max(rect.top(), rect.bottom())),
        )

    def _settle_anchor(self, token: str, release_global: QPoint) -> None:
        """只收拢 anchor 一张：棋盘侧/托盘侧按最终容器分别起动画。"""
        if token in self._board.icons:
            self._board.settle_token(token, self._clamp_to(self._board, release_global))
        elif token in self._tray_panel.chips:
            self._tray_panel.settle_token(
                token, self._clamp_to(self._tray_panel, release_global)
            )

    def _preview_with(self, drag: dict, target):
        """end_drag 用：drag 已从 self._drag 摘下，直接按参数算。"""
        saved = self._drag
        self._drag = drag
        try:
            return self._preview(target)
        finally:
            self._drag = saved

    # ------------------------------------------------------------------
    # 按钮
    # ------------------------------------------------------------------
    def _on_kick(self) -> None:
        if self._drag is not None:
            return
        if not self._selection:
            return
        moved = kick_to_tray(self._state, self._page_index, list(self._selection))
        self._selection.clear()
        self._anchor = None
        self._status.setText(f"已踢出 {moved} 个到等候区，后续图前向补洞")
        self._refresh(animate=True)

    def _on_backfill(self) -> None:
        if self._drag is not None:
            return
        placed = backfill(self._state, self._page_index)
        self._status.setText(
            f"已回填 {placed} 个到「{self.current_page_state().title}」末尾空位"
            if placed else "本页没有空位或等候区为空"
        )
        self._refresh(animate=True)

    def _on_reset(self) -> None:
        if self._drag is not None:
            return
        self._state = clone_state(self._initial)
        self._selection.clear()
        self._anchor = None
        self._status.setText("已重置为打开对话框时的快照")
        self._refresh(animate=True)

    def _on_accept(self) -> None:
        if self._drag is not None:
            return
        if not can_commit(self._state):
            return
        if not self._structure_ok():
            QMessageBox.information(
                self, "整理子图", "内部状态异常（图数不一致），请先「重置」再试。"
            )
            return
        if not validate_state(self._snapshot, self._state):
            QMessageBox.information(
                self, "整理子图",
                "数据或曲线在整理过程中发生了变化，请关闭后重新打开。",
            )
            return
        self.accept()

    def _on_cancel_clicked(self) -> None:
        """拖拽中只取消拖拽（不关窗、不改状态）；否则正常取消。"""
        if self._drag is not None:
            self._cancel_drag()
            return
        self.reject()

    # ------------------------------------------------------------------
    # 键盘
    # ------------------------------------------------------------------
    def keyPressEvent(self, event):
        key = event.key()
        # 拖拽期只认 Esc（取消拖动）；Del/Ctrl+A 等不得在半途改状态
        if self._drag is not None:
            if key == Qt.Key.Key_Escape:
                self._cancel_drag()
            return
        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            if self._selection:
                self._on_kick()
                return
        elif key == Qt.Key.Key_Escape:
            if self._selection:
                self.clear_selection()
                return
        elif key == Qt.Key.Key_A and event.modifiers() & (
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier
        ):
            self._selection = set(ordered_ids(self.current_page_state()))
            self._refresh(animate=True)
            return
        super().keyPressEvent(event)