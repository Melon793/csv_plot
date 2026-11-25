"""曲线暂存区（Curve Shelf）：跨绘图页搬运曲线的非模态浮窗托盘。

设计口径（tmp/2026-11-13/curve-transfer-preview-plan.md）：
- 托盘条目自持全量样式快照（入区即拷完，出区零源依赖）；
- legend → 托盘：无修饰=剪切（源端延迟删除）/ Alt(⌥)=复制；
- 变量列表 → 托盘：名字入区，默认样式，复制语义；
- 托盘 → plot：无修饰=移出（条目消失）/ Alt(⌥)=复制出（条目保留）；
- stale（reload 后变量不在新数据中）置灰不可拖，删除入口保留。

本模块只放数据模型 + 托盘 QListWidget + 浮窗壳；plot 侧批量协议见
PlotWidget._transfer_curves 与 _handle_shelf_drop。
"""

from __future__ import annotations

import html
import uuid
import weakref
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QEvent, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QDrag, QIcon, QPainter, QPalette, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.core.config import widget_alive
from src.core.data_types import (
    DRAW_MODE_AUTO,
    DRAW_MODE_SOURCE_AUTO,
    LINE_PRESET_AUTO,
    SYMBOL_POLICY_AUTO,
)
from src.core.logger import get_logger
from src.ui.drag_drop import (
    LEGEND_MIME_FORMAT,
    SHELF_MIME_FORMAT,
    build_shelf_mimedata,
    clear_active_shelf_drag,
    get_active_legend_drag_source,
    is_copy_modifier,
    parse_var_names_from_mimedata,
    set_active_shelf_drag,
)
from src.ui.drag_visuals import (
    DragItem,
    build_drag_pixmap,
    drag_hotspot,
    transfer_hint,
)
from src.ui.transfer_broadcast import (
    broadcast_return,
    broadcast_text,
    broadcast_transfer,
    loading_reject_message,
    source_closed_message,
)

logger = get_logger("widget.curve_shelf")

SHELF_ITEM_HEIGHT = 26
SHELF_KEY_ROLE = Qt.ItemDataRole.UserRole + 101
SHELF_STALE_MARK = " ✕"


@dataclass
class ShelfEntry:
    """托盘条目：入区时从源 CurveInfo 拷贝的全量样式快照。"""

    key: str
    var_name: str
    color: str = ""
    visible: bool = True
    draw_mode: str = DRAW_MODE_AUTO
    symbol_policy: str = SYMBOL_POLICY_AUTO
    # 快照保真：出区落地后必须回写，不得被 apply 入口误钉成 'user'
    draw_mode_source: str = DRAW_MODE_SOURCE_AUTO
    line_preset: str = LINE_PRESET_AUTO
    # 编辑器强化轮 D3/D6：显示名/换算随暂存保真（出区落地回写）
    display_name: str = ""
    factor: float = 1.0
    offset: float = 0.0
    origin_label: str = ""
    stale: bool = False
    #: 剪切入区的来源绘图（弱引用，不延长生命周期）；复制/变量列表入区为 None
    origin_plot_ref: weakref.ReferenceType | None = None

    @classmethod
    def from_curve_info(
        cls,
        var_name: str,
        ci: Any,
        origin_label: str = "",
        *,
        origin_plot: Any = None,
    ) -> "ShelfEntry":
        """从源曲线快照全量样式（须在 drop 调用栈内、源存活时读完）。

        ``origin_plot`` 只在无修饰剪切入区时传入；复制不记来源，避免
        「原图仍有一条、退回却无事可做」的假入口。
        """
        ref = None
        if origin_plot is not None:
            try:
                ref = weakref.ref(origin_plot)
            except TypeError:
                # 测试替身等不可弱引用对象：退回入口自然降级为不可用
                logger.debug("暂存来源不可弱引用：%r", type(origin_plot))
        return cls(
            key=uuid.uuid4().hex,
            var_name=var_name,
            color=getattr(ci, "color", "") or "",
            visible=bool(getattr(ci, "visible", True)),
            draw_mode=getattr(ci, "draw_mode", DRAW_MODE_AUTO),
            symbol_policy=getattr(ci, "symbol_policy", SYMBOL_POLICY_AUTO),
            draw_mode_source=getattr(
                ci, "draw_mode_source", DRAW_MODE_SOURCE_AUTO
            ),
            line_preset=getattr(ci, "line_preset", LINE_PRESET_AUTO),
            display_name=getattr(ci, "display_name", "") or "",
            factor=getattr(ci, "factor", 1.0),
            offset=getattr(ci, "offset", 0.0),
            origin_label=origin_label,
            origin_plot_ref=ref,
        )

    @classmethod
    def from_name(cls, var_name: str, origin_label: str = "变量列表") -> "ShelfEntry":
        """变量列表来源：默认样式，复制语义。"""
        return cls(key=uuid.uuid4().hex, var_name=var_name, origin_label=origin_label)

    def origin_plot(self) -> Any | None:
        """存活的来源绘图；引用为空 / 目标已销毁返回 None。

        ``widget_alive`` 自身已判 None（``config.py`` 的实现是
        ``widget is not None and ...``）；这里显式写 None 分支只是让意图清楚。
        """
        plot = self.origin_plot_ref() if self.origin_plot_ref is not None else None
        return plot if widget_alive(plot) else None

    def style_summary(self) -> str:
        """tooltip 用的样式摘要。"""
        vis = "显示" if self.visible else "隐藏"
        # 第 6 轮 B-2：预设纳入摘要（字段已进搬运链，摘要漏格会信息缺失）
        preset = getattr(self, "line_preset", LINE_PRESET_AUTO)
        preset_part = f" · 预设：{preset}" if preset != LINE_PRESET_AUTO else ""
        return (
            f"{self.var_name}\n来源：{self.origin_label or '未知'}\n"
            f"颜色：{self.color or '默认'} · {vis}\n"
            f"绘制方式：{self.draw_mode} · 数据点：{self.symbol_policy}{preset_part}"
        )


def _color_icon(color: str, size: int = 12) -> QIcon:
    """色块小图标（QListWidget 条目左侧）。"""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor(color) if color else QColor("#808080"))
    painter.setPen(QColor(0, 0, 0, 60))
    painter.drawRoundedRect(0, 0, size - 1, size - 1, 3, 3)
    painter.end()
    return QIcon(pixmap)


class CurveShelfWidget(QListWidget):
    """托盘列表：多选拖出 + legend/变量列表拖入 + stale 置灰。

    宿主窗口（MainWindow）经 ``main_window`` 注入：用于加载闸门与统一
    播报；独立测试可留空，闸门/播报自动降级为 no-op。
    """

    entries_changed = Signal()

    def __init__(self, main_window: Any = None, parent: QWidget | None = None):
        super().__init__(parent)
        self._main_window = main_window
        self._entries: dict[str, ShelfEntry] = {}
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setAlternatingRowColors(False)

    # ------------------------------------------------------------------
    # 查询 / 条目管理
    # ------------------------------------------------------------------
    @property
    def entries(self) -> list[ShelfEntry]:
        """按界面顺序返回条目。"""
        result = []
        for row in range(self.count()):
            item = self.item(row)
            entry = self._entries.get(item.data(SHELF_KEY_ROLE)) if item else None
            if entry is not None:
                result.append(entry)
        return result

    def entry_for_key(self, key: str) -> ShelfEntry | None:
        return self._entries.get(key)

    def entries_for_keys(self, keys: list[str]) -> list[ShelfEntry]:
        return [self._entries[k] for k in keys if k in self._entries]

    def entry_for_name(self, var_name: str) -> ShelfEntry | None:
        for entry in self._entries.values():
            if entry.var_name == var_name:
                return entry
        return None

    def selected_entries(self) -> list[ShelfEntry]:
        """按界面行序返回选中条目（拖出 MIME/预览图顺序稳定）。"""
        items = sorted(self.selectedItems(), key=self.row)
        return [
            entry
            for item in items
            if (entry := self._entries.get(item.data(SHELF_KEY_ROLE))) is not None
        ]

    def selected_keys(self) -> list[str]:
        return [entry.key for entry in self.selected_entries()]

    def add_entry(self, entry: ShelfEntry) -> str:
        """入区（按变量名去重）：返回 "added" / "updated"。"""
        existing = self.entry_for_name(entry.var_name)
        if existing is not None:
            # D10：按名去重更新（保留原 key，避免在飞 MIME 失效）
            for attr in (
                "color",
                "visible",
                "draw_mode",
                "symbol_policy",
                "draw_mode_source",
                "line_preset",
                "display_name",
                "factor",
                "offset",
                "origin_label",
                # 来源引用必须跟随最近一次入区：剪入 A 后又从 B 剪入同名
                # 变量时，退回目标必须是 B（漏掉会静默退错图）
                "origin_plot_ref",
            ):
                setattr(existing, attr, getattr(entry, attr))
            existing.stale = False
            self._refresh_item(existing)
            self.entries_changed.emit()
            return "updated"

        self._entries[entry.key] = entry
        item = QListWidgetItem()
        item.setData(SHELF_KEY_ROLE, entry.key)
        item.setSizeHint(QSize(0, SHELF_ITEM_HEIGHT))
        self.addItem(item)
        self._refresh_item(entry)
        self.entries_changed.emit()
        return "added"

    def remove_entries(self, keys: list[str]) -> int:
        """删除指定 key 的条目；返回实际删除数。"""
        removed = 0
        key_set = {str(k) for k in keys}
        for row in reversed(range(self.count())):
            item = self.item(row)
            if item is None or item.data(SHELF_KEY_ROLE) not in key_set:
                continue
            self._entries.pop(item.data(SHELF_KEY_ROLE), None)
            self.takeItem(row)
            removed += 1
        if removed:
            self.entries_changed.emit()
        return removed

    def clear_entries(self) -> int:
        count = self.count()
        if count:
            self._entries.clear()
            self.clear()
            self.entries_changed.emit()
        return count

    def revalidate(self, var_names: set[str]) -> list[str]:
        """reload 后置灰：变量不在新数据中 → stale；重新出现则恢复。

        Returns:
            本次新变为 stale 的变量名列表（供调用方汇总播报）。
        """
        newly_stale: list[str] = []
        changed = False
        for entry in self._entries.values():
            want_stale = entry.var_name not in var_names
            if entry.stale != want_stale:
                entry.stale = want_stale
                changed = True
                if want_stale:
                    newly_stale.append(entry.var_name)
            self._refresh_item(entry)
        if changed:
            self.entries_changed.emit()
        return newly_stale

    def _refresh_item(self, entry: ShelfEntry) -> None:
        """按条目状态重绘文本 / 颜色 / 拖拽标志。"""
        item = self.item_for_key(entry.key)
        if item is None:
            return
        text = entry.var_name
        if entry.origin_label:
            text += f"    {entry.origin_label}"
        if entry.stale:
            text += SHELF_STALE_MARK
        item.setText(text)
        item.setIcon(_color_icon(entry.color))
        tooltip = entry.style_summary()
        if entry.stale:
            # 第 6 轮 P2-51：stale 用 Disabled Text（暗色下硬编码灰不可读）
            item.setForeground(
                self.palette().color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text)
            )
            tooltip += "\n当前数据中不存在（不可拖出）"
        else:
            # 评审 NB1：清空前景 brush 让 delegate 走实时 palette——
            # 快照式 setForeground 在运行期暗色切换后不重刷，且会覆盖
            # 选中态 HighlightedText
            item.setData(Qt.ItemDataRole.ForegroundRole, None)
            extra = self._return_tooltip_line(entry)
            if extra:
                tooltip += "\n" + extra
        # 第 6 轮 P2-50：与 var_tooltip 同口径转义 + <qt> 富文本还原，
        # 变量名含 <b> 等序列不被当标签吃掉、换行不塌
        item.setToolTip("<qt>" + html.escape(tooltip).replace("\n", "<br>") + "</qt>")
        flags = (
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsDragEnabled
        )
        if entry.stale:
            flags &= ~Qt.ItemFlag.ItemIsDragEnabled
        item.setFlags(flags)

    def refresh_entries(self, names: list[str]) -> None:
        """按变量名刷新条目展示/状态行（延迟删源后刷新退回判定）。"""
        changed = False
        for name in names:
            entry = self.entry_for_name(name)
            if entry is None:
                continue
            self._refresh_item(entry)
            changed = True
        if changed:
            self.entries_changed.emit()

    def item_for_key(self, key: str) -> QListWidgetItem | None:
        for row in range(self.count()):
            item = self.item(row)
            if item is not None and item.data(SHELF_KEY_ROLE) == key:
                return item
        return None

    # ------------------------------------------------------------------
    # 拖出
    # ------------------------------------------------------------------
    def startDrag(self, supportedActions):
        """拖出：只取非 stale 的选中条目，MIME 携带 key 列表 + 文本兼容层。"""
        entries = [entry for entry in self.selected_entries() if not entry.stale]
        if not entries:
            return
        names = [entry.var_name for entry in entries]
        keys = [entry.key for entry in entries]
        drag = QDrag(self)
        drag.setMimeData(build_shelf_mimedata(keys, names, self))
        pixmap = build_drag_pixmap(
            [DragItem(entry.var_name, entry.color or None) for entry in entries],
            self.font(),
            hint=transfer_hint(),
        )
        if pixmap:
            drag.setPixmap(pixmap)
            drag.setHotSpot(drag_hotspot())
        set_active_shelf_drag(self, keys)
        try:
            # 无修饰=移出、Alt=复制出，由 drop 端按修饰键决定语义
            drag.exec(
                Qt.DropAction.CopyAction | Qt.DropAction.MoveAction
            )
        finally:
            clear_active_shelf_drag()
            # P1-6（第 5 轮）：QDrag(self) 的 C++ 所有权挂在源控件下，exec
            # 返回后不 deleteLater 会连同 QMimeData/缩略图累积到源控件销毁。
            # exec 期间源控件被销毁时 C++ 对象已亡，静默跳过（评审 N1）。
            try:
                drag.deleteLater()
            except RuntimeError:
                pass

    # ------------------------------------------------------------------
    # 拖入（legend / 变量列表）
    # ------------------------------------------------------------------
    def _host_window(self) -> Any:
        if self._main_window is not None:
            try:
                self._main_window.window()
            except RuntimeError:
                self._main_window = None
            else:
                return self._main_window
        return self.window()

    def _announce(self, message: str, level: str = "info") -> None:
        show = getattr(self._host_window(), "_show_status_message", None)
        if callable(show):
            show(message)
        else:
            logger.info(message)

    def _reject_when_loading(self, action: str) -> bool:
        host = self._host_window()
        is_loading = getattr(host, "is_data_loading", None)
        if callable(is_loading) and is_loading():
            # 搬运口径的统一文案（§5.3），不复用通用闸门的「请稍候再…」
            broadcast_text(host, loading_reject_message())
            return True
        reject = getattr(host, "reject_when_loading", None)
        return bool(reject and reject(action))

    def _can_accept_mime(self, mime) -> bool:
        if mime is None or not mime.hasText():
            return False
        # 托盘自身拖回托盘不做重排（无意义），交给 plot drop 端消费
        if mime.hasFormat(SHELF_MIME_FORMAT):
            return False
        return True

    def dragEnterEvent(self, event):
        if self._can_accept_mime(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if self._can_accept_mime(event.mimeData()):
            if event.mimeData().hasFormat(LEGEND_MIME_FORMAT):
                # 光标与 drop 语义一致（Alt=复制，无修饰=剪切）
                event.setDropAction(
                    Qt.DropAction.CopyAction
                    if is_copy_modifier(event.keyboardModifiers())
                    else Qt.DropAction.MoveAction
                )
                event.accept()
            else:
                event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        mime = event.mimeData()
        if not self._can_accept_mime(mime):
            event.ignore()
            return
        if self._reject_when_loading("暂存变量"):
            event.ignore()
            return
        names = parse_var_names_from_mimedata(mime)
        if not names:
            event.ignore()
            return

        if mime.hasFormat(LEGEND_MIME_FORMAT):
            self._drop_from_legend(event, names)
        else:
            self._drop_from_variable_list(event, names)

    def _drop_from_legend(self, event, names: list[str]) -> None:
        """legend 来源：无修饰=剪切入区（源端延迟删除）/ Alt=复制入区。"""
        source = get_active_legend_drag_source(event.mimeData())
        if source is None:
            # 源已销毁：快照不可读，拒绝入区（区别于 plot drop 端的按复制降级）
            broadcast_text(self._host_window(), source_closed_message(names))
            event.ignore()
            return

        copy_mode = is_copy_modifier(event.keyboardModifiers())
        origin = self._origin_label_for(source)
        # 快照必须在 drop 调用栈内读完（源端删除延迟一拍）
        added: list[str] = []
        updated = 0
        for name in names:
            ci = source.curves.get(name)
            if ci is None:
                continue
            entry = ShelfEntry.from_curve_info(
                name,
                ci,
                origin_label=origin,
                # 复制入区不记来源：原图仍有该曲线，退回无事可做
                origin_plot=None if copy_mode else source,
            )
            status = self.add_entry(entry)
            if status == "updated":
                updated += 1
            added.append(name)

        if not added:
            self._announce(
                f"曲线不可用，无法暂存 {'、'.join(names)}", level="warn"
            )
            event.ignore()
            return

        if not copy_mode:
            self._schedule_source_remove(source, added)
            # 源端删除是延迟一拍：入区瞬间的 present 判定会在删除后失效，
            # 必须再刷一次 tooltip/按钮（否则一直显示「原图已有」）
            self._schedule_post_cut_refresh(added)

        event.acceptProposedAction()
        self._announce_shelf_intake(
            cut=not copy_mode, names=added, origin=origin, updated=updated
        )

    def _drop_from_variable_list(self, event, names: list[str]) -> None:
        """变量列表来源：名字入区，默认样式，复制语义。"""
        updated = 0
        for name in names:
            if self.add_entry(ShelfEntry.from_name(name)) == "updated":
                updated += 1
        event.acceptProposedAction()
        self._announce_shelf_intake(
            cut=False, names=names, origin="变量列表", updated=updated
        )

    def _schedule_source_remove(self, source: Any, names: list[str]) -> None:
        """剪切语义：延迟一拍删除源曲线（避开 drop 栈内场景中间态）。"""

        def _make_remove(src: Any, var: str):
            def _remove():
                if not widget_alive(self):
                    return
                source_remains = False
                try:
                    removed = src.remove_variable_from_plot(var)
                    source_remains = (
                        removed is False
                        and widget_alive(src)
                        and var in getattr(src, "curves", {})
                    )
                except RuntimeError:
                    source_remains = (
                        widget_alive(src)
                        and var in getattr(src, "curves", {})
                    )
                    if source_remains:
                        logger.warning(
                            "暂存区入区：源曲线删除失败 %s", var, exc_info=True
                        )
                    else:
                        logger.debug("暂存区入区：源绘图已销毁，跳过删除")
                except Exception:
                    source_remains = (
                        widget_alive(src)
                        and var in getattr(src, "curves", {})
                    )
                    logger.warning(
                        "暂存区入区：源曲线删除失败 %s", var, exc_info=True
                    )
                if not source_remains:
                    return

                entry = self.entry_for_name(var)
                if entry is not None:
                    self.remove_entries([entry.key])
                message = (
                    f"曲线 {var} 删除失败，已保留原图曲线并移除暂存快照"
                )
                show = getattr(self._host_window(), "_show_status_message", None)
                if callable(show):
                    show(message, level="warn")
                else:
                    logger.warning(message)

            return _remove

        for name in names:
            QTimer.singleShot(0, _make_remove(source, name))

    def _schedule_post_cut_refresh(self, names: list[str]) -> None:
        """延迟删源后刷新状态行（排在所有删除 singleShot 之后）。"""

        def _refresh():
            if not widget_alive(self):
                return
            self.refresh_entries(list(names))

        QTimer.singleShot(0, _refresh)

    def _announce_shelf_intake(
        self, *, cut: bool, names: list[str], origin: str, updated: int
    ) -> None:
        broadcast_transfer(
            self._host_window(),
            "cut" if cut else "shelf_copy",
            names,
            src_label=origin,
            updated=updated,
        )

    def _resolve_origin(self, entry: ShelfEntry) -> tuple[Any | None, str]:
        """解析条目可退回的来源绘图 → ``(plot, live_label)``。

        不可退回返回 ``(None, "")``。判定必须覆盖三层：弱引用目标存活、
        所属页仍在宿主 ``pages`` 里、绘图仍在该页 ``plot_widgets`` 里。
        后两条由 ``_origin_label_for`` 的成员身份反算一并完成；按标题/序号
        猜测的路径一律不走（重排/改名后会退错目标）。
        """
        plot = entry.origin_plot()
        host = self._host_window()
        if plot is None or host is None:
            return None, ""
        if getattr(plot, "_is_being_destroyed", False):
            # deleteLater 与包装器存活之间的窗口：此时落地会撞 RuntimeError
            return None, ""
        label = self._origin_label_for(plot)
        return (plot, label) if label else (None, "")

    def _origin_label_for(self, source: Any) -> str:
        """反算来源徽章「页N·图M」（快照语义：之后重命名/重排不追更）。"""
        page = getattr(source, "page", None)
        host = self._host_window()
        if page is None or host is None:
            return ""
        try:
            page_index = host.page_index(page)
            plot_index = next(
                (
                    index
                    for index, container in enumerate(page.plot_widgets)
                    if getattr(container, "plot_widget", None) is source
                ),
                -1,
            )
        except Exception:
            return ""
        if page_index < 0 or plot_index < 0:
            return ""
        return f"页{page_index + 1}·图{plot_index + 1}"

    # ------------------------------------------------------------------
    # 退回原图（撤销剪切）
    # ------------------------------------------------------------------
    def _return_block_reason(self, entry: ShelfEntry) -> str:
        """不可退回原因；空串 = 可退回。

        返回值同时供菜单/按钮启用条件使用，保证「能看见」与「点了有效」
        共用同一套判定。
        """
        if entry.stale:
            return "stale"
        if entry.origin_plot_ref is None:
            return "no_origin"
        plot, _ = self._resolve_origin(entry)
        if plot is None:
            return "closed"
        if entry.var_name in getattr(plot, "curves", {}):
            return "present"
        return ""

    def returnable_entries(
        self, entries: list[ShelfEntry] | None = None
    ) -> list[ShelfEntry]:
        """筛选可退回条目；None → 当前选中集。菜单/按钮共用。"""
        candidates = self.selected_entries() if entries is None else list(entries)
        return [
            entry
            for entry in candidates
            if self._return_block_reason(entry) == ""
        ]

    def _return_tooltip_line(self, entry: ShelfEntry) -> str:
        """条目 tooltip 的退回状态行（懒计算，选择变化时刷新）。"""
        try:
            reason = self._return_block_reason(entry)
            if reason == "no_origin":
                return "复制入区：没有可退回的剪切（删除条目即可丢弃）"
            if reason == "closed":
                return "原绘图已关闭，无法退回（可拖到目标图）"
            if reason == "present":
                return "原图已有该曲线，无需退回"
            if reason == "stale":
                return ""
            plot, label = self._resolve_origin(entry)
            if plot is None:
                return ""
            lines = ["可退回原图"]
            if label and label != entry.origin_label:
                lines.append(f"当前位置：{label}（行内为入区快照）")
            return "\n".join(lines)
        except Exception:
            logger.debug("暂存条目 tooltip 计算失败", exc_info=True)
            return ""

    def _scene_for(self, plot: Any) -> bool:
        """退回落地的 scene 口径：仅当前页允许写场景。

        与 ``cursor_sync_manager._defers_scene_write`` 同义；这里不直接调
        私有方法：``current_page is None``（关页/启动窗口期）必须判成延迟，
        否则隐藏页会重新写游标场景。
        """
        host = self._host_window()
        page = getattr(plot, "page", None)
        current = getattr(host, "current_page", None)
        return page is not None and current is page

    def _origin_cell_hidden(self, plot: Any) -> bool:
        """原图所在格是否被布局隐藏（仅用于播报提示）。"""
        page = getattr(plot, "page", None)
        if page is None:
            return False
        try:
            index = next(
                (
                    i
                    for i, container in enumerate(page.plot_widgets)
                    if getattr(container, "plot_widget", None) is plot
                ),
                -1,
            )
            cell_visible = getattr(page, "cell_visible", None)
            return bool(
                index >= 0 and callable(cell_visible) and not cell_visible(index)
            )
        except Exception:
            return False

    def return_entries(self, keys: list[str]) -> list[str]:
        """把指定条目退回各自被剪出的原图；返回实际退回的变量名。"""
        host = self._host_window()
        if self._reject_when_loading("退回暂存变量"):
            return []

        groups: dict[int, tuple[Any, str, list[ShelfEntry]]] = {}
        skipped: list[str] = []
        failed: list[str] = []
        origin_closed: list[str] = []
        no_origin: list[str] = []
        for entry in self.entries_for_keys(keys):
            reason = self._return_block_reason(entry)
            if reason == "stale":
                continue
            if reason == "no_origin":
                no_origin.append(entry.var_name)
                continue
            if reason == "closed":
                origin_closed.append(entry.var_name)
                continue
            if reason == "present":
                skipped.append(entry.var_name)
                continue
            plot, label = self._resolve_origin(entry)
            groups.setdefault(id(plot), (plot, label, []))[2].append(entry)

        returned_groups: list[tuple[str, list[str], bool]] = []
        returned_names: list[str] = []
        for plot, label, group_entries in groups.values():
            scene = self._scene_for(plot)
            hidden = self._origin_cell_hidden(plot)

            def _remove_added(added: list[str], entries=group_entries) -> None:
                # 只删成功落地者，按 key 精确删除（防同名重入误删）
                added_set = set(added)
                self.remove_entries(
                    [entry.key for entry in entries if entry.var_name in added_set]
                )

            try:
                result = plot.land_shelf_entries(
                    group_entries,
                    batch=len(group_entries) > 1 or not scene,
                    scene=scene,
                    remove_entries_cb=_remove_added,
                )
            except Exception:
                # 单组异常不中断后续组；条目保留是安全侧（可能已落地，
                # 最坏是重复条目，绝不丢曲线）
                logger.warning("退回原图失败：%s", label, exc_info=True)
                failed.extend(entry.var_name for entry in group_entries)
                continue

            if result.added:
                returned_groups.append((label, list(result.added), hidden))
                returned_names.extend(result.added)
            skipped.extend(result.skipped)
            failed.extend(result.failed)

        broadcast_return(
            host,
            returned_groups,
            skipped=skipped,
            failed=failed,
            origin_closed=origin_closed,
            no_origin=no_origin,
        )
        return returned_names

    def return_selected(self) -> list[str]:
        """退回选中条目（按钮/菜单共用入口）。"""
        return self.return_entries(self.selected_keys())

    # ------------------------------------------------------------------
    # 条目删除（右键 / 清空）
    # ------------------------------------------------------------------
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            # 显式消费，避免 QAbstractItemView 把中键当选中并清掉用户多选。
            event.accept()
            return
        super().mousePressEvent(event)

    def contextMenuEvent(self, event):
        item = self.itemAt(event.pos())
        menu = QMenu(self)
        return_act = None
        return_keys: list[str] = []
        delete_act = None
        delete_keys: list[str] = []
        if item is not None:
            key = item.data(SHELF_KEY_ROLE)
            if item.isSelected():
                delete_keys = [k for k in self.selected_keys() if k]
            else:
                delete_keys = [key]
            candidates = self.selected_entries() if item.isSelected() else [
                entry
                for entry in (self.entry_for_key(key),)
                if entry is not None
            ]
            # 菜单打开时刷新 tooltip：原图状态可能已变化（关闭/已有同名）
            for entry in candidates:
                self._refresh_item(entry)
            returnable = self.returnable_entries(candidates)
            if returnable:
                label = (
                    f"退回原图（{len(returnable)}）"
                    if len(returnable) > 1
                    else "退回原图"
                )
                return_act = menu.addAction(label)
                return_keys = [entry.key for entry in returnable]
                menu.addSeparator()
            label = (
                f"删除所选条目（{len(delete_keys)}）"
                if len(delete_keys) > 1
                else "删除条目"
            )
            delete_act = menu.addAction(label)
        clear_act = menu.addAction("清空暂存区")
        clear_act.setEnabled(self.count() > 0)

        chosen = menu.exec(event.globalPos())
        menu.deleteLater()
        if chosen is return_act and return_keys:
            self.return_entries(return_keys)
        elif chosen is delete_act and delete_keys:
            if len(delete_keys) > 1:
                reply = QMessageBox.question(
                    self,
                    "删除暂存条目",
                    f"确定要删除暂存的 {len(delete_keys)} 个条目吗？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return
            self.remove_entries(delete_keys)
        elif chosen is clear_act:
            self.request_clear()

    def request_clear(self) -> int:
        """清空入口（按钮/菜单共用）：N>1 一次汇总确认。"""
        count = self.count()
        if count == 0:
            return 0
        if count > 1:
            reply = QMessageBox.question(
                self,
                "清空暂存区",
                f"确定要清空暂存的 {count} 个条目吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return 0
        return self.clear_entries()


class CurveShelfDialog(QDialog):
    """非模态浮窗壳（Qt.Tool）：几何记忆 + 计数标题 + 清空/收起。"""

    visibility_changed = Signal(bool)

    def __init__(self, main_window: Any = None, parent: QWidget | None = None):
        super().__init__(parent if parent is not None else main_window)
        self.main_window = main_window
        self.setWindowTitle("曲线暂存区")
        self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setModal(False)
        self.setMinimumSize(260, 220)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title_row = QHBoxLayout()
        self.title_label = QLabel("暂存区 (0)", self)
        self.title_label.setStyleSheet("font-weight: bold;")
        title_row.addWidget(self.title_label)
        title_row.addStretch(1)
        layout.addLayout(title_row)

        self.shelf = CurveShelfWidget(main_window=main_window, parent=self)
        layout.addWidget(self.shelf, 1)

        btn_row = QHBoxLayout()
        self.return_btn = QPushButton("退回原图", self)
        self.return_btn.setToolTip("选中暂存条目后可退回它被剪出的绘图区")
        self.return_btn.clicked.connect(self.shelf.return_selected)
        self.clear_btn = QPushButton("清空", self)
        self.clear_btn.clicked.connect(self.shelf.request_clear)
        self.close_btn = QPushButton("收起", self)
        self.close_btn.clicked.connect(self.hide)
        btn_row.addWidget(self.return_btn)
        btn_row.addWidget(self.clear_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self.close_btn)
        layout.addLayout(btn_row)

        self.shelf.entries_changed.connect(self._on_entries_changed)
        self.shelf.itemSelectionChanged.connect(self._on_selection_changed)
        self._update_return_btn()
        self._restore_geometry()

    # ------------------------------------------------------------------
    # 标题 / 可见性
    # ------------------------------------------------------------------
    def _on_entries_changed(self) -> None:
        count = self.shelf.count()
        self.title_label.setText(f"暂存区 ({count})")
        self.setWindowTitle(f"曲线暂存区 ({count})")
        self.clear_btn.setEnabled(count > 0)
        self._update_return_btn()

    def _on_selection_changed(self) -> None:
        """选中集变化：刷新 tooltip 状态行与退回按钮可用性。"""
        for entry in self.shelf.selected_entries():
            self.shelf._refresh_item(entry)
        self._update_return_btn()

    def _update_return_btn(self) -> None:
        self.return_btn.setEnabled(bool(self.shelf.returnable_entries()))

    def showEvent(self, event):
        super().showEvent(event)
        self._on_entries_changed()
        self.visibility_changed.emit(True)

    def hideEvent(self, event):
        self._save_geometry()
        super().hideEvent(event)
        self.visibility_changed.emit(False)

    def changeEvent(self, event):
        """最小化/恢复时同步开合态。

        部分平台最小化不伴随 hideEvent，单靠 hideEvent 会让胶囊停在勾选态，
        下一次点击被当成"关闭"，窗口永远唤不回。这里把"最小化"也视为关。
        """
        if event.type() == QEvent.Type.WindowStateChange:
            self.visibility_changed.emit(
                self.isVisible() and not self.isMinimized()
            )
        super().changeEvent(event)

    def closeEvent(self, event):
        """关闭 = 隐藏单例（条目保留），几何落盘。"""
        self._save_geometry()
        event.accept()

    # ------------------------------------------------------------------
    # reload 生命周期
    # ------------------------------------------------------------------
    def revalidate_against_loader(self, loader: Any) -> list[str]:
        """reload 后按新 loader 变量集置灰；返回新 stale 列表。"""
        var_names = set(getattr(loader, "var_names", []) or [])
        return self.shelf.revalidate(var_names)

    # ------------------------------------------------------------------
    # 几何记忆（抄 plot_variable_editor 的多显示器拔插守卫）
    # ------------------------------------------------------------------
    def _geometry_on_screen(self) -> bool:
        from PySide6.QtGui import QGuiApplication

        geometry = self.normalGeometry()
        if geometry.isNull():
            return True
        return any(
            geometry.intersects(screen.availableGeometry())
            for screen in QGuiApplication.screens()
        )

    def _restore_geometry(self) -> None:
        from PySide6.QtGui import QGuiApplication

        from src.core.settings import AppSettings

        settings = AppSettings()
        geometry = settings.get_curve_shelf_geometry()
        if (
            geometry is not None
            and self.restoreGeometry(geometry)
            and self._geometry_on_screen()
        ):
            return
        self.resize(300, 260)
        # 默认落点：页签栏右端胶囊下方 6px、右对齐。getattr 兜底是硬要求：
        # 组件测试的 ShelfHost 没有这个方法，不能 AttributeError。
        anchor_getter = getattr(self.main_window, "curve_shelf_anchor_global", None)
        anchor = anchor_getter() if callable(anchor_getter) else None
        if anchor is not None:
            size = self.size()
            x = anchor.x() - size.width() + 1
            y = anchor.y() + 6
            # G7：screenAt 会返回 None（虚拟屏 / 跨屏边缘），必须退回主屏
            screen = QGuiApplication.screenAt(anchor) or QGuiApplication.primaryScreen()
            if screen is not None:
                available = screen.availableGeometry()
                x = min(max(x, available.left()), available.right() - size.width() + 1)
                y = min(max(y, available.top()), available.bottom() - size.height() + 1)
            self.move(x, y)
            return
        parent = self.parentWidget()
        if parent is not None:
            geo = self.geometry()
            self.move(
                parent.x() + parent.width() - geo.width() - 40,
                parent.y() + 80,
            )

    def _save_geometry(self) -> None:
        # 最小化/最大化时 saveGeometry 会把窗口状态一并存盘，下次启动会
        # 直接以最小化出现（等于丢窗）；与 mark_stats.save_geom 同口径跳过。
        if self.isMinimized() or self.isMaximized():
            return
        from src.core.settings import AppSettings

        AppSettings().set_curve_shelf_geometry(self.saveGeometry())
